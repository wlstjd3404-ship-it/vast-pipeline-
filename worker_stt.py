import os
import gc
import argparse
import subprocess
import torch
import numpy as np
import librosa
import soundfile as sf
import noisereduce as nr
from espnet2.bin.asr_inference import Speech2Text
import reazonspeech.espnet.asr.transcribe as rz_transcribe
from reazonspeech.espnet.asr import transcribe, audio_from_path

parser = argparse.ArgumentParser()
parser.add_argument("--rclone-remote", type=str, default="gdrive")
parser.add_argument("--beam-size", type=int, default=10)
args = parser.parse_args()

WORK_DIR = "/workspace"
AUDIO_DIR = os.path.join(WORK_DIR, "audio")
REF_FILE = os.path.join(AUDIO_DIR, "References.txt")
OUTPUT_SRT = os.path.join(AUDIO_DIR, "audio_files.srt")

print("▶ 1단계: 구글 드라이브에서 audio 폴더 다운로드", flush=True)
os.makedirs(AUDIO_DIR, exist_ok=True)
subprocess.run(["rclone", "copy", f"{args.rclone_remote}:audio", AUDIO_DIR], check=True)

if not os.path.exists(REF_FILE):
    raise FileNotFoundError(f"References.txt 없음: {REF_FILE}")

with open(REF_FILE, "r", encoding="utf-8") as f:
    filenames = [line.strip() for line in f if line.strip()]

audio_files = [os.path.join(AUDIO_DIR, name) for name in filenames]

print(f"▶ 2단계: STT 모델 로드 (beam_size={args.beam_size})", flush=True)
def _patched_ctc_decode(model, samples):
    speech = torch.tensor(samples, dtype=torch.bfloat16).to(model.device).unsqueeze(0)
    length = torch.tensor([len(samples)]).to(model.device)
    enc = model.asr_model.encode(speech, length)[0]
    lpz = model.asr_model.ctc.softmax(enc)
    return lpz.detach().float().squeeze(0).cpu().numpy()

# 실제 호출되는 모듈의 ctc_decode를 직접 패치
rz_transcribe.ctc_decode = _patched_ctc_decode

speech2text = Speech2Text.from_pretrained(
    "reazon-research/reazonspeech-espnet-v2",
    beam_size=args.beam_size,
    ctc_weight=0.4,
    lm_weight=0.6,
    normalize_length=True,
    dtype="bfloat16",
    device="cuda",
)

def format_time_srt(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

print("▶ 3단계: 순차 자막 추출 (비파괴 노이즈 제거)", flush=True)
global_subtitle_index = 1
cumulative_time_offset = 0.0

with open(OUTPUT_SRT, "w", encoding="utf-8") as f:
    for i, audio_file in enumerate(audio_files, 1):
        if not os.path.exists(audio_file):
            print(f"⚠️ [{i}/{len(audio_files)}] 파일 없음: {audio_file}", flush=True)
            # 10분 단위 누적 시간 보정
            cumulative_time_offset += 600.0
            continue

        file_duration = librosa.get_duration(path=audio_file)
        print(f"[{i}/{len(audio_files)}] 처리 중 ({file_duration/60:.1f}분): {os.path.basename(audio_file)}", flush=True)

        dn_path = audio_file.replace(".wav", "_dn.wav")
        data, rate = librosa.load(audio_file, sr=16000, mono=True)
        reduced = nr.reduce_noise(y=data, sr=rate, stationary=True, prop_decrease=0.8)
        sf.write(dn_path, reduced, rate)

        audio = audio_from_path(dn_path)
        result = transcribe(speech2text, audio)

        for seg in result.segments:
            text = seg.text.strip()
            if not text:
                continue
            start = cumulative_time_offset + seg.start_seconds
            end = cumulative_time_offset + seg.end_seconds
            f.write(f"{global_subtitle_index}\n")
            f.write(f"{format_time_srt(start)} --> {format_time_srt(end)}\n")
            f.write(f"{text}\n\n")
            global_subtitle_index += 1

        cumulative_time_offset += file_duration
        if os.path.exists(dn_path):
            os.remove(dn_path)

        torch.cuda.empty_cache()
        gc.collect()

print("▶ 4단계: 구글 드라이브 업로드", flush=True)
subprocess.run(["rclone", "copyto", OUTPUT_SRT, f"{args.rclone_remote}:audio/audio_files.srt"], check=True)
print("✔ 자막 추출 및 구글 드라이브 동기화 완료!", flush=True)