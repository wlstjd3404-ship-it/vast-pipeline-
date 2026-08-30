import os
import gc
import sys
import argparse
import subprocess

import torch
import numpy as np
import librosa
import soundfile as sf
import noisereduce as nr

print(f"✅ NumPy 버전: {np.__version__}", flush=True)
if not np.__version__.startswith("1."):
    print("❌ NumPy 2.x 입니다. espnet이 실패합니다. venv를 재설치하세요.", flush=True)
    sys.exit(1)

from espnet2.bin.asr_inference import Speech2Text
from reazonspeech.espnet.asr import transcribe, audio_from_path
import reazonspeech.espnet.asr.ctc as _ctc

parser = argparse.ArgumentParser()
parser.add_argument("--rclone-remote", type=str, default="gdrive")
parser.add_argument("--remote-dir", type=str, default="audio")
parser.add_argument("--beam-size", type=int, default=10)
args = parser.parse_args()

WORK_DIR   = "/workspace"
AUDIO_DIR  = os.path.join(WORK_DIR, "audio")
REF_FILE   = os.path.join(AUDIO_DIR, "References.txt")
OUTPUT_SRT = os.path.join(AUDIO_DIR, "audio_files.srt")
REMOTE     = f"{args.rclone_remote}:{args.remote_dir}"

# ------------------------------------------------------------------ 1
print("▶ 1단계: 드라이브에서 audio 폴더 다운로드", flush=True)
os.makedirs(AUDIO_DIR, exist_ok=True)
subprocess.run(
    ["rclone", "copy", REMOTE, AUDIO_DIR, "--progress", "--stats", "10s"],
    check=True,
)

if not os.path.exists(REF_FILE):
    raise FileNotFoundError(f"References.txt 없음: {REF_FILE}")

with open(REF_FILE, "r", encoding="utf-8") as f:
    filenames = [line.strip() for line in f if line.strip()]
audio_files = [os.path.join(AUDIO_DIR, name) for name in filenames]
print(f"  총 {len(audio_files)}개 파일 로드됨", flush=True)

# ------------------------------------------------------------------ 2
print(f"▶ 2단계: 모델 로드 (beam_size={args.beam_size})", flush=True)

def _patched_ctc_decode(model, samples):
    speech = torch.tensor(samples, dtype=torch.bfloat16).to(model.device).unsqueeze(0)
    length = torch.tensor([len(samples)]).to(model.device)
    enc = model.asr_model.encode(speech, length)[0]
    lpz = model.asr_model.ctc.softmax(enc)
    return lpz.detach().float().squeeze(0).cpu().numpy()

# ★ 원본 코랩과 동일한 지점 패치
_ctc.ctc_decode = _patched_ctc_decode
# ★ from-import 로 이미 이름을 복사해간 모듈까지 모두 교체 (안전망)
_patched_count = 1
for _name, _mod in list(sys.modules.items()):
    if _name.startswith("reazonspeech") and _mod is not None:
        if getattr(_mod, "ctc_decode", None) is not None and _mod is not _ctc:
            setattr(_mod, "ctc_decode", _patched_ctc_decode)
            _patched_count += 1
print(f"  bfloat16 CTC 패치 적용: {_patched_count}곳", flush=True)

speech2text = Speech2Text.from_pretrained(
    "reazon-research/reazonspeech-espnet-v2",
    beam_size        = args.beam_size,
    ctc_weight       = 0.4,
    lm_weight        = 0.6,
    normalize_length = True,
    dtype            = "bfloat16",
    nbest            = 10,               # ★ 원본과 동일
    device           = "cuda",
)
print("✅ 모델 로드 완료", flush=True)

# ------------------------------------------------------------------ 3
def format_time_srt(seconds):
    h  = int(seconds // 3600)
    m  = int((seconds % 3600) // 60)
    s  = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

def upload_srt():
    subprocess.run(
        ["rclone", "copyto", OUTPUT_SRT, f"{REMOTE}/audio_files.srt"],
        check=False,
    )

print("▶ 3단계: 순차 자막 추출 (비파괴 노이즈 제거)", flush=True)
global_subtitle_index  = 1
cumulative_time_offset = 0.0

with open(OUTPUT_SRT, "w", encoding="utf-8") as f:
    for i, audio_file in enumerate(audio_files, 1):
        if not os.path.exists(audio_file):
            # ★ 임의의 600초 보정 금지 → 싱크 붕괴 방지
            print(f"⚠️ [{i}/{len(audio_files)}] 파일 없음, 건너뜀: "
                  f"{os.path.basename(audio_file)}", flush=True)
            continue

        file_duration = librosa.get_duration(path=audio_file)
        print(f"[{i}/{len(audio_files)}] {os.path.basename(audio_file)} "
              f"({file_duration/60:.1f}분) 처리 중...", flush=True)

        # ★ 확장자 무관 안전한 임시 경로 (원본 덮어쓰기 방지)
        base, _ = os.path.splitext(audio_file)
        dn_path = base + "__dn.wav"

        data, rate = librosa.load(audio_file, sr=16000, mono=True)
        reduced = nr.reduce_noise(y=data, sr=rate, stationary=True, prop_decrease=0.8)
        sf.write(dn_path, reduced, rate)
        del data, reduced
        gc.collect()

        seg_count = 0
        try:
            audio  = audio_from_path(dn_path)
            result = transcribe(speech2text, audio)

            for seg in result.segments:
                text = seg.text.strip()
                if not text:
                    continue
                start = cumulative_time_offset + seg.start_seconds
                end   = cumulative_time_offset + seg.end_seconds
                f.write(f"{global_subtitle_index}\n")
                f.write(f"{format_time_srt(start)} --> {format_time_srt(end)}\n")
                f.write(f"{text}\n\n")
                global_subtitle_index += 1
                seg_count += 1
            f.flush()
        finally:
            if os.path.exists(dn_path):
                os.remove(dn_path)

        cumulative_time_offset += file_duration
        print(f"  ✅ 자막 {seg_count}개 | 누적 오프셋 "
              f"{cumulative_time_offset/60:.1f}분", flush=True)

        torch.cuda.empty_cache()
        gc.collect()
        upload_srt()          # 중간 결과 즉시 백업

print("▶ 4단계: 최종 업로드", flush=True)
subprocess.run(["rclone", "copyto", OUTPUT_SRT, f"{REMOTE}/audio_files.srt"], check=True)
print(f"🎉 완료! 총 {global_subtitle_index - 1}개 자막 → {REMOTE}/audio_files.srt", flush=True)
