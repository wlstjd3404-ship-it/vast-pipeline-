import os
import gc
import sys
import argparse
import subprocess
import gdown
import torch
import numpy as np
import librosa
import soundfile as sf
import noisereduce as nr

from espnet2.bin.asr_inference import Speech2Text
from reazonspeech.espnet.asr import transcribe, audio_from_path
import reazonspeech.espnet.asr.ctc as _ctc

parser = argparse.ArgumentParser()
parser.add_argument("--drive-url", type=str, required=True)
parser.add_argument("--beam-size", type=int, default=10)
args = parser.parse_args()

WORK_DIR   = "/workspace"
AUDIO_DIR  = os.path.join(WORK_DIR, "audio")
OUTPUT_DIR = os.path.join(WORK_DIR, "output_audio")
REF_FILE   = os.path.join(AUDIO_DIR, "References.txt")
OUTPUT_SRT = os.path.join(OUTPUT_DIR, "audio_files.srt")

print("▶ 1단계: 공개 URL에서 audio 폴더 다운로드", flush=True)
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
gdown.download_folder(url=args.drive_url, output=AUDIO_DIR, quiet=False, use_cookies=False)

if not os.path.exists(REF_FILE):
    raise FileNotFoundError(f"References.txt 없음: {REF_FILE}")

with open(REF_FILE, "r", encoding="utf-8") as f:
    filenames = [line.strip() for line in f if line.strip()]
audio_files = [os.path.join(AUDIO_DIR, name) for name in filenames]

print(f"▶ 2단계: 모델 로드 (beam_size={args.beam_size})", flush=True)

def _patched_ctc_decode(model, samples):
    speech = torch.tensor(samples, dtype=torch.bfloat16).to(model.device).unsqueeze(0)
    length = torch.tensor([len(samples)]).to(model.device)
    enc = model.asr_model.encode(speech, length)[0]
    lpz = model.asr_model.ctc.softmax(enc)
    return lpz.detach().float().squeeze(0).cpu().numpy()

_ctc.ctc_decode = _patched_ctc_decode
for _name, _mod in list(sys.modules.items()):
    if _name.startswith("reazonspeech") and _mod is not None:
        if getattr(_mod, "ctc_decode", None) is not None and _mod is not _ctc:
            setattr(_mod, "ctc_decode", _patched_ctc_decode)

speech2text = Speech2Text.from_pretrained(
    "reazon-research/reazonspeech-espnet-v2",
    beam_size        = args.beam_size,
    ctc_weight       = 0.4,
    lm_weight        = 0.6,
    normalize_length = True,
    dtype            = "bfloat16",
    nbest            = 10,
    device           = "cuda",
)

def format_time_srt(seconds):
    h  = int(seconds // 3600)
    m  = int((seconds % 3600) // 60)
    s  = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

print("▶ 3단계: 자막 추출", flush=True)
global_subtitle_index  = 1
cumulative_time_offset = 0.0

with open(OUTPUT_SRT, "w", encoding="utf-8") as f:
    for i, audio_file in enumerate(audio_files, 1):
        if not os.path.exists(audio_file):
            continue

        file_duration = librosa.get_duration(path=audio_file)
        base, _ = os.path.splitext(audio_file)
        dn_path = base + "__dn.wav"

        data, rate = librosa.load(audio_file, sr=16000, mono=True)
        reduced = nr.reduce_noise(y=data, sr=rate, stationary=True, prop_decrease=0.8)
        sf.write(dn_path, reduced, rate)
        del data, reduced
        gc.collect()

        try:
            audio  = audio_from_path(dn_path)
            result = transcribe(speech2text, audio)
            for seg in result.segments:
                text = seg.text.strip()
                if not text: continue
                start = cumulative_time_offset + seg.start_seconds
                end   = cumulative_time_offset + seg.end_seconds
                f.write(f"{global_subtitle_index}\n")
                f.write(f"{format_time_srt(start)} --> {format_time_srt(end)}\n")
                f.write(f"{text}\n\n")
                global_subtitle_index += 1
            f.flush()
        finally:
            if os.path.exists(dn_path):
                os.remove(dn_path)

        cumulative_time_offset += file_duration
        torch.cuda.empty_cache()
        gc.collect()

print(f"🎉 완료! 자막 파일 로컬 저장 위치: {OUTPUT_SRT}", flush=True)