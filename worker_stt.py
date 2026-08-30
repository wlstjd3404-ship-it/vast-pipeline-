import os

# ★★ torch 2.6+ 부터 torch.load 기본값이 weights_only=True 로 바뀌므로
#    espnet 체크포인트 로드를 위해 torch import 이전에 반드시 설정해야 한다.
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
os.environ["TORCH_FORCE_WEIGHTS_ONLY_LOAD"]    = "0"

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

# ★★ 하드코딩된 bfloat16 제거 → 모델 파라미터 dtype 을 그대로 따라간다 (CPU 폴백 대응)
def _patched_ctc_decode(model, samples):
    param  = next(model.asr_model.parameters())
    speech = torch.tensor(samples, dtype=param.dtype).to(model.device).unsqueeze(0)
    length = torch.tensor([len(samples)]).to(model.device)
    enc = model.asr_model.encode(speech, length)[0]
    lpz = model.asr_model.ctc.softmax(enc)
    return lpz.detach().float().squeeze(0).cpu().numpy()

_ctc.ctc_decode = _patched_ctc_decode
for _name, _mod in list(sys.modules.items()):
    if _name.startswith("reazonspeech") and _mod is not None:
        if getattr(_mod, "ctc_decode", None) is not None and _mod is not _ctc:
            setattr(_mod, "ctc_decode", _patched_ctc_decode)


# ★★ GPU 아키텍처(sm_120 등) 지원 여부를 실제 커널 실행으로 사전 검증
def cuda_usable():
    if not torch.cuda.is_available():
        print("  ⚠ torch.cuda.is_available() == False", flush=True)
        return False
    cap   = torch.cuda.get_device_capability()
    mine  = f"sm_{cap[0]}{cap[1]}"
    archs = torch.cuda.get_arch_list()
    print(f"  GPU={torch.cuda.get_device_name(0)} cap={mine}", flush=True)
    print(f"  torch={torch.__version__} archs={archs}", flush=True)
    if not any(a.startswith(mine) for a in archs):
        print(f"  ⚠ 현재 torch 빌드에 {mine} 커널이 없습니다 "
              f"(setup.sh 재실행으로 cu128 휠 설치 필요)", flush=True)
        return False
    try:
        x = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
        (x @ x).float().sum().item()
    except Exception as e:
        print(f"  ⚠ CUDA 커널 실행 실패: {e}", flush=True)
        return False
    return True


def build_model(device, dtype):
    return Speech2Text.from_pretrained(
        "reazon-research/reazonspeech-espnet-v2",
        beam_size        = args.beam_size,
        ctc_weight       = 0.4,
        lm_weight        = 0.6,
        normalize_length = True,
        dtype            = dtype,
        nbest            = 10,
        device           = device,
    )


if cuda_usable():
    try:
        speech2text = build_model("cuda", "bfloat16")
        print("  ✔ CUDA / bfloat16 모드", flush=True)
    except Exception as e:
        print(f"  ⚠ CUDA 모델 로드 실패({e}) → CPU / float32 폴백", flush=True)
        torch.cuda.empty_cache()
        speech2text = build_model("cpu", "float32")
else:
    print("  ▶ CPU / float32 모드로 진행합니다 (속도 매우 느림)", flush=True)
    speech2text = build_model("cpu", "float32")


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
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

print(f"🎉 완료! 자막 파일 로컬 저장 위치: {OUTPUT_SRT}", flush=True)
