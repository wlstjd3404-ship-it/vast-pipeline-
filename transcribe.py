#!/usr/bin/env python3
# ==============================================================================
# ReazonSpeech(ESPnet) 일본어 자막 추출 - Vast.ai 터미널 실행용
#
#   구글 코랩 노트북 [셀 2] 를 그대로 스크립트로 옮긴 것.
#   (노이즈 제거 → 모델 로드(bfloat16) → STT → 하나의 SRT 로 병합)
#
# 사용법:
#   python transcribe.py --audio-dir /workspace/audio --out /workspace/output/audio_files.srt
#
#   --audio-dir  : References.txt 와 wav 파일들이 들어있는 폴더
#   --out        : 결과 SRT 경로 (생략시 <audio-dir>/audio_files.srt)
#   --no-denoise : 노이즈 제거 건너뛰기
#   --beam-size  : 빔 크기 (기본 10)
# ==============================================================================

import os
import gc
import sys
import argparse

import torch
import numpy as np
import librosa
import soundfile as sf
import noisereduce as nr


# ------------------------------------------------------------------------------
# 인자 처리
# ------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="ReazonSpeech 일본어 자막(SRT) 추출")
parser.add_argument("--audio-dir", default=os.environ.get("AUDIO_DIR", "/workspace/audio"))
parser.add_argument("--out",       default=None)
parser.add_argument("--no-denoise", action="store_true")
parser.add_argument("--beam-size", type=int, default=10)
args = parser.parse_args()

AUDIO_DIR  = args.audio_dir
REF_FILE   = os.path.join(AUDIO_DIR, "References.txt")
OUTPUT_SRT = args.out or os.path.join(AUDIO_DIR, "audio_files.srt")
os.makedirs(os.path.dirname(os.path.abspath(OUTPUT_SRT)), exist_ok=True)

print(f"✅ NumPy {np.__version__} | PyTorch {torch.__version__} | CUDA {torch.version.cuda}")
if not torch.cuda.is_available():
    sys.exit("❌ CUDA GPU 를 사용할 수 없습니다. 인스턴스/드라이버를 확인하세요.")
print(f"✅ GPU: {torch.cuda.get_device_name(0)}")

# ReazonSpeech 및 ESPnet 모듈 로드
from reazonspeech.espnet.asr import transcribe, audio_from_path
from espnet2.bin.asr_inference import Speech2Text
import reazonspeech.espnet.asr.ctc as _ctc

# ------------------------------------------------------------------------------
# 파일 목록 읽기
# ------------------------------------------------------------------------------
if not os.path.exists(REF_FILE):
    sys.exit(f"❌ References.txt 를 찾을 수 없습니다: {REF_FILE}")

with open(REF_FILE, "r", encoding="utf-8") as f:
    filenames = [line.strip() for line in f if line.strip()]

audio_files = [os.path.join(AUDIO_DIR, name) for name in filenames]
print(f"총 {len(audio_files)}개 파일 로드됨 (폴더: {AUDIO_DIR})")

# ------------------------------------------------------------------------------
# 1. 노이즈 제거 프로세스
# ------------------------------------------------------------------------------
if args.no_denoise:
    print("\n▶ 노이즈 제거 건너뜀 (--no-denoise)")
else:
    print("\n▶ 노이즈 제거 시작...")
    for i, audio_path in enumerate(audio_files, 1):
        if not os.path.exists(audio_path):
            print(f"  ⚠ [{i}/{len(audio_files)}] 파일 없음, 건너뜀: {os.path.basename(audio_path)}")
            continue

        print(f"  [{i}/{len(audio_files)}] {os.path.basename(audio_path)} 처리 중...")
        data, rate = librosa.load(audio_path, sr=None)
        reduced = nr.reduce_noise(y=data, sr=rate, stationary=True, prop_decrease=0.8)
        sf.write(audio_path, reduced, rate)

    try:
        del data, reduced
    except NameError:
        pass
    gc.collect()
    print("✅ 전체 노이즈 제거 완료")

# ------------------------------------------------------------------------------
# 2. STT 모델 로드 및 bfloat16 패치 적용
# ------------------------------------------------------------------------------
print("\n▶ STT 모델 로딩 중...")
speech2text = Speech2Text.from_pretrained(
    "reazon-research/reazonspeech-espnet-v2",
    beam_size        = args.beam_size,  # 기본 20 → 10
    ctc_weight       = 0.4,
    lm_weight        = 0.6,
    # STFT/cuFFT는 BFloat16 입력을 지원하지 않으므로 전체 추론을 float32로 실행
    dtype            = "float32",
    nbest            = 10,
    device           = "cuda",
)
# ESPnet 202609부터 normalize_length가 Speech2Text 생성자에서 제거됨.
# 동일 동작이 필요하면 생성 후 beam_search에 직접 적용한다.
if hasattr(speech2text, "beam_search"):
    speech2text.beam_search.normalize_length = True

def _patched_ctc_decode(model, samples):
    # STFT/cuFFT는 float32 입력을 요구한다. 출력만 float32 NumPy로 반환한다.
    speech = torch.as_tensor(samples, dtype=torch.float32, device=model.device).unsqueeze(0)
    length = torch.tensor([len(samples)], dtype=torch.long, device=model.device)
    with torch.inference_mode():
        enc = model.asr_model.encode(speech, length)[0]
        lpz = model.asr_model.ctc.softmax(enc)
    return lpz.detach().float().squeeze(0).cpu().numpy()

_ctc.ctc_decode = _patched_ctc_decode
print("✅ 모델 로드 완료")

# ------------------------------------------------------------------------------
# 3. STT 실행 및 SRT 자막 생성
# ------------------------------------------------------------------------------
def format_time_srt(seconds):
    h  = int(seconds // 3600)
    m  = int((seconds % 3600) // 60)
    s  = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

print("\n▶ STT 시작...")
global_subtitle_index  = 1
cumulative_time_offset = 0.0

with open(OUTPUT_SRT, "w", encoding="utf-8") as f:
    for i, audio_file in enumerate(audio_files, 1):
        if not os.path.exists(audio_file):
            print(f"  ⚠ [{i}/{len(audio_files)}] 파일 없음, 건너뜀")
            continue

        print(f"\n[{i}/{len(audio_files)}] {os.path.basename(audio_file)} 추출 시작...")
        # librosa 버전에 따라 path/filename 인자가 달라지므로 soundfile 사용
        file_duration = sf.info(audio_file).duration
        print(f"  파일 길이: {file_duration/60:.1f}분")

        audio = audio_from_path(audio_file)
        result = transcribe(speech2text, audio)

        seg_count = 0
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

        cumulative_time_offset += file_duration
        print(f"  ✅ 완료 | 자막 {seg_count}개 | 누적 오프셋: {cumulative_time_offset/60:.1f}분")
        f.flush()

        torch.cuda.empty_cache()
        gc.collect()

print(f"\n🎉 전체 완료! 총 {global_subtitle_index - 1}개 자막")
print(f"📁 저장: {OUTPUT_SRT}")
