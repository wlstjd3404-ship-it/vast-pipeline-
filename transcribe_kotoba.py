#!/usr/bin/env python3
# ==============================================================================
# Kotoba-Whisper v2.2 일본어 자막 추출 - Vast.ai 터미널 실행용
# ==============================================================================

import os
import gc
import sys
import argparse
import torch
import soundfile as sf
import noisereduce as nr
import librosa
from faster_whisper import WhisperModel

parser = argparse.ArgumentParser(description="Kotoba-Whisper 일본어 자막(SRT) 추출")
parser.add_argument("--audio-dir", default=os.environ.get("AUDIO_DIR", "/workspace/audio"))
parser.add_argument("--out",       default=None)
parser.add_argument("--no-denoise", action="store_true")
parser.add_argument("--beam-size", type=int, default=5)
parser.add_argument("--model-id", type=str, default="kotoba-tech/kotoba-whisper-v2.0-faster")
args = parser.parse_args()

AUDIO_DIR  = args.audio_dir
REF_FILE   = os.path.join(AUDIO_DIR, "References.txt")
OUTPUT_SRT = args.out or os.path.join(AUDIO_DIR, "audio_files_kotoba.srt")
os.makedirs(os.path.dirname(os.path.abspath(OUTPUT_SRT)), exist_ok=True)

if not torch.cuda.is_available():
    sys.exit("❌ CUDA GPU 를 사용할 수 없습니다.")
print(f"✅ GPU: {torch.cuda.get_device_name(0)}")

# 1. 파일 목록 읽기
if not os.path.exists(REF_FILE):
    sys.exit(f"❌ References.txt 를 찾을 수 없습니다: {REF_FILE}")

with open(REF_FILE, "r", encoding="utf-8") as f:
    filenames = [line.strip() for line in f if line.strip()]

audio_files = [os.path.join(AUDIO_DIR, name) for name in filenames]
print(f"총 {len(audio_files)}개 파일 로드됨 (폴더: {AUDIO_DIR})")

# 2. 노이즈 제거
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

# 3. 모델 로드
print(f"\n▶ Kotoba-Whisper 모델 로딩 중 ({args.model_id})...")
model = WhisperModel(args.model_id, device="cuda", compute_type="float16")
print("✅ 모델 로드 완료")

# 4. SRT 시간 포맷팅 함수
def format_time_srt(seconds: float) -> str:
    h  = int(seconds // 3600)
    m  = int((seconds % 3600) // 60)
    s  = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    if ms >= 1000:
        s += 1
        ms = 0
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

# 5. STT 실행 및 단일 SRT 작성
print("\n▶ STT 시작...")
global_subtitle_index  = 1
cumulative_time_offset = 0.0

with open(OUTPUT_SRT, "w", encoding="utf-8") as f:
    for i, audio_file in enumerate(audio_files, 1):
        if not os.path.exists(audio_file):
            print(f"  ⚠ [{i}/{len(audio_files)}] 파일 없음, 건너뜀")
            continue

        file_duration = sf.info(audio_file).duration
        print(f"\n[{i}/{len(audio_files)}] {os.path.basename(audio_file)} (길이: {file_duration/60:.1f}분) 추출 중...")

        segments, _ = model.transcribe(
            audio_file,
            language="ja",
            task="transcribe",
            beam_size=args.beam_size,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
            word_timestamps=False
        )

        seg_count = 0
        for seg in segments:
            text = seg.text.strip()
            if not text:
                continue

            start = cumulative_time_offset + seg.start
            end   = cumulative_time_offset + seg.end

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
