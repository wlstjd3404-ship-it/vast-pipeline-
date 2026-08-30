import gc
import os
import shutil
import threading
import traceback
import uuid
from pathlib import Path

import gdown
import gradio as gr
import librosa
import noisereduce as nr
import numpy as np
import soundfile as sf
import torch

from espnet2.bin.asr_inference import Speech2Text
from reazonspeech.espnet.asr import audio_from_path, transcribe
import reazonspeech.espnet.asr.ctc as _ctc


# =============================================================================
# 기본 설정
# =============================================================================

DEFAULT_DRIVE_URL = (
    "https://drive.google.com/drive/folders/"
    "1uzD-xr6TC3r6y1mCKyrYUkyzV2oWdkax?usp=drive_link"
)

WORKSPACE = Path("/workspace")
JOBS_DIR = WORKSPACE / "jobs"
JOBS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "reazon-research/reazonspeech-espnet-v2"

MODEL_LOCK = threading.Lock()
PROCESS_LOCK = threading.Lock()
speech2text = None


# =============================================================================
# 유틸리티
# =============================================================================

def add_log(logs: list[str], message: str) -> str:
    print(message, flush=True)
    logs.append(message)
    return "\n".join(logs)


def format_time_srt(seconds: float) -> str:
    seconds = max(0.0, float(seconds))

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    milliseconds = int(round((seconds - int(seconds)) * 1000))

    if milliseconds >= 1000:
        milliseconds = 0
        secs += 1

        if secs >= 60:
            secs = 0
            minutes += 1

        if minutes >= 60:
            minutes = 0
            hours += 1

    return f"{hours:02}:{minutes:02}:{secs:02},{milliseconds:03}"


def clear_gpu_memory():
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def find_reference_file(download_dir: Path) -> Path:
    matches = sorted(download_dir.rglob("References.txt"))

    if not matches:
        matches = sorted(download_dir.rglob("references.txt"))

    if not matches:
        raise FileNotFoundError(
            "Google Drive 폴더에서 References.txt를 찾지 못했습니다."
        )

    return matches[0]


def read_audio_file_list(reference_file: Path) -> list[Path]:
    base_dir = reference_file.parent

    with reference_file.open("r", encoding="utf-8-sig") as file:
        names = [
            line.strip().strip('"').strip("'")
            for line in file
            if line.strip()
        ]

    if not names:
        raise ValueError("References.txt에 파일 이름이 없습니다.")

    audio_files = []

    for name in names:
        direct_path = base_dir / name

        if direct_path.is_file():
            audio_files.append(direct_path)
            continue

        # 하위 폴더에 파일이 있을 경우 파일명으로 한 번 더 검색
        basename = Path(name).name
        matches = [
            path
            for path in base_dir.rglob(basename)
            if path.is_file()
        ]

        if matches:
            audio_files.append(matches[0])
        else:
            audio_files.append(direct_path)

    return audio_files


def denoise_audio(
    source_path: Path,
    output_dir: Path,
    index: int,
    prop_decrease: float,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    # MP3 등에 soundfile을 직접 덮어쓸 수 없기 때문에 항상 WAV로 저장
    output_path = output_dir / f"{index:04d}_{source_path.stem}.wav"

    data, sample_rate = librosa.load(
        str(source_path),
        sr=None,
        mono=True,
    )

    reduced = nr.reduce_noise(
        y=data,
        sr=sample_rate,
        stationary=True,
        prop_decrease=prop_decrease,
    )

    sf.write(
        str(output_path),
        reduced,
        sample_rate,
        subtype="PCM_16",
    )

    del data
    del reduced
    gc.collect()

    return output_path


# =============================================================================
# 모델
# =============================================================================

def patched_ctc_decode(model, samples):
    speech = (
        torch.tensor(samples, dtype=torch.bfloat16)
        .to(model.device)
        .unsqueeze(0)
    )

    length = torch.tensor(
        [len(samples)],
        dtype=torch.long,
        device=model.device,
    )

    encoded = model.asr_model.encode(speech, length)[0]
    probabilities = model.asr_model.ctc.softmax(encoded)

    return (
        probabilities
        .detach()
        .float()
        .squeeze(0)
        .cpu()
        .numpy()
    )


def load_model(hf_token: str | None = None):
    global speech2text

    if speech2text is not None:
        return speech2text

    with MODEL_LOCK:
        if speech2text is not None:
            return speech2text

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA GPU를 사용할 수 없습니다. "
                "Vast.ai 인스턴스와 PyTorch CUDA 설정을 확인하세요."
            )

        if hf_token and hf_token.strip():
            token = hf_token.strip()
            os.environ["HF_TOKEN"] = token
            os.environ["HUGGING_FACE_HUB_TOKEN"] = token

        speech2text = Speech2Text.from_pretrained(
            MODEL_NAME,
            beam_size=10,
            ctc_weight=0.4,
            lm_weight=0.6,
            normalize_length=True,
            dtype="bfloat16",
            nbest=10,
            device="cuda",
        )

        _ctc.ctc_decode = patched_ctc_decode
        return speech2text


# =============================================================================
# 자막 추출
# =============================================================================

def extract_subtitles(
    drive_url: str,
    hf_token: str,
    use_noise_reduction: bool,
    prop_decrease: float,
    progress=gr.Progress(track_tqdm=False),
):
    logs = []
    output_srt = None

    if not drive_url or not drive_url.strip():
        return None, "Google Drive 폴더 URL을 입력하세요."

    # 모델을 동시에 두 번 실행하면 VRAM 부족이 발생할 수 있으므로 작업 1개로 제한
    if not PROCESS_LOCK.acquire(blocking=False):
        return None, "다른 자막 추출 작업이 실행 중입니다. 완료 후 다시 시도하세요."

    job_id = uuid.uuid4().hex
    job_dir = JOBS_DIR / job_id
    download_dir = job_dir / "drive"
    processed_dir = job_dir / "processed"
    output_srt = job_dir / "audio_files.srt"

    try:
        job_dir.mkdir(parents=True, exist_ok=True)
        download_dir.mkdir(parents=True, exist_ok=True)

        progress(0.02, desc="Google Drive 폴더 다운로드 중")
        add_log(logs, "▶ Google Drive 폴더 다운로드 시작")
        add_log(logs, f"URL: {drive_url.strip()}")

        result = gdown.download_folder(
            url=drive_url.strip(),
            output=str(download_dir),
            quiet=False,
            use_cookies=False,
            remaining_ok=True,
        )

        if not result:
            raise RuntimeError(
                "Google Drive 폴더를 다운로드하지 못했습니다. "
                "폴더 공유 권한을 '링크가 있는 모든 사용자'로 설정하세요."
            )

        reference_file = find_reference_file(download_dir)
        audio_files = read_audio_file_list(reference_file)

        add_log(logs, f"✅ References.txt: {reference_file}")
        add_log(logs, f"✅ 총 {len(audio_files)}개 파일 로드됨")

        missing_files = [
            str(path)
            for path in audio_files
            if not path.is_file()
        ]

        if missing_files:
            missing_text = "\n".join(missing_files)
            raise FileNotFoundError(
                "다음 오디오 파일을 찾지 못했습니다:\n"
                f"{missing_text}"
            )

        process_files = []

        if use_noise_reduction:
            add_log(logs, "\n▶ 노이즈 제거 시작")

            for index, source_path in enumerate(audio_files, start=1):
                progress_value = 0.05 + (index / len(audio_files)) * 0.25
                progress(
                    progress_value,
                    desc=f"노이즈 제거 {index}/{len(audio_files)}",
                )

                add_log(
                    logs,
                    f"[{index}/{len(audio_files)}] "
                    f"{source_path.name} 노이즈 제거 중",
                )

                processed_path = denoise_audio(
                    source_path=source_path,
                    output_dir=processed_dir,
                    index=index,
                    prop_decrease=float(prop_decrease),
                )

                process_files.append(processed_path)

            add_log(logs, "✅ 전체 노이즈 제거 완료")
        else:
            process_files = audio_files
            add_log(logs, "ℹ️ 노이즈 제거를 사용하지 않습니다.")

        progress(0.32, desc="STT 모델 로딩 중")
        add_log(logs, "\n▶ ReazonSpeech 모델 로딩 중")

        model = load_model(hf_token)

        add_log(logs, "✅ 모델 로드 완료")
        add_log(
            logs,
            f"GPU: {torch.cuda.get_device_name(0)}",
        )
        add_log(
            logs,
            f"PyTorch CUDA: {torch.version.cuda}",
        )

        progress(0.38, desc="자막 추출 준비")
        add_log(logs, "\n▶ STT 및 SRT 생성 시작")

        global_subtitle_index = 1
        cumulative_time_offset = 0.0

        with output_srt.open("w", encoding="utf-8") as srt_file:
            for index, audio_file in enumerate(process_files, start=1):
                original_file = audio_files[index - 1]

                progress_value = 0.38 + (index / len(process_files)) * 0.60
                progress(
                    min(progress_value, 0.98),
                    desc=f"자막 추출 {index}/{len(process_files)}",
                )

                add_log(
                    logs,
                    f"\n[{index}/{len(process_files)}] "
                    f"{original_file.name} 추출 시작",
                )

                file_duration = librosa.get_duration(
                    path=str(audio_file)
                )

                add_log(
                    logs,
                    f"파일 길이: {file_duration / 60:.1f}분",
                )

                audio = audio_from_path(str(audio_file))
                result = transcribe(model, audio)

                segment_count = 0

                for segment in result.segments:
                    text = segment.text.strip()

                    if not text:
                        continue

                    start_time = (
                        cumulative_time_offset
                        + segment.start_seconds
                    )
                    end_time = (
                        cumulative_time_offset
                        + segment.end_seconds
                    )

                    srt_file.write(f"{global_subtitle_index}\n")
                    srt_file.write(
                        f"{format_time_srt(start_time)} --> "
                        f"{format_time_srt(end_time)}\n"
                    )
                    srt_file.write(f"{text}\n\n")

                    global_subtitle_index += 1
                    segment_count += 1

                cumulative_time_offset += file_duration

                add_log(
                    logs,
                    f"✅ 완료 | 자막 {segment_count}개 | "
                    f"누적 {cumulative_time_offset / 60:.1f}분",
                )

                del audio
                del result
                clear_gpu_memory()

        total_subtitles = global_subtitle_index - 1

        progress(1.0, desc="완료")

        add_log(logs, "\n========================================")
        add_log(logs, f"🎉 전체 완료: 총 {total_subtitles}개 자막")
        add_log(logs, f"📁 결과 파일: {output_srt}")
        add_log(logs, "아래 파일 버튼을 눌러 SRT를 다운로드하세요.")

        return str(output_srt), "\n".join(logs)

    except Exception as error:
        error_detail = traceback.format_exc()
        print(error_detail, flush=True)

        add_log(logs, "\n❌ 작업 중 오류가 발생했습니다.")
        add_log(logs, str(error))

        return None, "\n".join(logs)

    finally:
        clear_gpu_memory()
        PROCESS_LOCK.release()


# =============================================================================
# 웹 인터페이스
# =============================================================================

with gr.Blocks(
    title="ReazonSpeech 자막 추출",
    delete_cache=(86400, 86400),
) as demo:
    gr.Markdown(
        """
# ReazonSpeech 일본어 자막 추출

Google Drive 공개 폴더의 오디오 파일을 다운로드하여 하나의 SRT로 만듭니다.

Google Drive 폴더에는 다음 파일이 있어야 합니다.

```text
References.txt
audio001.wav
audio002.wav
...
