#!/usr/bin/env bash
# ==============================================================================
# Vast.ai Kotoba-Whisper v2.2 자막 추출 파이프라인
# ==============================================================================
set -Eeuo pipefail

# ------------------------------------------------------------------------------
# 설정
# ------------------------------------------------------------------------------
DRIVE_URL="${1:-${DRIVE_URL:-https://drive.google.com/drive/folders/1uzD-xr6TC3r6y1mCKyrYUkyzV2oWdkax?usp=drive_link}}"
WORK="${WORKSPACE:-/workspace}"
AUDIO_DIR="${AUDIO_DIR:-${WORK}/audio}"
OUT_DIR="${OUT_DIR:-${WORK}/output}"
OUT_SRT="${OUT_DIR}/audio_files_kotoba.srt"
SETUP_MARK="${WORK}/.setup_kotoba_done"
APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export DEBIAN_FRONTEND=noninteractive
export PIP_DISABLE_PIP_VERSION_CHECK=1
export HF_HOME="${WORK}/.cache/huggingface"
export TORCH_HOME="${WORK}/.cache/torch"
export HF_HUB_DISABLE_TELEMETRY=1
mkdir -p "${AUDIO_DIR}" "${OUT_DIR}" "${HF_HOME}" "${TORCH_HOME}"

# Kotoba 전용 독립 가상환경 설정
KOTOBA_VENV="${WORK}/venv_kotoba"
if [ ! -d "${KOTOBA_VENV}" ]; then
    python3 -m venv "${KOTOBA_VENV}"
fi
source "${KOTOBA_VENV}/bin/activate"
PY="${KOTOBA_VENV}/bin/python"

banner() {
    echo
    echo "=============================================================="
    echo " $*"
    echo "=============================================================="
}

# ------------------------------------------------------------------------------
# 1. 환경 설치 (최초 1회)
# ------------------------------------------------------------------------------
if [ -f "${SETUP_MARK}" ]; then
    banner "1. Kotoba 환경 설치 - 이미 완료됨, 건너뜀 (${SETUP_MARK})"
else
    banner "1-1. 시스템 패키지 설치"
    apt-get update -qq
    apt-get install -y -qq --no-install-recommends git ffmpeg libsndfile1 > /dev/null

    banner "1-2. RTX 5090 / CUDA 호환성 확인"
    rc=0
    "${PY}" - <<'PYCODE' || rc=$?
import sys, torch
print("Python :", sys.version.split()[0])
print("Torch  :", torch.__version__, "| CUDA", torch.version.cuda)
if not torch.cuda.is_available():
    sys.exit("CUDA GPU 를 사용할 수 없습니다.")
cap = torch.cuda.get_device_capability(0)
arch = f"sm_{cap[0]}{cap[1]}"
print("GPU    :", torch.cuda.get_device_name(0), f"({arch})")
if arch not in torch.cuda.get_arch_list():
    print(f"경고: 현재 torch 빌드가 {arch} 를 지원하지 않습니다 → 재설치 필요")
    sys.exit(42)
PYCODE
    if [ "${rc}" -eq 42 ]; then
        echo "▶ RTX 5090(Blackwell) 지원 PyTorch(cu128) 재설치 중..."
        "${PY}" -m pip install -q --upgrade torch torchaudio --index-url https://download.pytorch.org/whl/cu128
    elif [ "${rc}" -ne 0 ]; then
        exit "${rc}"
    fi

    banner "1-3. Kotoba-Whisper 의존성 설치"
    "${PY}" -m pip install -q --upgrade pip wheel
    "${PY}" -m pip install -q gdown huggingface_hub soundfile librosa noisereduce faster-whisper

    banner "1-4. 임포트 검증"
    "${PY}" - <<'PYCODE'
import torch, faster_whisper
print("Torch :", torch.__version__, "| faster-whisper OK")
PYCODE

    touch "${SETUP_MARK}"
    echo "✅ Kotoba 환경 설치 완료"
fi

# ------------------------------------------------------------------------------
# 2. 구글 드라이브 폴더 다운로드
# ------------------------------------------------------------------------------
if [ "${SKIP_DOWNLOAD:-0}" = "1" ]; then
    banner "2. 다운로드 건너뜀 (SKIP_DOWNLOAD=1) → ${AUDIO_DIR}"
else
    banner "2. 구글 드라이브 폴더 다운로드"
    echo "  링크 : ${DRIVE_URL}"
    echo "  저장 : ${AUDIO_DIR}"
    rm -rf "${AUDIO_DIR:?}"/*
    "${PY}" -m gdown --folder "${DRIVE_URL}" -O "${AUDIO_DIR}"
fi

if [ ! -f "${AUDIO_DIR}/References.txt" ]; then
    found="$(find "${AUDIO_DIR}" -maxdepth 3 -name References.txt | head -n 1 || true)"
    if [ -n "${found}" ]; then
        mv "$(dirname "${found}")"/* "${AUDIO_DIR}/" 2>/dev/null || true
    fi
fi
if [ ! -f "${AUDIO_DIR}/References.txt" ]; then
    echo "❌ References.txt 를 찾을 수 없습니다: ${AUDIO_DIR}"
    exit 1
fi
echo "  🎧 wav 파일 수: $(find "${AUDIO_DIR}" -maxdepth 1 -name '*.wav' | wc -l)"

# ------------------------------------------------------------------------------
# 3. 자막 추출 실행
# ------------------------------------------------------------------------------
banner "3. Kotoba-Whisper 자막 추출 시작"
EXTRA=""
if [ "${NO_DENOISE:-0}" = "1" ]; then
    EXTRA="--no-denoise"
fi

# transcribe_kotoba.py 실행
"${PY}" "${APP_DIR}/transcribe_kotoba.py" --audio-dir "${AUDIO_DIR}" --out "${OUT_SRT}" ${EXTRA}

# ------------------------------------------------------------------------------
# 4. 결과 안내
# ------------------------------------------------------------------------------
banner "4. 완료"
echo "  📁 결과 파일: ${OUT_SRT}  ($(wc -l < "${OUT_SRT}") 줄)"
