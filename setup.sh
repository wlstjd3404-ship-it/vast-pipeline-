#!/usr/bin/env bash
set -Eeuo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${APP_DIR}/.venv"
LOG_FILE="/workspace/subtitle-web.log"
PID_FILE="/workspace/subtitle-web.pid"

export DEBIAN_FRONTEND=noninteractive
export PIP_DISABLE_PIP_VERSION_CHECK=1
export HF_HOME="/workspace/.cache/huggingface"
export TORCH_HOME="/workspace/.cache/torch"
export XDG_CACHE_HOME="/workspace/.cache"

echo "========================================"
echo " 시스템 패키지 설치"
echo "========================================"

apt-get update
apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    curl \
    ca-certificates \
    build-essential \
    libsndfile1 \
    libsndfile1-dev \
    python3-dev \
    python3-venv

# Python 3.13에서는 NumPy 1.x 및 일부 ESPnet 패키지가 동작하지 않으므로
# 3.10~3.12 버전을 사용한다.
PYTHON_BIN=""

for candidate in python3.12 python3.11 python3.10 python3; do
    if command -v "${candidate}" >/dev/null 2>&1; then
        if "${candidate}" - <<'PY' >/dev/null 2>&1
import sys
assert (3, 10) <= sys.version_info[:2] <= (3, 12)
PY
        then
            PYTHON_BIN="$(command -v "${candidate}")"
            break
        fi
    fi
done

if [ -z "${PYTHON_BIN}" ]; then
    echo "Python 3.10 설치 중..."
    apt-get install -y python3.10 python3.10-venv python3.10-dev
    PYTHON_BIN="$(command -v python3.10)"
fi

echo "사용할 Python: ${PYTHON_BIN}"
"${PYTHON_BIN}" --version

echo "========================================"
echo " 가상환경 생성"
echo "========================================"

if [ ! -d "${VENV_DIR}" ]; then
    # Vast PyTorch 이미지에 설치된 torch를 재사용한다.
    "${PYTHON_BIN}" -m venv --system-site-packages "${VENV_DIR}"
fi

PYTHON="${VENV_DIR}/bin/python"
PIP="${VENV_DIR}/bin/pip"

"${PYTHON}" -m pip install --upgrade \
    pip \
    setuptools \
    wheel \
    packaging

# ESPnet/ReazonSpeech 호환성을 위해 NumPy 1.x 고정
"${PIP}" install --upgrade "numpy==1.26.4"

"${PIP}" install --upgrade \
    "gradio>=5.0,<7.0" \
    "gdown>=5.2.0" \
    "huggingface_hub>=0.25.0" \
    "librosa>=0.10.2,<0.12" \
    "soundfile>=0.12.1" \
    "noisereduce>=3.0.2"

"${PIP}" install \
    "espnet" \
    "espnet_model_zoo"

"${PIP}" install \
    "git+https://github.com/reazon-research/ReazonSpeech.git#subdirectory=pkg/espnet-asr"

# 현재 Python 환경에서 torch가 없는 경우 CUDA 12.8 버전 설치
if ! "${PYTHON}" -c "import torch" >/dev/null 2>&1; then
    echo "PyTorch CUDA 12.8 설치 중..."
    "${PIP}" install \
        torch \
        torchvision \
        torchaudio \
        --index-url https://download.pytorch.org/whl/cu128
fi

echo "========================================"
echo " 설치 확인"
echo "========================================"

"${PYTHON}" - <<'PY'
import sys
import numpy
import torch

print("Python:", sys.version)
print("NumPy:", numpy.__version__)
print("PyTorch:", torch.__version__)
print("CUDA 사용 가능:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA:", torch.version.cuda)
else:
    raise RuntimeError("CUDA GPU를 사용할 수 없습니다.")
PY

mkdir -p \
    /workspace/jobs \
    /workspace/.cache/huggingface \
    /workspace/.cache/torch

# 기존 웹 서버 종료
if [ -f "${PID_FILE}" ]; then
    OLD_PID="$(cat "${PID_FILE}" || true)"
    if [ -n "${OLD_PID}" ] && kill -0 "${OLD_PID}" 2>/dev/null; then
        kill "${OLD_PID}" || true
        sleep 2
    fi
    rm -f "${PID_FILE}"
fi

echo "========================================"
echo " 웹 서버 시작"
echo "========================================"

cd "${APP_DIR}"

nohup env \
    HF_HOME="${HF_HOME}" \
    TORCH_HOME="${TORCH_HOME}" \
    XDG_CACHE_HOME="${XDG_CACHE_HOME}" \
    GRADIO_SERVER_NAME="0.0.0.0" \
    GRADIO_SERVER_PORT="11111" \
    "${PYTHON}" "${APP_DIR}/app.py" \
    >"${LOG_FILE}" 2>&1 &

APP_PID=$!
echo "${APP_PID}" > "${PID_FILE}"

echo "웹 서버 PID: ${APP_PID}"
echo "로그 파일: ${LOG_FILE}"

for i in $(seq 1 30); do
    if curl -fsS "http://127.0.0.1:11111/" >/dev/null 2>&1; then
        echo "웹 서버가 정상적으로 시작되었습니다."
        echo "Vast.ai의 Open 버튼을 눌러 접속하세요."
        exit 0
    fi

    if ! kill -0 "${APP_PID}" 2>/dev/null; then
        echo "웹 서버 실행에 실패했습니다."
        tail -n 100 "${LOG_FILE}" || true
        exit 1
    fi

    sleep 2
done

echo "서버는 실행 중이지만 준비 시간이 오래 걸리고 있습니다."
echo "다음 명령으로 로그를 확인하세요:"
echo "tail -f ${LOG_FILE}"
