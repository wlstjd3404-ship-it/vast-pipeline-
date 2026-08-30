#!/usr/bin/env bash
set -Eeuo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${APP_DIR}/.venv"

export DEBIAN_FRONTEND=noninteractive
export PIP_DISABLE_PIP_VERSION_CHECK=1
export HF_HOME="/workspace/.cache/huggingface"
export TORCH_HOME="/workspace/.cache/torch"
export XDG_CACHE_HOME="/workspace/.cache"

echo "============================================================"
echo " 1. 시스템 패키지 설치"
echo "============================================================"

apt-get update

apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    curl \
    wget \
    ca-certificates \
    build-essential \
    libsndfile1 \
    libsndfile1-dev \
    python3-dev \
    python3-venv
    procps

echo
echo "============================================================"
echo " 2. 호환되는 Python 확인"
echo "============================================================"

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
    echo "호환되는 Python이 없어 Python 3.10을 설치합니다."

    apt-get install -y \
        python3.10 \
        python3.10-venv \
        python3.10-dev

    PYTHON_BIN="$(command -v python3.10)"
fi

echo "사용할 Python:"
"${PYTHON_BIN}" --version

echo
echo "============================================================"
echo " 3. Python 가상환경 생성"
echo "============================================================"

if [ ! -d "${VENV_DIR}" ]; then
    "${PYTHON_BIN}" -m venv \
        --system-site-packages \
        "${VENV_DIR}"
fi

PYTHON="${VENV_DIR}/bin/python"
PIP="${VENV_DIR}/bin/pip"

"${PYTHON}" -m pip install --upgrade \
    pip \
    setuptools \
    wheel \
    packaging

echo
echo "============================================================"
echo " 4. Python 패키지 설치"
echo "============================================================"

"${PIP}" install --upgrade \
    "numpy==1.26.4" \
    "scipy<1.14"

"${PIP}" install --upgrade \
    "gradio>=5.0,<7.0" \
    "gdown>=5.2.0" \
    "huggingface_hub>=0.25.0" \
    "librosa>=0.10.2,<0.12" \
    "soundfile>=0.12.1" \
    "noisereduce>=3.0.2"

"${PIP}" install \
    espnet \
    espnet_model_zoo

"${PIP}" install \
    "git+https://github.com/reazon-research/ReazonSpeech.git#subdirectory=pkg/espnet-asr"

# 패키지 의존성으로 인해 버전이 깨지는 것을 방지
"${PIP}" install --force-reinstall \
    "numpy==1.26.4" \
    "scipy<1.14"

echo
echo "============================================================"
echo " 5. PyTorch와 CUDA 확인"
echo "============================================================"

if ! "${PYTHON}" -c "import torch" >/dev/null 2>&1; then
    echo "PyTorch가 없어 CUDA 12.8 버전을 설치합니다."

    "${PIP}" install \
        torch \
        torchvision \
        torchaudio \
        --index-url https://download.pytorch.org/whl/cu128
fi

"${PYTHON}" - <<'PY'
import sys
import numpy
import torch

print("Python:", sys.version)
print("NumPy:", numpy.__version__)
print("PyTorch:", torch.__version__)
print("PyTorch CUDA:", torch.version.cuda)
print("CUDA 사용 가능:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA GPU를 사용할 수 없습니다. "
        "Vast.ai 인스턴스와 PyTorch 환경을 확인하세요."
    )

print("GPU:", torch.cuda.get_device_name(0))
PY

echo
echo "============================================================"
echo " 6. 작업 디렉터리 생성"
echo "============================================================"

mkdir -p \
    /workspace/jobs \
    /workspace/.cache/huggingface \
    /workspace/.cache/torch

echo
echo "============================================================"
echo " 7. Cloudflare Tunnel 설치"
echo "============================================================"

if ! command -v cloudflared >/dev/null 2>&1; then
    ARCH="$(dpkg --print-architecture)"

    case "${ARCH}" in
        amd64)
            CLOUDFLARED_ARCH="amd64"
            ;;
        arm64)
            CLOUDFLARED_ARCH="arm64"
            ;;
        *)
            echo "지원하지 않는 CPU 아키텍처입니다: ${ARCH}"
            exit 1
            ;;
    esac

    wget -q --show-progress \
        -O /usr/local/bin/cloudflared \
        "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-${CLOUDFLARED_ARCH}"

    chmod +x /usr/local/bin/cloudflared
fi

echo "Cloudflared:"
cloudflared --version

echo
echo "============================================================"
echo " 8. 기존 웹서버 및 터널 정리"
echo "============================================================"

pkill -f "${APP_DIR}/app.py" 2>/dev/null || true
pkill -f "cloudflared tunnel.*127.0.0.1" 2>/dev/null || true
sleep 2

echo
echo "============================================================"
echo " 9. 사용 가능한 포트 검색"
echo "============================================================"

SERVER_PORT="$("${PYTHON}" - <<'PY'
import socket

for port in range(11111, 11212):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            continue

        print(port)
        break
else:
    raise RuntimeError("11111~11211 범위에서 빈 포트를 찾지 못했습니다.")
PY
)"

echo "사용할 포트: ${SERVER_PORT}"

echo
echo "======================================================================"
echo " 자막 추출 웹서버를 시작합니다."
echo
echo " 잠시 후 아래 형식의 Cloudflare 공개 주소가 출력됩니다."
echo
echo " https://xxxxxxxxxxxxxxxx.trycloudflare.com"
echo
echo " 해당 주소를 클릭하면 자막 추출 웹페이지가 열립니다."
echo " 서버가 실행되는 동안 이 터미널을 닫지 마세요."
echo " 종료하려면 Ctrl+C를 누르세요."
echo "======================================================================"
echo

cd "${APP_DIR}"

APP_PID=""

cleanup() {
    echo
    echo "웹서버와 Cloudflare 터널을 종료합니다."

    if [ -n "${APP_PID}" ]; then
        kill "${APP_PID}" 2>/dev/null || true
        wait "${APP_PID}" 2>/dev/null || true
    fi
}

trap cleanup EXIT INT TERM

env \
    HF_HOME="${HF_HOME}" \
    TORCH_HOME="${TORCH_HOME}" \
    XDG_CACHE_HOME="${XDG_CACHE_HOME}" \
    GRADIO_ANALYTICS_ENABLED="False" \
    GRADIO_SERVER_NAME="0.0.0.0" \
    GRADIO_SERVER_PORT="${SERVER_PORT}" \
    "${PYTHON}" "${APP_DIR}/app.py" &

APP_PID=$!

echo "Gradio 웹서버가 준비될 때까지 기다립니다."

SERVER_READY="False"

for attempt in $(seq 1 180); do
    if ! kill -0 "${APP_PID}" 2>/dev/null; then
        echo "Gradio 웹서버가 비정상 종료되었습니다."
        wait "${APP_PID}" || true
        exit 1
    fi

    if curl \
        --silent \
        --fail \
        --max-time 2 \
        "http://127.0.0.1:${SERVER_PORT}/" \
        >/dev/null 2>&1
    then
        SERVER_READY="True"
        break
    fi

    sleep 1
done

if [ "${SERVER_READY}" != "True" ]; then
    echo "제한 시간 안에 Gradio 웹서버가 준비되지 않았습니다."
    exit 1
fi

echo
echo "Gradio 웹서버가 준비되었습니다."
echo "Cloudflare 공개 주소를 생성합니다."
echo

cloudflared tunnel \
    --url "http://127.0.0.1:${SERVER_PORT}" \
    --no-autoupdate

