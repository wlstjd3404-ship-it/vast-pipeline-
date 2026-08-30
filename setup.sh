#!/usr/bin/env bash
set -e

export DEBIAN_FRONTEND=noninteractive

echo "=== 1. 시스템 패키지 및 최신 Rclone 설치 ==="
apt-get update -y
apt-get install -y ffmpeg git curl python3-pip python3-venv

if ! command -v rclone &> /dev/null; then
    curl https://rclone.org/install.sh | bash
fi

echo "=== 2. 통합 가상환경 구축 ==="
cd /workspace
python3 -m venv /workspace/venv
/workspace/venv/bin/pip install --upgrade pip

# NumPy 1.x를 먼저 고정하여 하위 패키지 ABI 호환성 유지
/workspace/venv/bin/pip install "numpy<2.0"
/workspace/venv/bin/pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# LADA 설치
if [ ! -d "lada" ]; then
    git clone https://github.com/ladaapp/lada.git
fi
cd /workspace/lada
/workspace/venv/bin/pip install -e ".[nvidia]"
/workspace/venv/bin/pip install ffmpeg-python huggingface_hub

# STT 및 WebUI 패키지 설치
cd /workspace
/workspace/venv/bin/pip install noisereduce librosa soundfile espnet espnet_model_zoo gradio
if [ ! -d "ReazonSpeech" ]; then
    git clone https://github.com/reazon-research/ReazonSpeech
fi
/workspace/venv/bin/pip install ReazonSpeech/pkg/espnet-asr

echo "=== 3. 모델 가중치 다운로드 ==="
cd /workspace/repo
/workspace/venv/bin/python download_models.py

echo "=== 4. Gradio UI 실행 ==="
nohup /workspace/venv/bin/python app.py > /workspace/app.log 2>&1 &
echo "✔ 모든 셋업 완료! /workspace/app.log 에서 구동 확인 가능."