#!/usr/bin/env bash
set -e

REPO_DIR="/workspace/repo"
VENV_LADA="/workspace/venv_lada"
VENV_STT="/workspace/venv_stt"

echo "=============================================="
echo "▶ [1/6] 시스템 패키지 점검"
echo "=============================================="
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y ffmpeg curl git python3-venv libsndfile1 unzip

echo "=============================================="
echo "▶ [2/6] Web UI 의존성 및 gdown 설치"
echo "=============================================="
pip install -q --upgrade pip
pip install -q gradio gdown

echo "=============================================="
echo "▶ [3/6] LADA 전용 venv 생성"
echo "=============================================="
cd /workspace
[ -d lada ] || git clone -q https://github.com/ladaapp/lada.git
[ -d "$VENV_LADA" ] || python3 -m venv --system-site-packages "$VENV_LADA"
"$VENV_LADA/bin/pip" install -q --upgrade pip
cd /workspace/lada
"$VENV_LADA/bin/pip" install -q -e ".[nvidia]"
"$VENV_LADA/bin/pip" install -q ffmpeg-python huggingface_hub gdown "numpy<2.0"

echo "=============================================="
echo "▶ [4/6] STT 전용 venv 생성"
echo "=============================================="
cd /workspace
[ -d ReazonSpeech ] || git clone -q https://github.com/reazon-research/ReazonSpeech
[ -d "$VENV_STT" ] || python3 -m venv --system-site-packages "$VENV_STT"
"$VENV_STT/bin/pip" install -q --upgrade pip
"$VENV_STT/bin/pip" install -q noisereduce librosa soundfile espnet espnet_model_zoo gdown "numpy<2.0"
"$VENV_STT/bin/pip" install -q /workspace/ReazonSpeech/pkg/espnet-asr

echo "=============================================="
echo "▶ [5/6] 모델 가중치 다운로드"
echo "=============================================="
cd "$REPO_DIR"
"$VENV_LADA/bin/python" download_models.py

echo "=============================================="
echo "▶ [6/6] Web UI 실행"
echo "=============================================="
cd "$REPO_DIR"
python app.py