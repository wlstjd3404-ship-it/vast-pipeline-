#!/usr/bin/env bash
set -e

REPO_DIR="/workspace/repo"
VENV_LADA="/workspace/venv_lada"
VENV_STT="/workspace/venv_stt"

echo "=============================================="
echo "▶ [1/6] 시스템 패키지"
echo "=============================================="
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y ffmpeg curl git python3-venv libsndfile1 unzip

if ! command -v rclone &> /dev/null; then
    curl -s https://rclone.org/install.sh | bash
fi
rclone version | head -n 1

echo "=============================================="
echo "▶ [2/6] Web UI 의존성 (시스템 파이썬)"
echo "=============================================="
pip install -q --upgrade pip
pip install -q gradio

echo "=============================================="
echo "▶ [3/6] LADA 전용 venv 생성"
echo "=============================================="
cd /workspace
[ -d lada ] || git clone -q https://github.com/ladaapp/lada.git

# --system-site-packages : vast 이미지의 CUDA 빌드 torch 재사용 (수 GB 다운로드 회피)
[ -d "$VENV_LADA" ] || python3 -m venv --system-site-packages "$VENV_LADA"
"$VENV_LADA/bin/pip" install -q --upgrade pip

cd /workspace/lada
"$VENV_LADA/bin/pip" install -q -e ".[nvidia]"     # ★ --no-deps 제거
"$VENV_LADA/bin/pip" install -q ffmpeg-python huggingface_hub
"$VENV_LADA/bin/pip" install -q "numpy<2.0"        # ★ 반드시 마지막

echo "--- LADA 환경 점검 ---"
"$VENV_LADA/bin/python" - <<'PY'
import torch, numpy
print("torch :", torch.__version__, "| cuda:", torch.cuda.is_available())
print("numpy :", numpy.__version__)
PY
test -x "$VENV_LADA/bin/lada-cli" && echo "lada-cli : OK"

echo "=============================================="
echo "▶ [4/6] STT 전용 venv 생성"
echo "=============================================="
cd /workspace
[ -d ReazonSpeech ] || git clone -q https://github.com/reazon-research/ReazonSpeech

[ -d "$VENV_STT" ] || python3 -m venv --system-site-packages "$VENV_STT"
"$VENV_STT/bin/pip" install -q --upgrade pip
"$VENV_STT/bin/pip" install -q noisereduce librosa soundfile
"$VENV_STT/bin/pip" install -q espnet espnet_model_zoo
"$VENV_STT/bin/pip" install -q /workspace/ReazonSpeech/pkg/espnet-asr
"$VENV_STT/bin/pip" install -q "numpy<2.0"         # ★ 반드시 마지막

echo "--- STT 환경 점검 ---"
"$VENV_STT/bin/python" - <<'PY'
import numpy, torch
print("numpy :", numpy.__version__)
assert numpy.__version__.startswith("1."), "numpy 1.x 아님! espnet 실패함"
print("torch :", torch.__version__, "| cuda:", torch.cuda.is_available())
import reazonspeech.espnet.asr.ctc as c
print("ctc_decode 존재 :", hasattr(c, "ctc_decode"))
PY

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
