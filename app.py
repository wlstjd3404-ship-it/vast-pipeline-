#!/usr/bin/env bash
set -e

REPO_DIR="/workspace/repo"
VENV_LADA="/workspace/venv_lada"
VENV_STT="/workspace/venv_stt"

echo "▶ [1/7] 시스템 패키지 및 Rclone 점검"
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y ffmpeg curl git python3-venv libsndfile1 unzip

command -v rclone >/dev/null || curl -s https://rclone.org/install.sh | bash
rclone version | head -n 1

RCLONE_CONF="/root/.config/rclone/rclone.conf"
if [ ! -f "$RCLONE_CONF" ] || ! grep -q "\[gdrive\]" "$RCLONE_CONF" 2>/dev/null; then
    echo "⚠️ Gdrive 토큰 JSON 전체를 붙여넣고 Enter"
    read -r -p "Gdrive Token JSON: " GDRIVE_TOKEN
    mkdir -p /root/.config/rclone
    cat > "$RCLONE_CONF" <<EOF_CONF
[gdrive]
type = drive
scope = drive
token = $GDRIVE_TOKEN
EOF_CONF
    echo "✅ rclone.conf 생성"
fi
rclone lsd gdrive: || echo "⚠️ gdrive 연결 실패"

echo "▶ [3/7] Web UI 의존성"
python3 -m pip install -q --upgrade pip --break-system-packages 2>/dev/null || python3 -m pip install -q --upgrade pip
python3 -m pip install -q gradio --break-system-packages 2>/dev/null || python3 -m pip install -q gradio

echo "▶ [4/7] LADA venv"
cd /workspace
[ -d lada ] || git clone -q https://github.com/ladaapp/lada.git
[ -d "$VENV_LADA" ] || python3 -m venv --system-site-packages "$VENV_LADA"
"$VENV_LADA/bin/pip" install -q --upgrade pip
cd /workspace/lada
"$VENV_LADA/bin/pip" install -q -e ".[nvidia]"
"$VENV_LADA/bin/pip" install -q ffmpeg-python huggingface_hub
"$VENV_LADA/bin/pip" install -q "numpy<2.0"
"$VENV_LADA/bin/python" - <<'PY'
import torch, numpy
print("torch:", torch.__version__, "| cuda:", torch.cuda.is_available())
print("numpy:", numpy.__version__)
PY
test -x "$VENV_LADA/bin/lada-cli" && echo "lada-cli : OK" || echo "⚠️ lada-cli 없음"

echo "▶ [5/7] STT venv"
cd /workspace
[ -d ReazonSpeech ] || git clone -q https://github.com/reazon-research/ReazonSpeech
[ -d "$VENV_STT" ] || python3 -m venv --system-site-packages "$VENV_STT"
"$VENV_STT/bin/pip" install -q --upgrade pip
"$VENV_STT/bin/pip" install -q noisereduce librosa soundfile
"$VENV_STT/bin/pip" install -q espnet espnet_model_zoo
"$VENV_STT/bin/pip" install -q /workspace/ReazonSpeech/pkg/espnet-asr
"$VENV_STT/bin/pip" install -q "numpy<2.0"
"$VENV_STT/bin/python" - <<'PY'
import numpy, torch
assert numpy.__version__.startswith("1."), "numpy 1.x 아님!"
print("numpy:", numpy.__version__, "| torch:", torch.__version__, "| cuda:", torch.cuda.is_available())
import reazonspeech.espnet.asr.ctc as c
print("ctc_decode:", hasattr(c, "ctc_decode"))
PY

echo "▶ [6/7] 모델 가중치"
cd "$REPO_DIR"
"$VENV_LADA/bin/python" download_models.py

echo "▶ [7/7] Web UI 실행"
cd "$REPO_DIR"
exec python3 app.py
