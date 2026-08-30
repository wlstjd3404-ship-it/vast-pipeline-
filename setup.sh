cat << 'EOF' > /workspace/repo/setup.sh
#!/usr/bin/env bash
set -e

REPO_DIR="/workspace/repo"
VENV_LADA="/workspace/venv_lada"
VENV_STT="/workspace/venv_stt"

CONSTRAINT_FILE="/workspace/constraints.txt"
printf 'setuptools<81\n' > "$CONSTRAINT_FILE"
export PIP_CONSTRAINT="$CONSTRAINT_FILE"

nvidia_ld_path() {
  local venv="$1" dirs
  dirs=$(ls -d "$venv"/lib/python*/site-packages/nvidia/*/lib \
                "$venv"/lib/python*/site-packages/torch/lib 2>/dev/null | tr '\n' ':')
  echo "${dirs%:}"
}

echo "=============================================="
echo "▶ [0/6] 이전 가상환경 완전 초기화"
echo "=============================================="
rm -rf "$VENV_LADA" "$VENV_STT"

echo "=============================================="
echo "▶ [1/6] 시스템 패키지 점검"
echo "=============================================="
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y ffmpeg curl git python3-venv libsndfile1 unzip

echo "=============================================="
echo "▶ [2/6] Web UI 의존성 설치"
echo "=============================================="
pip install -q --upgrade pip --root-user-action=ignore
pip install -q "setuptools<81" wheel gradio gdown --root-user-action=ignore

echo "=============================================="
echo "▶ [3/6] LADA 독립 venv 구축 (PyTorch cu124)"
echo "=============================================="
cd /workspace
[ -d lada ] || git clone -q https://github.com/ladaapp/lada.git
python3 -m venv "$VENV_LADA"
"$VENV_LADA/bin/pip" install -q --upgrade pip
"$VENV_LADA/bin/pip" install -q "setuptools<81" wheel

# PyTorch 공식 cu124 바이너리 및 CUDA 12 런타임 설치
"$VENV_LADA/bin/pip" install -q torch torchvision --index-url https://download.pytorch.org/whl/cu124
"$VENV_LADA/bin/pip" install -q nvidia-cusparselt-cu12

cd /workspace/lada
"$VENV_LADA/bin/pip" install -q --no-deps -e .
"$VENV_LADA/bin/pip" install -q ffmpeg-python huggingface_hub gdown opencv-python-headless scipy numpy tqdm requests

LADA_SP=$(ls -d "$VENV_LADA"/lib/python*/site-packages 2>/dev/null | head -1)
cat > "$LADA_SP/sitecustomize.py" <<'PYSITE'
import os, glob, ctypes
_base = os.path.join(os.path.dirname(__file__), "nvidia")
if os.path.isdir(_base):
    _libs = sorted(glob.glob(os.path.join(_base, "*", "lib", "*.so*")))
    for _ in range(2):
        for _p in _libs:
            try:
                ctypes.CDLL(_p, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass
PYSITE

echo "  ▶ LADA 환경 검증"
export LD_LIBRARY_PATH="$(nvidia_ld_path "$VENV_LADA"):${LD_LIBRARY_PATH}"
"$VENV_LADA/bin/python" - <<'PYEOF'
import torch
print(f"     LADA Torch: {torch.__version__} | CUDA Available: {torch.cuda.is_available()}")
assert torch.cuda.is_available(), "CUDA를 인식하지 못했습니다."
PYEOF

[ -x "$VENV_LADA/bin/lada-cli" ] || { echo "  ✘ lada-cli 생성 실패"; exit 1; }

echo "=============================================="
echo "▶ [4/6] STT venv 구축 (Python 3.10 + numpy 1.26)"
echo "=============================================="
cd /workspace
[ -d ReazonSpeech ] || git clone -q https://github.com/reazon-research/ReazonSpeech
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
uv venv --python 3.10 --seed "$VENV_STT"

PIP_STT="$VENV_STT/bin/pip"
$PIP_STT install -q --upgrade pip
$PIP_STT install -q --force-reinstall "setuptools==79.0.1" wheel
$PIP_STT install -q torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
$PIP_STT install -q "numpy==1.26.4" "librosa>=0.11.0" soundfile noisereduce gdown
$PIP_STT install -q espnet espnet_model_zoo
$PIP_STT install -q /workspace/ReazonSpeech/pkg/espnet-asr
$PIP_STT install -q --force-reinstall "setuptools==79.0.1" "numpy==1.26.4"

STT_SP=$(ls -d "$VENV_STT"/lib/python*/site-packages 2>/dev/null | head -1)
cp "$LADA_SP/sitecustomize.py" "$STT_SP/sitecustomize.py" 2>/dev/null || true

echo "  ▶ STT 환경 검증"
LD_LIBRARY_PATH="$(nvidia_ld_path "$VENV_STT"):${LD_LIBRARY_PATH}" \
"$VENV_STT/bin/python" - <<'PYEOF'
import torch
from espnet2.bin.asr_inference import Speech2Text
print(f"     STT Torch: {torch.__version__} | CUDA Available: {torch.cuda.is_available()}")
print("     ✔ STT import 정상 완료")
PYEOF

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
EOF

bash /workspace/repo/setup.sh
