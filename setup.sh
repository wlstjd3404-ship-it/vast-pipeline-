cat << 'EOF' > /workspace/repo/setup.sh
#!/usr/bin/env bash
set -e

REPO_DIR="/workspace/repo"
VENV_LADA="/workspace/venv_lada"
VENV_STT="/workspace/venv_stt"
CONSTRAINT="/workspace/constraints_stt.txt"

echo "=============================================="
echo "▶ [0/6] 이전 실패 흔적 정리"
echo "=============================================="
if [ -x "$VENV_STT/bin/python" ]; then
  V=$("$VENV_STT/bin/python" -c 'import sys;print("%d.%d"%sys.version_info[:2])' 2>/dev/null || echo none)
  if [ "$V" != "3.10" ]; then
    echo "  기존 STT venv(python $V) 삭제 → 3.10 재생성"
    rm -rf "$VENV_STT"
  fi
fi
if [ -x "$VENV_LADA/bin/pip" ]; then
  NV=$("$VENV_LADA/bin/python" -c 'import numpy;print(numpy.__version__)' 2>/dev/null || echo none)
  case "$NV" in
    1.*) echo "  venv_lada 의 numpy $NV 제거 → 시스템 numpy2 복귀"
         "$VENV_LADA/bin/pip" uninstall -y -q numpy || true ;;
  esac
fi

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
echo "▶ [3/6] LADA 전용 venv (시스템 py3.12 + 시스템 torch 재사용)"
echo "=============================================="
cd /workspace
[ -d lada ] || git clone -q https://github.com/ladaapp/lada.git
[ -d "$VENV_LADA" ] || python3 -m venv --system-site-packages "$VENV_LADA"
"$VENV_LADA/bin/pip" install -q --upgrade pip setuptools wheel
cd /workspace/lada
"$VENV_LADA/bin/pip" install -q -e ".[nvidia]"
"$VENV_LADA/bin/pip" install -q ffmpeg-python huggingface_hub gdown
# ★ numpy<2 강제 금지 (시스템 scipy/opencv ABI 파괴 방지)

echo "=============================================="
echo "▶ [4/6] STT 전용 venv — Python 3.10 + numpy 1.26 고정"
echo "=============================================="
cd /workspace
[ -d ReazonSpeech ] || git clone -q https://github.com/reazon-research/ReazonSpeech
if [ ! -x "$VENV_STT/bin/python" ]; then
  command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
  uv venv --python 3.10 --seed "$VENV_STT"
fi

# ★ 핵심: 모든 후속 설치에서 numpy 승격 차단 (빌드 환경에도 적용됨)
echo "numpy==1.26.4" > "$CONSTRAINT"
export PIP_CONSTRAINT="$CONSTRAINT"

PIP_STT="$VENV_STT/bin/pip"
$PIP_STT install -q --upgrade pip setuptools wheel
$PIP_STT install -q torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
$PIP_STT install -q "numpy==1.26.4" librosa soundfile noisereduce gdown
$PIP_STT install -q espnet espnet_model_zoo
$PIP_STT install -q /workspace/ReazonSpeech/pkg/espnet-asr
# 혹시라도 올라갔으면 되돌림
$PIP_STT install -q "numpy==1.26.4"

echo "  ▶ STT 환경 자체 검증"
"$VENV_STT/bin/python" - <<'PY'
import sys, numpy, torch, ctc_segmentation, espnet2
from reazonspeech.espnet.asr import transcribe, audio_from_path
assert numpy.__version__.startswith("1."), f"numpy 승격 감지: {numpy.__version__}"
print(f"  ✔ py{sys.version.split()[0]} / numpy {numpy.__version__} / torch {torch.__version__} / cuda={torch.cuda.is_available()}")
PY

unset PIP_CONSTRAINT

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

cd /workspace/repo && bash setup.sh
