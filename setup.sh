cat << 'EOF' > /workspace/repo/setup.sh
#!/usr/bin/env bash
set -e

REPO_DIR="/workspace/repo"
VENV_LADA="/workspace/venv_lada"
VENV_STT="/workspace/venv_stt"

# ★ 전역 제약: 모든 pip 호출에서 setuptools 81+ 차단 (pkg_resources 제거 방지)
CONSTRAINT_FILE="/workspace/constraints.txt"
printf 'setuptools<81\n' > "$CONSTRAINT_FILE"
export PIP_CONSTRAINT="$CONSTRAINT_FILE"

# ★ venv 내부 nvidia 휠의 .so 경로를 모아 LD_LIBRARY_PATH 문자열 생성
nvidia_ld_path() {
  local venv="$1" dirs
  dirs=$(ls -d "$venv"/lib/python*/site-packages/nvidia/*/lib \
                "$venv"/lib/python*/site-packages/torch/lib 2>/dev/null | tr '\n' ':')
  echo "${dirs%:}"
}

echo "=============================================="
echo "▶ [0/6] 이전 실패 흔적 정리"
echo "=============================================="
if [ -x "$VENV_STT/bin/python" ]; then
  V=$("$VENV_STT/bin/python" -c 'import sys;print("%d.%d"%sys.version_info[:2])' 2>/dev/null || echo none)
  if [ "$V" != "3.10" ]; then
    echo "  기존 STT venv(python $V) 삭제 → 3.10 으로 재생성"
    rm -rf "$VENV_STT"
  fi
fi
if [ -x "$VENV_STT/bin/python" ]; then
  if ! "$VENV_STT/bin/python" -c 'import pkg_resources' >/dev/null 2>&1; then
    echo "  venv_stt 의 pkg_resources 없음 → setuptools 79.0.1 강제 복구"
    "$VENV_STT/bin/pip" install -q --force-reinstall "setuptools==79.0.1" || rm -rf "$VENV_STT"
  fi
fi
if [ -x "$VENV_LADA/bin/pip" ]; then
  NV=$("$VENV_LADA/bin/python" -c 'import numpy;print(numpy.__version__)' 2>/dev/null || echo none)
  case "$NV" in
    1.*) echo "  venv_lada 의 numpy $NV 제거 → 시스템 numpy2 로 복귀"
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
pip install -q --upgrade pip --root-user-action=ignore
pip install -q "setuptools<81" wheel --root-user-action=ignore
pip install -q gradio gdown --root-user-action=ignore

echo "=============================================="
echo "▶ [3/6] LADA 전용 venv (시스템 py3.12 + 시스템 torch 재사용)"
echo "=============================================="
cd /workspace
[ -d lada ] || git clone -q https://github.com/ladaapp/lada.git
[ -d "$VENV_LADA" ] || python3 -m venv --system-site-packages "$VENV_LADA"
"$VENV_LADA/bin/pip" install -q --upgrade pip
"$VENV_LADA/bin/pip" install -q "setuptools<81" wheel
cd /workspace/lada
"$VENV_LADA/bin/pip" install -q -e ".[nvidia]"
"$VENV_LADA/bin/pip" install -q ffmpeg-python huggingface_hub gdown
# ★ numpy<2 를 절대 강제하지 않음 (시스템 scipy/opencv ABI 파괴 방지)

# ★ venv 안에 torch 가 새로 깔렸는지 확인 → 깔렸다면 CUDA 런타임 휠 보강
LADA_SP=$(ls -d "$VENV_LADA"/lib/python*/site-packages 2>/dev/null | head -1)
if [ -d "$LADA_SP/torch" ]; then
  echo "  venv-local torch 감지 → CUDA 런타임 휠 보강 (libcusparseLt 등)"
  "$VENV_LADA/bin/pip" install -q \
      nvidia-cusparselt-cu12 nvidia-cusparse-cu12 nvidia-cublas-cu12 \
      nvidia-cudnn-cu12 nvidia-cuda-runtime-cu12 nvidia-cuda-nvrtc-cu12 \
      nvidia-cufft-cu12 nvidia-curand-cu12 nvidia-cusolver-cu12 \
      nvidia-nccl-cu12 nvidia-nvtx-cu12 nvidia-nvjitlink-cu12 || true
fi

# ★ sitecustomize: nvidia 휠의 .so 를 ctypes 로 프리로드 (LD_LIBRARY_PATH 의존 제거)
cat > "$LADA_SP/sitecustomize.py" <<'PYSITE'
import os, glob, ctypes
_base = os.path.join(os.path.dirname(__file__), "nvidia")
if os.path.isdir(_base):
    _libs = sorted(glob.glob(os.path.join(_base, "*", "lib", "*.so*")))
    for _ in range(2):                      # 의존 순서 문제 대비 2-pass
        for _p in _libs:
            try:
                ctypes.CDLL(_p, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass
PYSITE

echo "  ▶ LADA 환경 자체 검증"
export LD_LIBRARY_PATH="$(nvidia_ld_path "$VENV_LADA"):${LD_LIBRARY_PATH}"
if ! "$VENV_LADA/bin/python" - <<'PYEOF'
import setuptools, numpy, torch
print(f"     setuptools={setuptools.__version__} numpy={numpy.__version__} "
      f"torch={torch.__version__} cuda={torch.cuda.is_available()}")
PYEOF
then
  echo "  ⚠ venv-local torch 로드 실패 → 제거 후 시스템 torch 로 폴백"
  "$VENV_LADA/bin/pip" uninstall -y -q torch torchvision torchaudio || true
  "$VENV_LADA/bin/python" - <<'PYEOF'
import torch, numpy
print(f"     [fallback] torch={torch.__version__} numpy={numpy.__version__} "
      f"cuda={torch.cuda.is_available()} path={torch.__file__}")
PYEOF
fi
[ -x "$VENV_LADA/bin/lada-cli" ] || { echo "  ✘ lada-cli 생성 실패"; exit 1; }

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
PIP_STT="$VENV_STT/bin/pip"
$PIP_STT install -q --upgrade pip
# ★ uv --seed 가 심어놓은 setuptools 84 를 79.0.1 로 교체 (pkg_resources 필수)
$PIP_STT install -q --force-reinstall "setuptools==79.0.1" wheel
$PIP_STT install -q torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
# ★ librosa 0.11+ 는 pkg_resources 의존 제거 → 이중 안전장치
$PIP_STT install -q "numpy==1.26.4" "librosa>=0.11.0" soundfile noisereduce gdown
$PIP_STT install -q espnet espnet_model_zoo
$PIP_STT install -q /workspace/ReazonSpeech/pkg/espnet-asr
$PIP_STT install -q --force-reinstall "setuptools==79.0.1"
$PIP_STT install -q "numpy==1.26.4"

# ★ STT venv 에도 동일한 프리로드 장치 설치
STT_SP=$(ls -d "$VENV_STT"/lib/python*/site-packages 2>/dev/null | head -1)
cp "$LADA_SP/sitecustomize.py" "$STT_SP/sitecustomize.py" 2>/dev/null || true

echo "  ▶ STT 환경 자체 검증"
LD_LIBRARY_PATH="$(nvidia_ld_path "$VENV_STT"):${LD_LIBRARY_PATH}" \
"$VENV_STT/bin/python" - <<'PYEOF'
import setuptools, pkg_resources, numpy, librosa, torch
from espnet2.bin.asr_inference import Speech2Text
from reazonspeech.espnet.asr import transcribe, audio_from_path
import reazonspeech.espnet.asr.ctc as _ctc
print(f"     setuptools={setuptools.__version__} numpy={numpy.__version__} librosa={librosa.__version__}")
print(f"     torch={torch.__version__} cuda={torch.cuda.is_available()}")
print("     ✔ import OK")
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

cd /workspace/repo && bash setup.sh
