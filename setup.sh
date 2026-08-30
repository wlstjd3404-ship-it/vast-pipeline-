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

echo "=============================================="
echo "▶ [0/6] 이전 실패 흔적 정리"
echo "=============================================="
# STT venv 가 3.10 이 아니면 폐기 (espnet 은 3.12 미지원)
if [ -x "$VENV_STT/bin/python" ]; then
  V=$("$VENV_STT/bin/python" -c 'import sys;print("%d.%d"%sys.version_info[:2])' 2>/dev/null || echo none)
  if [ "$V" != "3.10" ]; then
    echo "  기존 STT venv(python $V) 삭제 → 3.10 으로 재생성"
    rm -rf "$VENV_STT"
  fi
fi
# ★ pkg_resources 가 죽은 STT venv 는 즉시 복구
if [ -x "$VENV_STT/bin/python" ]; then
  if ! "$VENV_STT/bin/python" -c 'import pkg_resources' >/dev/null 2>&1; then
    echo "  venv_stt 의 pkg_resources 없음 → setuptools 79.0.1 강제 복구"
    "$VENV_STT/bin/pip" install -q --force-reinstall "setuptools==79.0.1" || rm -rf "$VENV_STT"
  fi
fi
# LADA venv 에 잘못 박힌 numpy1.x 제거 (시스템 scipy/opencv 는 numpy2 ABI)
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
# ★ setuptools 는 업그레이드하지 않고 80.x 로 고정 (torch 2.11 요구: <82)
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

echo "  ▶ LADA 환경 자체 검증"
"$VENV_LADA/bin/python" - <<'PYEOF'
import setuptools, numpy, torch
print(f"     setuptools={setuptools.__version__} numpy={numpy.__version__} torch={torch.__version__} cuda={torch.cuda.is_available()}")
PYEOF
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
# ★ uv --seed 가 심어놓은 setuptools 84 를 즉시 79.0.1 로 교체 (pkg_resources 필수)
$PIP_STT install -q --force-reinstall "setuptools==79.0.1" wheel
$PIP_STT install -q torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
# ★ librosa 0.11+ 는 pkg_resources 의존을 제거 → 이중 안전장치
$PIP_STT install -q "numpy==1.26.4" "librosa>=0.11.0" soundfile noisereduce gdown
$PIP_STT install -q espnet espnet_model_zoo
$PIP_STT install -q /workspace/ReazonSpeech/pkg/espnet-asr
# ★ 위 설치들이 setuptools 를 다시 올렸을 수 있으므로 최종 되돌림 + numpy 재고정
$PIP_STT install -q --force-reinstall "setuptools==79.0.1"
$PIP_STT install -q "numpy==1.26.4"

echo "  ▶ STT 환경 자체 검증"
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
