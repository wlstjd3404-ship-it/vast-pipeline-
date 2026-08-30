#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/workspace/repo"
VENV_LADA="/workspace/venv_lada"
VENV_STT="/workspace/venv_stt"
CONSTRAINTS="/workspace/constraints.txt"

# ── 0. 레포 파일 sanity check (app.py에 셸 코드가 섞이는 사고 방지)
if head -n 20 "$REPO_DIR/app.py" | grep -qE '^\s*(echo |cat <<|#!/usr/bin/env bash|set -e)'; then
  echo "❌ app.py 안에 셸 스크립트가 들어있습니다. GitHub의 app.py를 파이썬 코드로 교체하세요."
  exit 1
fi
python3 - <<'PY'
import ast,sys
src=open("/workspace/repo/app.py",encoding="utf-8").read()
try: ast.parse(src)
except SyntaxError as e:
    print(f"❌ app.py 문법 오류: line {e.lineno}: {e.text}"); sys.exit(1)
print("✔ app.py 파이썬 문법 OK")
PY

echo "=============================================="
echo "▶ [1/7] 시스템 패키지 및 Rclone"
echo "=============================================="
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y ffmpeg curl git python3-venv libsndfile1 unzip
command -v rclone >/dev/null || curl -s https://rclone.org/install.sh | bash
rclone version | head -n 1

RCLONE_CONF="/root/.config/rclone/rclone.conf"
if ! grep -q "^\[gdrive\]" "$RCLONE_CONF" 2>/dev/null; then
  if [ -t 0 ]; then
    echo "⚠️ Gdrive token JSON 전체를 붙여넣고 Enter:"
    read -r GDRIVE_TOKEN
    mkdir -p /root/.config/rclone
    printf '[gdrive]\ntype = drive\nscope = drive\ntoken = %s\n' "$GDRIVE_TOKEN" > "$RCLONE_CONF"
    echo "✅ rclone.conf 생성 완료"
  else
    echo "⚠️ 비대화형 실행 → rclone 설정 건너뜀"
  fi
fi
rclone lsd gdrive: >/dev/null 2>&1 && echo "✔ gdrive 연결 OK" || echo "⚠️ gdrive 연결 실패(토큰 확인)"

echo "=============================================="
echo "▶ [2/7] 버전 제약 파일 생성 (numpy 충돌 근본 차단)"
echo "=============================================="
cat > "$CONSTRAINTS" <<'C'
numpy==1.26.4
scipy==1.13.1
opencv-python==4.10.0.84
opencv-python-headless==4.10.0.84
librosa==0.10.2.post1
numba==0.60.0
llvmlite==0.43.0
C
export PIP_CONSTRAINT="$CONSTRAINTS"
cat "$CONSTRAINTS"

echo "=============================================="
echo "▶ [3/7] Web UI 의존성 (시스템 파이썬)"
echo "=============================================="
pip install -q -U pip
pip install -q gradio

# ── nvidia 공유 라이브러리 경로를 venv 단위로 등록하는 헬퍼
register_nvidia_libs () {   # $1=venv 경로  $2=conf 이름
  find "$1/lib" -type d -path "*/site-packages/nvidia/*/lib" 2>/dev/null \
    | sort > "/etc/ld.so.conf.d/zz_$2.conf"
  ldconfig
  echo "  ld.so 등록: $(wc -l < "/etc/ld.so.conf.d/zz_$2.conf") 개 경로"
}

echo "=============================================="
echo "▶ [4/7] LADA 전용 venv"
echo "=============================================="
cd /workspace
[ -d lada ] || git clone -q https://github.com/ladaapp/lada.git
[ -d "$VENV_LADA" ] || python3 -m venv "$VENV_LADA"          # ★ system-site-packages 사용 안 함
PL="$VENV_LADA/bin/pip"
$PL install -q -U pip wheel setuptools
cd /workspace/lada
$PL install -q -e ".[nvidia]"
$PL install -q ffmpeg-python huggingface_hub
$PL install -q -U nvidia-cusparselt-cu12                      # ★ 문제 1 해결
register_nvidia_libs "$VENV_LADA" "lada"

echo "--- LADA 환경 점검 ---"
"$VENV_LADA/bin/python" - <<'PY'
import torch, numpy, cv2, scipy
print("torch :", torch.__version__, "| cuda:", torch.cuda.is_available())
print("numpy :", numpy.__version__, "| cv2:", cv2.__version__, "| scipy:", scipy.__version__)
assert numpy.__version__.startswith("1."), "numpy 1.x 아님"
assert torch.cuda.is_available(), "CUDA 사용 불가"
PY
test -x "$VENV_LADA/bin/lada-cli" && echo "lada-cli : OK"
$PL check || echo "⚠️ (경고만) 의존성 점검 메시지 확인"

echo "=============================================="
echo "▶ [5/7] STT 전용 venv"
echo "=============================================="
cd /workspace
[ -d ReazonSpeech ] || git clone -q https://github.com/reazon-research/ReazonSpeech
[ -d "$VENV_STT" ] || python3 -m venv "$VENV_STT"             # ★ system-site-packages 사용 안 함
PS="$VENV_STT/bin/pip"
$PS install -q -U pip wheel setuptools
$PS install -q "numpy==1.26.4"                                # ★ 먼저 못 박기
$PS install -q torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
$PS install -q -U nvidia-cusparselt-cu12
$PS install -q espnet espnet_model_zoo                        # 제약파일 때문에 numpy1.x 호환 버전 자동 선택
$PS install -q noisereduce librosa soundfile
$PS install -q /workspace/ReazonSpeech/pkg/espnet-asr
register_nvidia_libs "$VENV_STT" "stt"

echo "--- STT 환경 점검 ---"
"$VENV_STT/bin/python" - <<'PY'
import numpy, torch, librosa, soundfile, noisereduce, espnet
print("numpy :", numpy.__version__, "| librosa:", librosa.__version__, "| espnet:", espnet.__version__)
print("torch :", torch.__version__, "| cuda:", torch.cuda.is_available())
assert numpy.__version__.startswith("1."), "numpy 1.x 아님! espnet 실패함"
assert torch.cuda.is_available(), "CUDA 사용 불가"
from espnet2.bin.asr_inference import Speech2Text
import reazonspeech.espnet.asr.ctc as c
print("ctc_decode 존재 :", hasattr(c, "ctc_decode"))
PY
$PS check || echo "⚠️ (경고만) 의존성 점검 메시지 확인"

echo "=============================================="
echo "▶ [6/7] 모델 가중치 다운로드"
echo "=============================================="
cd "$REPO_DIR"
"$VENV_LADA/bin/python" download_models.py

echo "=============================================="
echo "▶ [7/7] Web UI 실행"
echo "=============================================="
cd "$REPO_DIR"
exec python3 app.py
