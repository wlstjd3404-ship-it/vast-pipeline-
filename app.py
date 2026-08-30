#!/usr/bin/env bash
set -e

# ==============================================
# 0. 경로 및 환경 설정
# ==============================================
if [ -f "$PWD/app.py" ]; then
    REPO_DIR="$PWD"
else
    REPO_DIR="/workspace/repo"
fi

VENV_LADA="/workspace/venv_lada"
VENV_STT="/workspace/venv_stt"
GITHUB_REPO_URL="${1:-}"

echo "=============================================="
echo "▶ [1/7] 시스템 패키지 및 Rclone 점검"
echo "=============================================="
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y ffmpeg curl git python3-venv libsndfile1 unzip

if ! command -v rclone &> /dev/null; then
    curl -s https://rclone.org/install.sh | bash
fi
rclone version | head -n 1

# Rclone 설정 파일 대화형 입력 (기존 설정 부재 시에만 프롬프트 표시)
RCLONE_CONF="/root/.config/rclone/rclone.conf"
if [ ! -f "$RCLONE_CONF" ] || ! grep -q "\[gdrive\]" "$RCLONE_CONF" 2>/dev/null; then
    echo "----------------------------------------------"
    echo "⚠️ Rclone 구글 드라이브 토큰 설정이 필요합니다."
    echo "터미널에 JSON 형태의 token 값 전체를 붙여넣고 [Enter]를 누르세요."
    echo "예시: {\"access_token\":\"...\",\"refresh_token\":\"...\"}"
    echo "----------------------------------------------"
    read -r -p "Gdrive Token JSON: " GDRIVE_TOKEN
    
    mkdir -p /root/.config/rclone
    cat << EOF > "$RCLONE_CONF"
[gdrive]
type = drive
scope = drive
token = $GDRIVE_TOKEN
EOF
    echo "✅ rclone.conf 생성이 완료되었습니다."
fi

echo "--- Rclone 드라이브 연결 테스트 ---"
rclone lsd gdrive: || echo "⚠️ gdrive 연결 실패 (토큰 값 재확인 필요)"

echo "=============================================="
echo "▶ [2/7] GitHub 레포지토리 준비"
echo "=============================================="
if [ ! -d "$REPO_DIR" ]; then
    if [ -n "$GITHUB_REPO_URL" ]; then
        git clone "$GITHUB_REPO_URL" "$REPO_DIR"
    else
        echo "❌ $REPO_DIR 디렉토리가 없습니다."
        echo "사용법: bash setup.sh <깃허브_레포지토리_URL>"
        exit 1
    fi
fi
echo "저장소 디렉토리: $REPO_DIR"

echo "=============================================="
echo "▶ [3/7] Web UI 의존성 (시스템 파이썬)"
echo "=============================================="
pip install -q --upgrade pip
pip install -q gradio

echo "=============================================="
echo "▶ [4/7] LADA 전용 venv 생성"
echo "=============================================="
cd /workspace
[ -d lada ] || git clone -q https://github.com/ladaapp/lada.git

[ -d "$VENV_LADA" ] || python3 -m venv --system-site-packages "$VENV_LADA"
"$VENV_LADA/bin/pip" install -q --upgrade pip

cd /workspace/lada
"$VENV_LADA/bin/pip" install -q -e ".[nvidia]"
"$VENV_LADA/bin/pip" install -q ffmpeg-python huggingface_hub
"$VENV_LADA/bin/pip" install -q "numpy<2.0"

echo "--- LADA 환경 점검 ---"
"$VENV_LADA/bin/python" - <<'PY'
import torch, numpy
print("torch :", torch.__version__, "| cuda:", torch.cuda.is_available())
print("numpy :", numpy.__version__)
PY
test -x "$VENV_LADA/bin/lada-cli" && echo "lada-cli : OK"

echo "=============================================="
echo "▶ [5/7] STT 전용 venv 생성"
echo "=============================================="
cd /workspace
[ -d ReazonSpeech ] || git clone -q https://github.com/reazon-research/ReazonSpeech

[ -d "$VENV_STT" ] || python3 -m venv --system-site-packages "$VENV_STT"
"$VENV_STT/bin/pip" install -q --upgrade pip
"$VENV_STT/bin/pip" install -q noisereduce librosa soundfile
"$VENV_STT/bin/pip" install -q espnet espnet_model_zoo
"$VENV_STT/bin/pip" install -q /workspace/ReazonSpeech/pkg/espnet-asr
"$VENV_STT/bin/pip" install -q "numpy<2.0"

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
echo "▶ [6/7] 모델 가중치 다운로드"
echo "=============================================="
cd "$REPO_DIR"
"$VENV_LADA/bin/python" download_models.py

echo "=============================================="
echo "▶ [7/7] Web UI 실행"
echo "=============================================="
cd "$REPO_DIR"
python app.py
