#!/usr/bin/env bash
# ==============================================================================
# Vast.ai 자막 추출 파이프라인 (원커맨드)
#
#   1) 필수 패키지 설치 (최초 1회, /workspace/.setup_done 마커로 재실행시 건너뜀)
#   2) 구글 드라이브 공유 폴더(audio_XX.wav + References.txt) 다운로드
#   3) transcribe.py 실행 → /workspace/output/audio_files.srt
#
# 사용법 (Vast.ai 터미널):
#   git clone https://github.com/wlstjd3404-ship-it/vast-pipeline-.git && cd vast-pipeline- && bash run.sh
#
# 옵션:
#   bash run.sh <다른 드라이브 폴더 링크>   # 기본 링크 대신 다른 폴더 사용
#   SKIP_DOWNLOAD=1 bash run.sh            # 이미 /workspace/audio 에 파일이 있을 때
#   NO_DENOISE=1 bash run.sh               # 노이즈 제거 건너뛰기
# ==============================================================================
set -Eeuo pipefail

# ------------------------------------------------------------------------------
# 설정
# ------------------------------------------------------------------------------
DRIVE_URL="${1:-${DRIVE_URL:-https://drive.google.com/drive/folders/1uzD-xr6TC3r6y1mCKyrYUkyzV2oWdkax?usp=drive_link}}"
WORK="${WORKSPACE:-/workspace}"
AUDIO_DIR="${AUDIO_DIR:-${WORK}/audio}"
OUT_DIR="${OUT_DIR:-${WORK}/output}"
OUT_SRT="${OUT_DIR}/audio_files.srt"
SETUP_MARK="${WORK}/.setup_done"
APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export DEBIAN_FRONTEND=noninteractive
export PIP_DISABLE_PIP_VERSION_CHECK=1
export HF_HOME="${WORK}/.cache/huggingface"
export TORCH_HOME="${WORK}/.cache/torch"
export HF_HUB_DISABLE_TELEMETRY=1
mkdir -p "${AUDIO_DIR}" "${OUT_DIR}" "${HF_HOME}" "${TORCH_HOME}"

# Vast.ai PyTorch 템플릿의 기본 가상환경 활성화 (/venv/main)
if [ -f /venv/main/bin/activate ]; then
    # shellcheck disable=SC1091
    source /venv/main/bin/activate
fi
PY="$(command -v python)"

banner() {
    echo
    echo "=============================================================="
    echo " $*"
    echo "=============================================================="
}

# ------------------------------------------------------------------------------
# 1. 환경 설치 (최초 1회)
# ------------------------------------------------------------------------------
if [ -f "${SETUP_MARK}" ]; then
    banner "1. 환경 설치 - 이미 완료됨, 건너뜀 (${SETUP_MARK})"
else
    banner "1-1. 시스템 패키지 설치"
    apt-get update -qq
    apt-get install -y -qq --no-install-recommends git ffmpeg libsndfile1 > /dev/null

    banner "1-2. PyTorch / GPU 확인"
    rc=0
    "${PY}" - <<'PYCODE' || rc=$?
import sys, torch
print("Python :", sys.version.split()[0])
print("Torch  :", torch.__version__, "| CUDA", torch.version.cuda)
if not torch.cuda.is_available():
    sys.exit("CUDA GPU 를 사용할 수 없습니다.")
cap = torch.cuda.get_device_capability(0)
arch = f"sm_{cap[0]}{cap[1]}"
print("GPU    :", torch.cuda.get_device_name(0), f"({arch})")
if arch not in torch.cuda.get_arch_list():
    print(f"경고: 현재 torch 빌드가 {arch} 를 지원하지 않습니다 → 재설치 필요")
    sys.exit(42)
PYCODE
    if [ "${rc}" -eq 42 ]; then
        echo "▶ RTX 5090(Blackwell) 지원 PyTorch(cu128) 재설치 중..."
        "${PY}" -m pip install -q --upgrade torch torchaudio --index-url https://download.pytorch.org/whl/cu128
    elif [ "${rc}" -ne 0 ]; then
        exit "${rc}"
    fi

    banner "1-3. Python 패키지 설치 (espnet / ReazonSpeech) - 수 분 소요"
    "${PY}" -m pip install -q --upgrade pip "setuptools<82" wheel
    # Python 3.12 호환 버전으로 강제 업그레이드 (기존 구형 SciPy가 조건만 만족해 남는 문제 방지)
    "${PY}" -m pip install -q --upgrade --force-reinstall "numpy==1.26.4" "scipy==1.13.1"
    "${PY}" -m pip install -q gdown huggingface_hub librosa soundfile noisereduce
    "${PY}" -m pip install -q espnet espnet_model_zoo
    "${PY}" -m pip install -q "git+https://github.com/reazon-research/ReazonSpeech.git#subdirectory=pkg/espnet-asr"
    # 의존성 설치 후 NumPy/SciPy 호환 조합과 PyTorch의 setuptools 상한을 다시 고정
    "${PY}" -m pip install -q --force-reinstall --no-deps "numpy==1.26.4" "scipy==1.13.1" "setuptools<82"

    banner "1-4. 임포트 검증"
    "${PY}" - <<'PYCODE'
import numpy, torch
from reazonspeech.espnet.asr import transcribe, audio_from_path
from espnet2.bin.asr_inference import Speech2Text
import reazonspeech.espnet.asr.ctc
print("NumPy", numpy.__version__, "| Torch", torch.__version__, "| ReazonSpeech OK")
PYCODE

    touch "${SETUP_MARK}"
    echo "✅ 환경 설치 완료"
fi

# ------------------------------------------------------------------------------
# 2. 구글 드라이브 폴더 다운로드
# ------------------------------------------------------------------------------
if [ "${SKIP_DOWNLOAD:-0}" = "1" ]; then
    banner "2. 다운로드 건너뜀 (SKIP_DOWNLOAD=1) → ${AUDIO_DIR}"
else
    banner "2. 구글 드라이브 폴더 다운로드"
    echo "  링크 : ${DRIVE_URL}"
    echo "  저장 : ${AUDIO_DIR}"
    rm -rf "${AUDIO_DIR:?}"/*
    "${PY}" -m gdown --folder "${DRIVE_URL}" -O "${AUDIO_DIR}"
fi

# gdown 이 하위 폴더를 만들었을 경우 평탄화
if [ ! -f "${AUDIO_DIR}/References.txt" ]; then
    found="$(find "${AUDIO_DIR}" -maxdepth 3 -name References.txt | head -n 1 || true)"
    if [ -n "${found}" ]; then
        mv "$(dirname "${found}")"/* "${AUDIO_DIR}/" 2>/dev/null || true
    fi
fi
if [ ! -f "${AUDIO_DIR}/References.txt" ]; then
    echo "❌ References.txt 를 찾을 수 없습니다: ${AUDIO_DIR}"
    echo "   - 드라이브 폴더가 '링크가 있는 모든 사용자' 로 공유되어 있는지 확인"
    echo "   - 파일이 많거나 크면 gdown 이 실패할 수 있음 → 잠시 후 재시도"
    exit 1
fi
echo "  📄 References.txt:"
sed 's/^/     /' "${AUDIO_DIR}/References.txt"
echo "  🎧 wav 파일 수: $(find "${AUDIO_DIR}" -maxdepth 1 -name '*.wav' | wc -l)"

# ------------------------------------------------------------------------------
# 3. 자막 추출
# ------------------------------------------------------------------------------
banner "3. 자막 추출 시작"
EXTRA=""
if [ "${NO_DENOISE:-0}" = "1" ]; then
    EXTRA="--no-denoise"
fi
# shellcheck disable=SC2086
"${PY}" "${APP_DIR}/transcribe.py" --audio-dir "${AUDIO_DIR}" --out "${OUT_SRT}" ${EXTRA}

# ------------------------------------------------------------------------------
# 4. 결과 안내
# ------------------------------------------------------------------------------
banner "4. 완료 - 결과 파일 받기"
echo "  📁 ${OUT_SRT}  ($(wc -l < "${OUT_SRT}") 줄)"
echo
echo "  [방법 A] Vast.ai 콘솔 → 인스턴스의 'Jupyter' 버튼 → 좌측 파일탐색기에서"
echo "           output/audio_files.srt 우클릭 → Download"
echo
if [ -n "${PUBLIC_IPADDR:-}" ] && [ -n "${VAST_TCP_PORT_22:-}" ]; then
    echo "  [방법 B] 내 PC PowerShell 에서:"
    echo "           scp -P ${VAST_TCP_PORT_22} root@${PUBLIC_IPADDR}:${OUT_SRT} ."
    echo
fi
echo "  [방법 C] 아래 명령으로 터미널에 출력 후 복사:"
echo "           cat ${OUT_SRT}"
echo
