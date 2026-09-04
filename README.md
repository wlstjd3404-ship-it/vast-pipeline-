# vast-pipeline — ReazonSpeech 일본어 자막 추출 (Vast.ai)

구글 코랩에서 돌리던 자막 추출 노트북을 Vast.ai(RTX 5090) 터미널에서 **한 줄 명령**으로 실행하도록 옮긴 것.

```
내 PC: 오디오추출.py  →  G:\내 드라이브\audio\ (audio_01.wav … + References.txt)
                                   │  (구글 드라이브 공유 폴더)
Vast.ai: bash run.sh  →  다운로드 → 노이즈 제거 → ReazonSpeech STT → /workspace/output/audio_files.srt
```

## 1. Vast.ai 인스턴스 만들기

- 템플릿: **PyTorch (Vast)** (CUDA 12.8 이상 이미지 권장)
- GPU: RTX 5090 1장
- 디스크: 40 GB 이상 (모델 + espnet 패키지 + 오디오)
- 생성 후 콘솔에서 **Open → Terminal** (또는 Jupyter → Terminal, SSH)

## 2. 실행 (터미널에 그대로 붙여넣기)

```bash
cd /workspace && git clone https://github.com/wlstjd3404-ship-it/vast-pipeline-.git && cd vast-pipeline- && bash run.sh
```

- 최초 실행: 패키지 설치 5~10분 + 모델 다운로드 후 자막 추출 시작
- 같은 인스턴스에서 다시 돌릴 때 (새 오디오를 드라이브에 올린 뒤):

```bash
cd /workspace/vast-pipeline- && bash run.sh
```

설치는 `/workspace/.setup_done` 마커로 1회만 수행되고, 드라이브 폴더는 매번 새로 다운로드한다.

## 3. 옵션

| 명령 | 설명 |
|---|---|
| `bash run.sh "<드라이브 폴더 링크>"` | 기본 폴더 대신 다른 공유 폴더 사용 |
| `SKIP_DOWNLOAD=1 bash run.sh` | `/workspace/audio` 에 이미 파일이 있을 때 다운로드 생략 |
| `NO_DENOISE=1 bash run.sh` | 노이즈 제거 건너뛰기 |
| `python transcribe.py --audio-dir /workspace/audio --out out.srt --beam-size 10` | 자막 추출만 직접 실행 |

## 4. 결과 받기

- 결과: `/workspace/output/audio_files.srt`
- **Jupyter** 버튼 → 파일탐색기 `output/audio_files.srt` 우클릭 → Download
- 또는 PC PowerShell 에서 `scp -P <포트> root@<IP>:/workspace/output/audio_files.srt .`

## 5. 문제 해결

- `References.txt 를 찾을 수 없습니다` → 드라이브 폴더 공유 설정이 **"링크가 있는 모든 사용자"** 인지 확인. gdown 이 일시적으로 막히면 잠시 후 재시도.
- 다운로드가 자주 실패하면 Jupyter 파일탐색기로 wav 파일들을 `/workspace/audio` 에 직접 업로드하고 `SKIP_DOWNLOAD=1 bash run.sh`.
- 설치를 처음부터 다시 하려면 `rm /workspace/.setup_done` 후 재실행.

## 파일

- `run.sh` — 설치 · 다운로드 · 실행 원커맨드 스크립트
- `transcribe.py` — 코랩 [셀 2] 그대로 옮긴 자막 추출 코드 (노이즈 제거 → bfloat16 모델 → SRT 병합)
