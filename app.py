import os
import glob
import threading
import subprocess
from collections import deque

import gradio as gr

STT_PY   = "/workspace/venv_stt/bin/python"
LADA_PY  = "/workspace/venv_lada/bin/python"
REPO     = "/workspace/repo"
MAX_LOG_LINES = 800

is_running = False
lock = threading.Lock()


def _nvidia_ld_path(py_bin: str) -> str:
    """해당 venv 안의 nvidia/*/lib 디렉터리를 LD_LIBRARY_PATH 형식으로 반환.
    ★ libcusparseLt.so.0 등 CUDA 서브라이브러리 로드 실패 방지"""
    venv = os.path.dirname(os.path.dirname(py_bin))
    dirs = sorted(glob.glob(f"{venv}/lib/python*/site-packages/nvidia/*/lib"))
    return ":".join(dirs)


def stream_process(cmd):
    """★ lada 진행률은 \\r 로만 출력되므로 문자 단위로 읽어 실시간 반영"""
    global is_running
    with lock:
        if is_running:
            yield "⚠️ 이미 다른 작업이 진행 중입니다. 완료 후 다시 시도하세요."
            return
        is_running = True

    lines = deque(maxlen=MAX_LOG_LINES)
    try:
        if not os.path.exists(cmd[0]):
            yield f"❌ 파이썬 실행 파일이 없습니다: {cmd[0]}\n→ setup.sh 를 다시 실행하세요."
            return

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        # ★ 호출 대상 venv 기준으로 CUDA 라이브러리 경로 주입
        nv = _nvidia_ld_path(cmd[0])
        if nv:
            env["LD_LIBRARY_PATH"] = nv + ":" + env.get("LD_LIBRARY_PATH", "")

        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=0, env=env, cwd=REPO,
        )

        buf = ""
        while True:
            ch = proc.stdout.read(1)
            if not ch:
                break
            if ch in ("\r", "\n"):
                if buf.strip():
                    lines.append(buf.rstrip())
                    yield "\n".join(lines)
                buf = ""
            else:
                buf += ch
        if buf.strip():
            lines.append(buf.rstrip())

        proc.wait()
        lines.append("")
        lines.append("🎉 작업이 성공적으로 완료되었습니다!" if proc.returncode == 0
                     else f"❌ 오류 발생 (종료 코드: {proc.returncode})")
        yield "\n".join(lines)
    finally:
        with lock:
            is_running = False


def _safe(name):
    return "".join(c for c in name if c.isalnum() or c in "_-") or "gdrive"


def run_mosaic(remote, max_clip):
    yield from stream_process([
        LADA_PY, "-u", f"{REPO}/worker_mosaic.py",
        "--rclone-remote", _safe(remote),
        "--max-clip-length", str(max_clip),
    ])


def run_stt(remote, beam_size):
    yield from stream_process([
        STT_PY, "-u", f"{REPO}/worker_stt.py",
        "--rclone-remote", _safe(remote),
        "--beam-size", str(beam_size),
    ])


with gr.Blocks(title="Vast.ai Cloud Pipeline") as demo:
    gr.Markdown("# 🚀 Vast.ai 클라우드 파이프라인")
    remote = gr.Textbox(label="Rclone 원격 이름", value="gdrive")

    with gr.Tab("1. 모자이크 제거"):
        max_clip = gr.Radio(["5000", "1000"], value="5000",
                            label="--max-clip-length (VR 영상은 1000)")
        b1 = gr.Button("▶ 모자이크 제거 시작", variant="primary")
        o1 = gr.TextArea(label="실시간 로그", lines=22, autoscroll=True)
        b1.click(run_mosaic, [remote, max_clip], o1)

    with gr.Tab("2. 자막 추출"):
        beam = gr.Radio(["10", "20"], value="10", label="beam_size")
        b2 = gr.Button("▶ 자막 추출 시작", variant="primary")
        o2 = gr.TextArea(label="실시간 로그", lines=22, autoscroll=True)
        b2.click(run_stt, [remote, beam], o2)

if __name__ == "__main__":
    port = int(os.environ.get("GRADIO_PORT", "7860"))
    user = os.environ.get("UI_USER", "admin")
    pw   = os.environ.get("UI_PASS", "changeme1234")
    kw = dict(server_name="0.0.0.0", server_port=port, auth=(user, pw),
              show_error=True)
    try:
        # ★ vast 방화벽에서 frpc 차단 시 launch 자체가 죽는 것을 방지
        demo.queue().launch(share=True, **kw)
    except Exception as e:
        print(f"⚠️ share 링크 생성 실패({e}) → 로컬 포트만 개방", flush=True)
        demo.queue().launch(share=False, **kw)
