import subprocess
import threading
import gradio as gr

PYTHON_BIN = "/workspace/venv/bin/python"
is_running = False
lock = threading.Lock()

def stream_process(cmd):
    global is_running
    with lock:
        if is_running:
            yield "⚠️ 이미 다른 작업이 진행 중입니다. 완료 후 다시 시도하세요."
            return
        is_running = True

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        logs = ""
        for line in iter(process.stdout.readline, ''):
            logs += line
            yield logs
        process.wait()
        if process.returncode == 0:
            logs += "\n🎉 작업이 성공적으로 완료되었습니다!"
        else:
            logs += f"\n❌ 작업 중 오류 발생 (종료 코드: {process.returncode})"
        yield logs
    finally:
        with lock:
            is_running = False

def run_mosaic(rclone_remote, max_clip):
    # 특수문자 차단
    safe_remote = "".join(c for c in rclone_remote if c.isalnum() or c in "_-")
    cmd = [
        PYTHON_BIN, "-u", "/workspace/repo/worker_mosaic.py",
        "--rclone-remote", safe_remote,
        "--max-clip-length", str(max_clip)
    ]
    yield from stream_process(cmd)

def run_stt(rclone_remote, beam_size):
    safe_remote = "".join(c for c in rclone_remote if c.isalnum() or c in "_-")
    cmd = [
        PYTHON_BIN, "-u", "/workspace/repo/worker_stt.py",
        "--rclone-remote", safe_remote,
        "--beam-size", str(beam_size)
    ]
    yield from stream_process(cmd)

with gr.Blocks(title="Vast.ai Cloud Pipeline") as demo:
    gr.Markdown("# 🚀 Vast.ai 클라우드 파이프라인")
    rclone_remote = gr.Textbox(label="Rclone 원격 이름", value="gdrive")

    with gr.Tab("1. 모자이크 제거"):
        max_clip = gr.Radio(choices=["5000", "1000"], value="5000", label="--max-clip-length")
        btn_mosaic = gr.Button("▶ 모자이크 제거 시작", variant="primary")
        out_mosaic = gr.TextArea(label="실시간 로그", lines=15, autoscroll=True)
        btn_mosaic.click(fn=run_mosaic, inputs=[rclone_remote, max_clip], outputs=out_mosaic)

    with gr.Tab("2. 자막 추출"):
        beam_size = gr.Radio(choices=["10", "20"], value="10", label="beam_size")
        btn_stt = gr.Button("▶ 자막 추출 시작", variant="primary")
        out_stt = gr.TextArea(label="실시간 로그", lines=15, autoscroll=True)
        btn_stt.click(fn=run_stt, inputs=[rclone_remote, beam_size], outputs=out_stt)

if __name__ == "__main__":
    # 보안을 위해 아이디/비밀번호를 설정합니다.
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        auth=("admin", "1234")
    )