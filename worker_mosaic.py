import os
import re
import sys
import json
import shutil
import argparse
import threading
import subprocess
import gdown

parser = argparse.ArgumentParser()
parser.add_argument("--drive-url", type=str, required=True)
parser.add_argument("--max-clip-length", type=str, default="5000")
args = parser.parse_args()

WORK_DIR        = "/workspace"
LADA_DIR        = "/workspace/lada"
LADA_BIN        = "/workspace/venv_lada/bin/lada-cli"
VIDEO_DIR       = os.path.join(WORK_DIR, "video")
LOCAL_WORK_DIR  = os.path.join(WORK_DIR, "workspace_mosaic")
OUTPUT_DIR      = os.path.join(WORK_DIR, "output_mosaic")
REFERENCES_FILE = os.path.join(VIDEO_DIR, "References.txt")

DET_MODEL_NAME  = "v4-accurate"
REST_MODEL_NAME = "basicvsrpp-v1.2"

os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
os.environ["TORCH_FORCE_WEIGHTS_ONLY_LOAD"]    = "0"

print("▶ 0단계: 환경 점검", flush=True)
if not os.path.exists(LADA_BIN):
    raise FileNotFoundError(f"lada-cli 없음: {LADA_BIN} → setup.sh 를 다시 실행하세요")

os.chdir(LADA_DIR)
for f_ in ["model_weights/lada_mosaic_detection_model_v4_accurate.pt",
           "model_weights/lada_mosaic_restoration_model_generic_v1.2.pth"]:
    if not os.path.exists(f_):
        raise FileNotFoundError(f"가중치 없음: {LADA_DIR}/{f_} → download_models.py 실행")

os.makedirs(VIDEO_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
if os.path.exists(LOCAL_WORK_DIR):
    shutil.rmtree(LOCAL_WORK_DIR)
os.makedirs(LOCAL_WORK_DIR, exist_ok=True)

print("▶ 1단계: 공개 URL에서 video 폴더 다운로드", flush=True)
gdown.download_folder(url=args.drive_url, output=VIDEO_DIR, quiet=False, use_cookies=False)

if not os.path.exists(REFERENCES_FILE):
    raise FileNotFoundError(f"References.txt 없음: {REFERENCES_FILE}")

print("▶ 2단계: References.txt 파싱", flush=True)
meta, sections, mode = {}, [], None
with open(REFERENCES_FILE, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        if line in ("[META]", "[SECTIONS]", "[GAP_FILES]", "[MERGE_ORDER]"):
            mode = line
            continue
        if mode == "[META]":
            k, _, v = line.partition("=")
            meta[k.strip()] = v.strip()
        elif mode == "[SECTIONS]":
            p = line.split("|")
            sections.append({"index": p[0], "start": p[1], "end": p[2], "file": p[3],
                             "size": int(p[4]), "frames": int(p[5]), "status": p[6]})

def get_frame_count(path):
    r = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json",
         "-show_streams", "-select_streams", "v:0", path],
        capture_output=True, text=True)
    try:
        info = json.loads(r.stdout)["streams"][0]
        num, den = map(int, info.get("r_frame_rate", "30/1").split("/"))
        fps = num / den if den else 30.0
        nb = info.get("nb_frames")
        return int(nb) if nb else int(round(fps * float(info.get("duration", 0))))
    except Exception:
        return 0

print("▶ 3단계: 파일 검증", flush=True)
fatal = []
for sec in sections:
    p = os.path.join(VIDEO_DIR, sec["file"])
    if not os.path.exists(p):
        fatal.append(f"  ✘ {sec['file']} → 파일 없음")
        continue
    a_size, a_frames = os.path.getsize(p), get_frame_count(p)
    if a_size != sec["size"]:
        fatal.append(f"  ✘ {sec['file']} 크기 불일치 {sec['size']} → {a_size}")
    else:
        print(f"  ✔ {sec['file']}", flush=True)
if fatal:
    raise SystemExit("\n❌ 검증 실패: 작업 중단")

print("▶ 4단계: 인코더 설정", flush=True)
orig_codec   = meta.get("codec", "h264")
orig_bitrate = meta.get("bitrate", "")

nvenc_ok = subprocess.run(
    "ffmpeg -f lavfi -i nullsrc -c:v hevc_nvenc -t 1 -f null -",
    shell=True, capture_output=True).returncode == 0

if nvenc_ok:
    target_encoder = "hevc_nvenc" if orig_codec == "hevc" else "h264_nvenc"
    enc_parts = ["-preset", "p7", "-tune", "hq", "-rc", "vbr"]
    if orig_bitrate:
        br = int(orig_bitrate)
        enc_parts += ["-b:v", str(br), "-maxrate", str(int(br * 1.5)), "-bufsize", str(br * 2)]
    else:
        enc_parts += ["-cq", "18", "-b:v", "0"]
else:
    target_encoder = "libx265" if orig_codec == "hevc" else "libx264"
    enc_parts = ["-preset", "medium"]
    if orig_bitrate:
        enc_parts += ["-b:v", str(int(orig_bitrate))]
    else:
        enc_parts += ["-crf", "18"]

encoder_options_str = " ".join(enc_parts)

def run_lada_with_progress(cmd, env, total_frames):
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                            text=True, env=env, bufsize=1)
    frame_pat = re.compile(r"\((\d+)f\)")
    speed_pat = re.compile(r"Speed:\s*([\d.]+)fps")
    last_pct = [-1]

    def reader():
        buf = ""
        while True:
            ch = proc.stderr.read(1)
            if not ch: break
            if ch in ("\r", "\n"):
                line, buf = buf.strip(), ""
                if not line: continue
                mf, ms = frame_pat.search(line), speed_pat.search(line)
                if mf and total_frames > 0:
                    done = int(mf.group(1))
                    pct  = min(int(done / total_frames * 100), 100)
                    spd  = f" | {ms.group(1)}fps" if ms else ""
                    if pct != last_pct[0]:
                        bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
                        print(f"  [{bar}] {pct}% ({done}/{total_frames}f{spd})", flush=True)
                        last_pct[0] = pct
            else:
                buf += ch

    t = threading.Thread(target=reader, daemon=True)
    t.start()
    proc.wait()
    t.join(timeout=5)
    return proc.returncode

print("▶ 5단계: 구간 복원 (로컬 저장)", flush=True)
failed = []
for sec in sections:
    i = int(sec["index"])
    target_path    = os.path.join(VIDEO_DIR, sec["file"])
    local_target   = os.path.join(LOCAL_WORK_DIR, f"target_{i}.mp4")
    local_restored = os.path.join(LOCAL_WORK_DIR, f"restored_{i}_video.mp4")
    final_output   = os.path.join(OUTPUT_DIR, f"restored_{i}.mp4")

    if os.path.exists(final_output) and os.path.getsize(final_output) > 0:
        print(f"⏭️ [구간 {i}] 이미 완료됨 → 스킵", flush=True)
        continue

    print(f"\n🚀 [구간 {i}] {sec['start']} ~ {sec['end']}", flush=True)
    try:
        shutil.copy(target_path, local_target)
        total_frames = get_frame_count(local_target)

        lada_cmd = [
            LADA_BIN,
            "--input",  local_target,
            "--output", local_restored,
            "--device", "cuda",
            "--mosaic-detection-model",   DET_MODEL_NAME,
            "--mosaic-restoration-model", REST_MODEL_NAME,
            "--no-detect-face-mosaics",
            "--max-clip-length", str(args.max_clip_length),
            "--encoder", target_encoder,
            "--encoder-options", encoder_options_str,
        ]
        rc = run_lada_with_progress(lada_cmd, os.environ.copy(), total_frames)
        if rc != 0 or not os.path.exists(local_restored):
            raise RuntimeError(f"lada-cli 실패 (rc={rc})")

        print("  🔊 오디오 병합 중...", flush=True)
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error",
             "-i", local_restored, "-i", local_target,
             "-map", "0:v:0", "-map", "1:a:0?",
             "-c", "copy", "-muxdelay", "0", final_output],
            check=True)

        print(f"✅ [구간 {i}] 완료 → {final_output}", flush=True)
    except Exception as e:
        print(f"❌ [구간 {i}] 실패: {e}", flush=True)
        failed.append(i)
    finally:
        for fp in [local_target, local_restored]:
            if os.path.exists(fp):
                os.remove(fp)

if failed:
    sys.exit(1)
print(f"\n✅ 모든 구간 복원 완료! 로컬 저장 위치: {OUTPUT_DIR}", flush=True)