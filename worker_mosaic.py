import os
import re
import sys
import json
import shutil
import argparse
import threading
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--rclone-remote", type=str, default="gdrive")
parser.add_argument("--remote-dir", type=str, default="video")
parser.add_argument("--max-clip-length", type=str, default="5000")
args = parser.parse_args()

WORK_DIR        = "/workspace"
LADA_DIR        = "/workspace/lada"
LADA_BIN        = "/workspace/venv_lada/bin/lada-cli"
VIDEO_DIR       = os.path.join(WORK_DIR, "video")
LOCAL_WORK_DIR  = os.path.join(WORK_DIR, "workspace_mosaic")
REFERENCES_FILE = os.path.join(VIDEO_DIR, "References.txt")
REMOTE          = f"{args.rclone_remote}:{args.remote_dir}"

DET_MODEL_NAME  = "v4-accurate"
REST_MODEL_NAME = "basicvsrpp-v1.2"

# ★ torch 버전별 변수명이 달라 둘 다 설정
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
os.environ["TORCH_FORCE_WEIGHTS_ONLY_LOAD"]    = "0"

# ------------------------------------------------------------------ 0
print("▶ 0단계: 환경 점검", flush=True)
if not os.path.exists(LADA_BIN):
    raise FileNotFoundError(f"lada-cli 없음: {LADA_BIN}  → setup.sh 를 다시 실행하세요")

# ★ lada 는 CWD 기준 model_weights/ 를 찾는다 (코랩 원본과 동일 구조)
os.chdir(LADA_DIR)
for f_ in ["model_weights/lada_mosaic_detection_model_v4_accurate.pt",
           "model_weights/lada_mosaic_restoration_model_generic_v1.2.pth"]:
    if not os.path.exists(f_):
        raise FileNotFoundError(f"가중치 없음: {LADA_DIR}/{f_}  → download_models.py 실행")
print(f"  CWD={os.getcwd()} / 가중치 확인 OK", flush=True)

os.makedirs(VIDEO_DIR, exist_ok=True)
if os.path.exists(LOCAL_WORK_DIR):
    shutil.rmtree(LOCAL_WORK_DIR)         # ★ 이전 잔여물 정리
os.makedirs(LOCAL_WORK_DIR, exist_ok=True)

# ------------------------------------------------------------------ 1
print("▶ 1단계: 드라이브에서 video 폴더 다운로드", flush=True)
subprocess.run(
    ["rclone", "copy", REMOTE, VIDEO_DIR, "--exclude", "restored/**",
     "--progress", "--stats", "15s"],
    check=True,
)
if not os.path.exists(REFERENCES_FILE):
    raise FileNotFoundError(f"References.txt 없음: {REFERENCES_FILE}")

# ------------------------------------------------------------------ 2
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

print(f"  {meta.get('codec')} / {meta.get('fps')}fps / {meta.get('resolution')}", flush=True)
print(f"  복원 구간: {len(sections)}개", flush=True)

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

# ------------------------------------------------------------------ 3
print("▶ 3단계: 파일 검증", flush=True)
fatal = []
for sec in sections:
    p = os.path.join(VIDEO_DIR, sec["file"])
    if not os.path.exists(p):
        fatal.append(f"  ✘ {sec['file']} → 파일 없음")
        continue
    a_size, a_frames = os.path.getsize(p), get_frame_count(p)
    if a_size != sec["size"]:
        fatal.append(f"  ✘ {sec['file']} 크기 불일치 {sec['size']} → {a_size} (재업로드 필요)")
    elif abs(a_frames - sec["frames"]) > 2:
        print(f"  ⚠️ {sec['file']} 프레임 차이 {sec['frames']} → {a_frames} (±2 초과, 계속 진행)", flush=True)
    else:
        print(f"  ✔ {sec['file']}", flush=True)
if fatal:
    print("\n❌ 검증 실패:", flush=True)
    for d in fatal:
        print(d, flush=True)
    raise SystemExit("작업 중단 - 손상 파일 재업로드 후 재실행")

# ------------------------------------------------------------------ 4
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
        enc_parts += ["-cq", "18", "-b:v", "0"]      # ★ nvenc는 crf가 아니라 cq + b:v 0
else:
    print("  ⚠️ NVENC 미지원 → libx264/265 폴백", flush=True)
    target_encoder = "libx265" if orig_codec == "hevc" else "libx264"
    enc_parts = ["-preset", "medium"]
    if orig_bitrate:
        enc_parts += ["-b:v", str(int(orig_bitrate))]
    else:
        enc_parts += ["-crf", "18"]

for key, flag in [("color_space", "-colorspace"),
                  ("color_primaries", "-color_primaries"),
                  ("color_transfer", "-color_trc")]:
    val = meta.get(key, "unknown")
    if val and val != "unknown":
        enc_parts += [flag, val]

encoder_options_str = " ".join(enc_parts)
print(f"  인코더: {target_encoder}", flush=True)
print(f"  파라미터: {encoder_options_str}", flush=True)

# ------------------------------------------------------------------ 유틸
def remote_exists(rel_path):
    """★ 인스턴스 재생성 후에도 체크포인트가 살아있게 클라우드를 직접 조회"""
    r = subprocess.run(["rclone", "lsf", f"{REMOTE}/{rel_path}"],
                       capture_output=True, text=True)
    return r.returncode == 0 and r.stdout.strip() != ""

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
            if not ch:
                break
            if ch in ("\r", "\n"):
                line, buf = buf.strip(), ""
                if not line:
                    continue
                if any(k in line.lower() for k in ["error", "traceback", "exception", "failed"]):
                    print(f"  ⚠️ {line}", flush=True)
                    continue
                mf, ms = frame_pat.search(line), speed_pat.search(line)
                if mf and total_frames > 0:
                    done = int(mf.group(1))
                    pct  = min(int(done / total_frames * 100), 100)
                    spd  = f" | {ms.group(1)}fps" if ms else ""
                    rem  = ""
                    if ms and float(ms.group(1)) > 0:
                        rs = (total_frames - done) / float(ms.group(1))
                        rem = f" | 잔여 {int(rs//60)}분 {int(rs%60)}초"
                    if pct != last_pct[0]:
                        bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
                        print(f"  [{bar}] {pct}% ({done}/{total_frames}f{spd}{rem})", flush=True)
                        last_pct[0] = pct
            else:
                buf += ch

    t = threading.Thread(target=reader, daemon=True)
    t.start()
    proc.wait()
    t.join(timeout=5)
    return proc.returncode

def update_references(index, status, restored_frames=None):
    with open(REFERENCES_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()
    in_sec = False
    with open(REFERENCES_FILE, "w", encoding="utf-8") as f:
        for line in lines:
            s = line.strip()
            if s == "[SECTIONS]":
                in_sec = True; f.write(line); continue
            if s.startswith("[") and s.endswith("]"):
                in_sec = False; f.write(line); continue
            if in_sec and s:
                p = s.split("|")
                if len(p) >= 7 and p[0] == str(index):
                    p[6] = status
                    if restored_frames is not None:
                        if len(p) == 7: p.append(str(restored_frames))
                        else:           p[7] = str(restored_frames)
                    f.write("|".join(p) + "\n")
                    continue
            f.write(line)

def push_references():
    subprocess.run(["rclone", "copyto", REFERENCES_FILE, f"{REMOTE}/References.txt"],
                   check=False)

# ------------------------------------------------------------------ 5
print("▶ 5단계: 구간 복원 및 즉시 동기화", flush=True)
failed = []
for sec in sections:
    i = int(sec["index"])
    target_path    = os.path.join(VIDEO_DIR, sec["file"])
    local_target   = os.path.join(LOCAL_WORK_DIR, f"target_{i}.mp4")
    local_restored = os.path.join(LOCAL_WORK_DIR, f"restored_{i}_video.mp4")
    local_ready    = os.path.join(LOCAL_WORK_DIR, f"restored_{i}.mp4")
    remote_rel     = f"restored/restored_{i}.mp4"

    if sec["status"] == "done" and remote_exists(remote_rel):
        print(f"⏭️ [구간 {i}] 클라우드에 이미 존재 → 스킵", flush=True)
        continue

    print(f"\n🚀 [구간 {i}] {sec['start']} ~ {sec['end']}", flush=True)
    try:
        shutil.copy(target_path, local_target)
        total_frames = get_frame_count(local_target)
        print(f"  총 {total_frames}프레임", flush=True)

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
        env = os.environ.copy()
        rc = run_lada_with_progress(lada_cmd, env, total_frames)
        if rc != 0 or not os.path.exists(local_restored) or os.path.getsize(local_restored) == 0:
            raise RuntimeError(f"lada-cli 실패 (rc={rc})")

        print("  🔊 오디오 병합 중...", flush=True)
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error",
             "-i", local_restored, "-i", local_target,
             "-map", "0:v:0", "-map", "1:a:0?",
             "-c", "copy", "-muxdelay", "0", local_ready],
            check=True)

        restored_frames = get_frame_count(local_ready)
        print(f"  복원 프레임수: {restored_frames} (원본 {total_frames})", flush=True)

        print("  📤 드라이브 업로드 중...", flush=True)
        subprocess.run(["rclone", "copyto", local_ready, f"{REMOTE}/{remote_rel}",
                        "--progress", "--stats", "15s"], check=True)

        update_references(i, "done", restored_frames)
        push_references()
        print(f"✅ [구간 {i}] 완료 → {REMOTE}/{remote_rel}", flush=True)

    except Exception as e:
        print(f"❌ [구간 {i}] 실패: {e}", flush=True)
        failed.append(i)
        update_references(i, "failed")
        push_references()

    finally:
        # ★ 디스크 절약: 로컬 사본은 남기지 않음
        for fp in [local_target, local_restored, local_ready]:
            if os.path.exists(fp):
                os.remove(fp)

print("\n" + "=" * 50, flush=True)
if failed:
    print(f"⚠️ 실패 구간: {failed}  (References.txt 에 failed 기록됨)", flush=True)
    sys.exit(1)
print("✅ 모든 구간 복원 완료!", flush=True)
print(f"📁 위치: {REMOTE}/restored/", flush=True)
