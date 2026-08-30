import os
import shutil
import json
import subprocess
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--rclone-remote", type=str, default="gdrive")
parser.add_argument("--max-clip-length", type=str, default="5000")
args = parser.parse_args()

WORK_DIR = "/workspace"
LADA_BIN = "/workspace/venv/bin/lada-cli"
DRIVE_VIDEO_DIR = os.path.join(WORK_DIR, "video")
RESTORED_DIR = os.path.join(DRIVE_VIDEO_DIR, "restored")
LOCAL_WORK_DIR = os.path.join(WORK_DIR, "workspace_mosaic")
REFERENCES_FILE = os.path.join(DRIVE_VIDEO_DIR, "References.txt")

os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'
os.makedirs(RESTORED_DIR, exist_ok=True)
os.makedirs(LOCAL_WORK_DIR, exist_ok=True)

print("▶ 1단계: 구글 드라이브에서 video 폴더 다운로드", flush=True)
subprocess.run(
    ["rclone", "copy", f"{args.rclone_remote}:video", DRIVE_VIDEO_DIR, "--exclude", "restored/**"],
    check=True
)

if not os.path.exists(REFERENCES_FILE):
    raise FileNotFoundError(f"References.txt 없음: {REFERENCES_FILE}")

print("▶ 2단계: References.txt 파싱", flush=True)
meta = {}
sections = []
mode = None

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
            parts = line.split("|")
            sections.append({
                "index":   parts[0],
                "start":   parts[1],
                "end":     parts[2],
                "file":    parts[3],
                "size":    int(parts[4]),
                "frames":  int(parts[5]),
                "status":  parts[6],
            })

def get_frame_count(path):
    r = subprocess.run(
        ['ffprobe', '-v', 'quiet', '-print_format', 'json',
         '-show_streams', '-select_streams', 'v:0', path],
        capture_output=True, text=True
    )
    try:
        info = json.loads(r.stdout)['streams'][0]
        fps_raw = info.get('r_frame_rate', '30/1')
        num, den = map(int, fps_raw.split('/'))
        fps = num / den
        duration = float(info.get('duration', 0))
        nb = info.get('nb_frames')
        return int(nb) if nb else int(round(fps * duration))
    except Exception:
        return 0

print("▶ 3단계: 파일 검증 (±2 프레임 허용)", flush=True)
for sec in sections:
    target_path = os.path.join(DRIVE_VIDEO_DIR, sec['file'])
    if not os.path.exists(target_path):
        raise FileNotFoundError(f"필수 파일 없음: {sec['file']}")
    actual_frames = get_frame_count(target_path)
    if abs(actual_frames - sec["frames"]) > 2:
        print(f"⚠️ 프레임 수 차이 발생: {sec['file']} (예상 {sec['frames']} / 실제 {actual_frames})", flush=True)

print("▶ 4단계: 인코더 설정", flush=True)
orig_codec = meta.get("codec", "h264")
orig_bitrate = meta.get("bitrate", "")

nvenc_test = subprocess.run("ffmpeg -f lavfi -i nullsrc -c:v hevc_nvenc -t 1 -f null -", shell=True, capture_output=True)
if nvenc_test.returncode == 0:
    target_encoder = "hevc_nvenc" if orig_codec == "hevc" else "h264_nvenc"
    enc_parts = ["-preset", "p7", "-tune", "hq", "-rc", "vbr"]
    if orig_bitrate:
        br = int(orig_bitrate)
        enc_parts += ["-b:v", str(br), "-maxrate", str(int(br * 1.5)), "-bufsize", str(br * 2)]
    else:
        enc_parts += ["-cq", "18"]
else:
    print("⚠️ NVENC 미지원 환경입니다. libx264/libx265로 폴백합니다.", flush=True)
    target_encoder = "libx265" if orig_codec == "hevc" else "libx264"
    enc_parts = ["-preset", "medium"]
    if orig_bitrate:
        br = int(orig_bitrate)
        enc_parts += ["-b:v", str(br)]
    else:
        enc_parts += ["-crf", "18"]

for key, flag in [("color_space", "-colorspace"), ("color_primaries", "-color_primaries"), ("color_transfer", "-color_trc")]:
    val = meta.get(key, "unknown")
    if val and val != "unknown":
        enc_parts += [flag, val]
encoder_options_str = " ".join(enc_parts)

def update_references(index, status, restored_frames=None):
    with open(REFERENCES_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()
    in_sections = False
    with open(REFERENCES_FILE, "w", encoding="utf-8") as f:
        for line in lines:
            stripped = line.strip()
            if stripped == "[SECTIONS]":
                in_sections = True
                f.write(line)
                continue
            elif stripped.startswith("[") and stripped.endswith("]"):
                in_sections = False
                f.write(line)
                continue

            if in_sections and stripped:
                parts = stripped.split("|")
                if len(parts) >= 7 and parts[0] == str(index):
                    parts[6] = status
                    if restored_frames is not None:
                        if len(parts) == 7:
                            parts.append(str(restored_frames))
                        else:
                            parts[7] = str(restored_frames)
                    f.write("|".join(parts) + "\n")
                    continue
            f.write(line)

print("▶ 5단계: 구간 복원 및 즉시 동기화", flush=True)
for sec in sections:
    i = int(sec["index"])
    target_name = sec["file"]
    target_path = os.path.join(DRIVE_VIDEO_DIR, target_name)
    local_target = os.path.join(LOCAL_WORK_DIR, f"target_{i}.mp4")
    local_restored = os.path.join(LOCAL_WORK_DIR, f"restored_{i}_video.mp4")
    local_ready = os.path.join(LOCAL_WORK_DIR, f"restored_{i}.mp4")
    drive_restored = os.path.join(RESTORED_DIR, f"restored_{i}.mp4")

    # 기존 완료 상태 확인
    if sec["status"] == "done" and os.path.exists(drive_restored):
        print(f"⏭️ [구간 {i}] 이미 완료됨 → 스킵", flush=True)
        continue

    print(f"\n🚀 [구간 {i}] {sec['start']} ~ {sec['end']} 시작", flush=True)
    shutil.copy(target_path, local_target)

    lada_cmd = [
        LADA_BIN,
        '--input', local_target,
        '--output', local_restored,
        '--device', 'cuda',
        '--mosaic-detection-model', 'v4-accurate',
        '--mosaic-restoration-model', 'basicvsrpp-v1.2',
        '--no-detect-face-mosaics',
        '--max-clip-length', str(args.max_clip_length),
        '--encoder', target_encoder,
        '--encoder-options', encoder_options_str,
    ]

    try:
        proc = subprocess.run(lada_cmd, check=True, stdout=sys.stdout, stderr=sys.stderr)
        
        subprocess.run(
            ['ffmpeg', '-y', '-v', 'error',
             '-i', local_restored, '-i', local_target,
             '-map', '0:v:0', '-map', '1:a:0?',
             '-c', 'copy', '-muxdelay', '0', local_ready],
            check=True
        )

        restored_frames = get_frame_count(local_ready)
        shutil.copy(local_ready, drive_restored)
        update_references(i, "done", restored_frames)

        # 구글 드라이브 즉시 단일 파일 업로드
        subprocess.run(
            ["rclone", "copyto", local_ready, f"{args.rclone_remote}:video/restored/restored_{i}.mp4"],
            check=True
        )
        subprocess.run(
            ["rclone", "copyto", REFERENCES_FILE, f"{args.rclone_remote}:video/References.txt"],
            check=True
        )
        print(f"✅ [구간 {i}] 완료 및 클라우드 업로드 성공", flush=True)

    except Exception as e:
        print(f"❌ [구간 {i}] 실패: {e}", flush=True)
        update_references(i, "failed")
        subprocess.run(
            ["rclone", "copyto", REFERENCES_FILE, f"{args.rclone_remote}:video/References.txt"],
            check=False
        )

    finally:
        for fp in [local_target, local_restored, local_ready]:
            if os.path.exists(fp):
                os.remove(fp)

print("\n✔ 전체 복원 파이프라인 종료", flush=True)