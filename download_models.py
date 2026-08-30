import os
import shutil
from huggingface_hub import hf_hub_download

MODEL_DIR = "/workspace/lada/model_weights"
os.makedirs(MODEL_DIR, exist_ok=True)

FILES = [
    "lada_mosaic_detection_model_v4_accurate.pt",
    "lada_mosaic_restoration_model_generic_v1.2.pth",
]

print("▶ 모델 다운로드 시작...", flush=True)
for fname in FILES:
    dst = os.path.join(MODEL_DIR, fname)
    if os.path.exists(dst) and os.path.getsize(dst) > 1024 * 1024:
        print(f"  ⏭️  이미 존재: {fname}", flush=True)
        continue
    print(f"  📥 {fname}", flush=True)
    cached = hf_hub_download(repo_id="ladaapp/lada", filename=fname)
    shutil.copy(cached, dst)     # 심볼릭 링크가 아닌 실제 파일로
    print(f"     → {dst} ({os.path.getsize(dst)/1e6:.1f} MB)", flush=True)

print("✔ 모델 다운로드 완료", flush=True)
