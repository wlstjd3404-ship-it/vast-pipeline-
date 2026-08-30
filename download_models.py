import os
from huggingface_hub import hf_hub_download

MODEL_DIR = "/workspace/model_weights"
os.makedirs(MODEL_DIR, exist_ok=True)

det_file = "lada_mosaic_detection_model_v4_accurate.pt"
rest_file = "lada_mosaic_restoration_model_generic_v1.2.pth"

print("▶ 모델 다운로드 시작...")
hf_hub_download(repo_id="ladaapp/lada", filename=det_file, local_dir=MODEL_DIR)
hf_hub_download(repo_id="ladaapp/lada", filename=rest_file, local_dir=MODEL_DIR)
print("✔ 모델 다운로드 완료")