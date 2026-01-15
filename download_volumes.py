# download_volumes.py
from pathlib import Path
import modal
from huggingface_hub import snapshot_download

volume = modal.Volume.from_name("embedding-model-vol", create_if_missing=True)
MODEL_DIR = Path("/models")

image = (
    modal.Image.debian_slim()
    .pip_install("huggingface_hub[hf_transfer]")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

app = modal.App("download-embedding", image=image, volumes={MODEL_DIR: volume})

@app.function(timeout=600)  # ✅ removed duplicate volumes
def download_embedding():
    snapshot_download(
        repo_id="BAAI/bge-large-en-v1.5",
        local_dir=MODEL_DIR / "BAAI_bge-large-en-v1.5"
    )
    print("✅ Downloaded embedding model into /models/BAAI_bge-large-en-v1.5")