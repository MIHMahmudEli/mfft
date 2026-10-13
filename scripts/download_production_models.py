"""
Download verified production checkpoints from Hugging Face Hub (MohsinElis/mfft-checkpoints)
into model/checkpoints/production/ for local serving and containerized deployment.
Repository is public: anonymous downloads work without an API token.
"""
import os
import shutil
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parent.parent
REPO_ID = "MohsinElis/mfft-checkpoints"
TARGET_DIR = ROOT / "model" / "checkpoints" / "production"
TARGET_DIR.mkdir(parents=True, exist_ok=True)

VARIANTS = {
    "tiny": "runs/mfft_tiny/checkpoints/best.pt",
    "base": "runs/mfft_base/checkpoints/best.pt",
    "large": "runs/mfft_large/checkpoints/best.pt",
}

def main():
    token = os.environ.get("HF_TOKEN", None)
    print("Downloading full-scale 20-epoch production checkpoints from Hugging Face (MohsinElis/mfft-checkpoints)...")
    for variant, repo_path in VARIANTS.items():
        dest = TARGET_DIR / f"{variant}.pt"
        print(f"  Fetching {variant} from {repo_path}...")
        try:
            downloaded = hf_hub_download(
                repo_id=REPO_ID,
                filename=repo_path,
                repo_type="model",
                token=token,
                force_download=True
            )
            shutil.copy2(downloaded, dest)
            size_mb = dest.stat().st_size / (1024 * 1024)
            print(f"  --> Saved {variant}.pt ({size_mb:.2f} MB)")
        except Exception as e:
            print(f"  [ERROR] Failed to download {variant}: {e}")

    print("\nProduction checkpoints ready in:", TARGET_DIR)

if __name__ == "__main__":
    main()
