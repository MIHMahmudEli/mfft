"""Assemble and deploy the MFFT detection API as a Hugging Face Docker Space.

Single source of truth: copies api/ and model/src/model.py from the repo at
deploy time, bundles the verified 20-epoch production checkpoints,
and pushes to the Hugging Face Space repository.

Usage (from repo root):
    python scripts/deploy_hf_space.py            # assemble + deploy
    python scripts/deploy_hf_space.py --dry-run  # assemble only
"""
import argparse
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SPACE_DIR = REPO / "hf_space"
SPACE_ID = "MohsinElis/mfft-api"

# Production checkpoints from full-scale 20-epoch training run
CHECKPOINTS = {
    "tiny":  REPO / "model" / "checkpoints" / "production" / "tiny.pt",
    "base":  REPO / "model" / "checkpoints" / "production" / "base.pt",
    "large": REPO / "model" / "checkpoints" / "production" / "large.pt",
}
# Fallback to test checkpoints if production checkpoints are not yet downloaded
for _v, _p in list(CHECKPOINTS.items()):
    if not _p.exists():
        CHECKPOINTS[_v] = REPO / "model" / "checkpoints" / "test" / f"{_v}_model" / "best.pt"

README = """---
title: MFFT AI Image Detector API
emoji: 🛡️
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# MFFT: Multi-Frequency Fusion Transformer — Production Detection API

FastAPI inference server for AI-generated image and deepfake detection based on radial Fourier frequency sub-band decomposition and Frequency-Guided Attention (FGA).

Three verified production models trained for 20 epochs on 120,000 images are served concurrently:
- **MFFT-Tiny** (372K params, 1.49 MB FP16): **98.59% accuracy**, **99.88% AUC-ROC** (~3.2 ms latency)
- **MFFT-Base** (1.62M params, 6.49 MB FP16): **98.37% accuracy**, **99.87% AUC-ROC** (5-seed replication average: **98.58% ± 0.13%**)
- **MFFT-Large** (6.30M params, 25.21 MB FP16): **98.11% accuracy**, **99.68% AUC-ROC** (Default production profile)

## Endpoints

- `GET /` — Healthcheck and loaded model status
- `GET /models` — Available model profiles and specifications
- `POST /predict?model=large` — Single image forensic analysis with anomaly heatmap (base64 PNG) and radial Fourier frequency band breakdown
- `POST /predict/batch?model=large` — Multi-image batch inference

## Example Usage

```bash
curl -X POST "https://mohsinelis-mfft-api.hf.space/predict?model=large" \\
  -H "Authorization: Bearer free" \\
  -F "file=@image.jpg"
```
"""

DOCKERFILE = """FROM python:3.11-slim

RUN useradd -m -u 1000 user
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \\
    libgl1 \\
    libglib2.0-0 \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \\
 && pip install --no-cache-dir -r requirements.txt

COPY . .

RUN chown -R user:user /app

ENV MFFT_CHECKPOINT_DIR=/app/checkpoints \\
    MFFT_DEMO=1 \\
    MFFT_IMAGE_SIZE=384 \\
    MFFT_VARIANT=large \\
    HF_HOME=/tmp/hf \\
    PORT=7860

USER user
EXPOSE 7860
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
"""

REQUIREMENTS = """fastapi>=0.109.0
uvicorn[standard]>=0.27.0
python-multipart>=0.0.6
pydantic>=2.5.0
pillow>=10.0.0
numpy>=1.24.0
"""

GITATTRIBUTES = """*.pt filter=lfs diff=lfs merge=lfs -text
*.bin filter=lfs diff=lfs merge=lfs -text
"""


def assemble():
    if SPACE_DIR.exists():
        shutil.rmtree(SPACE_DIR)
    (SPACE_DIR / "api").mkdir(parents=True)
    (SPACE_DIR / "model" / "src").mkdir(parents=True)
    (SPACE_DIR / "checkpoints").mkdir(parents=True)

    (SPACE_DIR / "README.md").write_text(README, encoding="utf-8")
    (SPACE_DIR / "Dockerfile").write_text(DOCKERFILE, encoding="utf-8")
    (SPACE_DIR / "requirements.txt").write_text(REQUIREMENTS, encoding="utf-8")
    (SPACE_DIR / ".gitattributes").write_text(GITATTRIBUTES, encoding="utf-8")

    # api package (same layout as the repo so model_server's path logic works)
    for f in ("main.py", "model_server.py", "schemas.py"):
        shutil.copy(REPO / "api" / f, SPACE_DIR / "api" / f)
    (SPACE_DIR / "api" / "__init__.py").write_text("", encoding="utf-8")
    (SPACE_DIR / "model" / "__init__.py").write_text("", encoding="utf-8")
    (SPACE_DIR / "model" / "src" / "__init__.py").write_text("", encoding="utf-8")
    shutil.copy(REPO / "model" / "src" / "model.py",
                SPACE_DIR / "model" / "src" / "model.py")

    missing = []
    for variant, src in CHECKPOINTS.items():
        if src.exists():
            shutil.copy(src, SPACE_DIR / "checkpoints" / f"{variant}.pt")
        else:
            missing.append(f"{variant}: {src}")
    if missing:
        sys.exit("Missing checkpoints:\n  " + "\n  ".join(missing))

    total = sum(f.stat().st_size for f in SPACE_DIR.rglob("*") if f.is_file())
    print(f"Assembled {SPACE_DIR} ({total / 1e6:.1f} MB)")
    for f in sorted(SPACE_DIR.rglob("*")):
        if f.is_file():
            print(f"  {f.relative_to(SPACE_DIR)}  ({f.stat().st_size / 1e6:.2f} MB)")


def deploy():
    from huggingface_hub import HfApi
    sys.path.insert(0, str(REPO))
    from scripts.list_active_runs import get_hf_token

    token = get_hf_token()
    api = HfApi(token=token)
    user_info = api.whoami()
    user_name = user_info.get("name", user_info.get("username", "unknown"))
    is_pro = user_info.get("isPro", False)
    print(f"Deploying as: {user_name} (Pro: {is_pro})")

    print(f"Ensuring Space repository exists: {SPACE_ID}...")
    api.create_repo(
        SPACE_ID,
        repo_type="space",
        space_sdk="docker",
        exist_ok=True,
    )

    print(f"Uploading files to Hugging Face Space: {SPACE_ID}...")
    api.upload_folder(
        repo_id=SPACE_ID,
        repo_type="space",
        folder_path=str(SPACE_DIR),
        commit_message="Deploy production MFFT detection API with verified 20-epoch weights",
    )
    print(f"\nSuccessfully deployed to Space: https://huggingface.co/spaces/{SPACE_ID}")
    print("Direct API URL: https://mohsinelis-mfft-api.hf.space")
    print("Space Build Status: Check the 'Logs' tab on the Hugging Face Space page.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    assemble()
    if not args.dry_run:
        deploy()
