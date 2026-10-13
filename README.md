# MFFT: Multi-Frequency Fusion Transformer for Explainable AI-Generated Image Detection

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23339178.svg)](https://doi.org/10.5281/zenodo.23339178)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg?style=flat&logo=pytorch)](https://pytorch.org/)
[![Hugging Face Models](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Checkpoints-yellow.svg)](https://huggingface.co/MohsinElis/mfft-checkpoints)
[![Hugging Face Manifest](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Master%20Manifest-blue.svg)](https://huggingface.co/MohsinElis/mfft-master-manifest)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Live Demo](https://img.shields.io/badge/Demo-ImageVerify%20AI-000000.svg?logo=vercel)](https://imageverify-ai.vercel.app/)
[![API Status](https://img.shields.io/badge/API-Render%20Online-46E3B7.svg?logo=render)](https://mfft-detection-api.onrender.com/)

Official open-source PyTorch implementation, pretrained models, and evaluation pipelines for the **Multi-Frequency Fusion Transformer (MFFT)**, a parameter-efficient vision architecture designed to detect AI-generated imagery and deepfakes across radial Fourier frequency sub-bands using depthwise separable convolutions and Frequency-Guided Cross-Attention (FGA).

---

## 🌟 Key Highlights

- **Frequency-Domain Forensic Representation**: Decomposes 2D Fast Fourier Transform (FFT) spectra into discrete radial annular sub-bands (Low, Mid, High), exposing subtle upsampling artifacts and spectral grid discrepancies imperceptible in spatial RGB pixel space.
- **Frequency-Guided Cross-Attention (FGA)**: Cross-attends spatial features guided by spectral magnitude weights, directing the transformer's receptive field to artifact-dense frequency components.
- **Extreme Parameter Efficiency**:
  - **MFFT-Tiny** (372K params): **98.59% accuracy**, **99.88% AUC-ROC** (~12ms inference latency).
  - **MFFT-Base** (1.62M params): **98.37% accuracy**, **99.87% AUC-ROC** (5-seed mean: **98.58% ± 0.13%**).
  - **MFFT-Large** (6.30M params): **98.11% accuracy**, **99.68% AUC-ROC** (maximum forensic representation capacity).
- **Comprehensive Benchmarks**: Rigorously evaluated on a frozen, stratified held-out benchmark of **18,000 unseen images** (84K train / 18K val / 18K test) against 10 baseline architectures including foundation models, Vision Transformers, and modern CNN backbones.
- **Out-of-Distribution Generalization**: Evaluated under Leave-One-Generator-Out (LOGO) cross-generator validation across latent diffusion (SDXL, SD 1.5, GLIDE), autoregressive transformers (DALL-E 3), GANs (StyleGAN, ProGAN, BigGAN), and facial deepfakes (DFDC, Celeb-DF v2, FaceForensics++).
- **Inherent Interpretability**: Produces high-resolution spatial forensic heatmaps and per-frequency band anomaly contribution breakdowns alongside confidence calibration.

---

## 🔬 Architecture Overview

```
Input Image (384x384 RGB)
         │
         ▼
┌────────────────────────────────────────────────────────┐
│     Radial 2D-FFT Decomposition (Low / Mid / High)     │
└──────────────┬──────────────────┬──────────────────┬───┘
               │                  │                  │
               ▼                  ▼                  ▼
       ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
       │   Low-Band   │   │   Mid-Band   │   │  High-Band   │
       │ Feature Net  │   │ Feature Net  │   │ Feature Net  │
       └───────┬──────┘   └───────┬──────┘   └───────┬──────┘
               │                  │                  │
               └──────────────┬───┴──────────────────┘
                              │
                              ▼
       ┌─────────────────────────────────────────────────┐
       │    Frequency-Guided Cross-Attention (FGA)       │
       │   - Computes dynamic spectral cross-attention   │
       │   - Discovers spatial-frequency correspondences │
       └──────────────────────┬──────────────────────────┘
                              │
                              ▼
       ┌─────────────────────────────────────────────────┐
       │             Transformer Fusion Blocks           │
       └──────────────────────┬──────────────────────────┘
                              │
               ┌──────────────┴──────────────┐
               ▼                             ▼
       Binary Classification         Forensic Anomaly
       (Authentic vs. AI)            Heatmap & Band Report
```

---

## 📊 Benchmark Results

Evaluated on the frozen, canonical test split of **18,000 unseen images** (SHA-256: `e8db1a7249ef22fbf9fcb6e6659e9ecc5ebbc5d81430a268fce26546f23bdde2` hosted on [MohsinElis/mfft-master-manifest](https://huggingface.co/MohsinElis/mfft-master-manifest)):

### 1. Proposed MFFT Family vs. Ten Baseline Models

| Model Architecture | Parameters | Input Res | Test Acc (%) | AUC-ROC (%) | Precision (%) | Recall (%) | F1-Score (%) | Specificity (%) | Loss |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **MFFT-Tiny (Proposed)** | **372,450** | 384x384 | **98.59** | **99.88** | 99.23 | 97.94 | **98.59** | 99.24 | 0.2232 |
| **MFFT-Base (Proposed)** | **1,622,690** | 384x384 | **98.37** | **99.87** | 98.59 | 98.13 | **98.36** | 98.60 | 0.2258 |
| **MFFT-Large (Proposed)** | **6,301,250** | 384x384 | **98.11** | **99.68** | 98.78 | 97.41 | **98.09** | 98.80 | 0.2362 |
| SimpleCNN | 422,530 | 384x384 | 98.51 | 99.82 | 99.13 | 97.87 | 98.50 | 99.14 | 0.2342 |
| FreqDetect | 14,658 | 384x384 | 50.00 | 49.95 | 50.00 | 100.00 | 66.67 | 0.00 | 0.6932 |
| EfficientNet-B0 | 4,010,110 | 384x384 | 99.51 | 99.95 | 99.71 | 99.31 | 99.51 | 99.71 | 0.2070 |
| ResNet-18 | 11,177,538 | 384x384 | 99.32 | 99.97 | 99.48 | 99.16 | 99.32 | 99.48 | 0.2110 |
| ResNet-50 | 23,512,130 | 384x384 | 99.38 | 99.96 | 99.20 | 99.56 | 99.38 | 99.20 | 0.2142 |
| Swin-T | 27,520,892 | 384x384 | 99.31 | 99.96 | 99.37 | 99.24 | 99.31 | 99.37 | 0.2108 |
| LightViT | 5,084,418 | 384x384 | 95.42 | 99.26 | 98.47 | 92.27 | 95.27 | 98.57 | 0.2741 |
| ViT-B/16 | 86,092,034 | 384x384 | 93.13 | 98.37 | 95.30 | 90.73 | 92.96 | 95.52 | 0.3103 |
| DeiT-Small | 21,813,892 | 384x384 | 88.12 | 95.30 | 92.60 | 82.87 | 87.46 | 93.38 | 0.3831 |
| CLIP ViT-B/32 | 87,924,226 | 384x384 | 85.76 | 92.66 | 88.53 | 82.17 | 85.23 | 89.36 | 0.4242 |

### 2. Multi-Seed Replication (MFFT-Base Stability)

| Run ID | Seed | Test Acc (%) | AUC-ROC (%) | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `mfft_base` (Canonical) | 42 | 98.37 | 99.87 | 98.59 | 98.13 | 98.36 |
| `mfft_base_seed42` | 42 | 98.66 | 99.91 | 99.03 | 98.28 | 98.65 |
| `mfft_base_seed999` | 999 | 98.67 | 99.89 | 99.04 | 98.28 | 98.66 |
| `mfft_base_seed1234` | 1234 | 98.68 | 99.90 | 99.08 | 98.28 | 98.67 |
| `mfft_base_seed2026` | 2026 | 98.55 | 99.89 | 98.81 | 98.28 | 98.54 |
| **Five-Seed Mean ± Std** | — | **98.58 ± 0.13** | **99.89 ± 0.02** | **98.91 ± 0.21** | **98.25 ± 0.07** | **98.58 ± 0.13** |

### 3. Cross-Generator Generalization (LOGO Protocol)

| Held-Out Family | Generator Sources | LOGO Accuracy (%) | LOGO AUC-ROC (%) |
| :--- | :--- | :---: | :---: |
| **Diffusion (SDXL)** | Stable Diffusion XL (SDXL) | **97.80** | **99.78** |
| **Deepfakes** | FaceForensics++, DFDC, Celeb-DF v2 | **98.09** | **99.82** |
| **GANs** | StyleGAN, ProGAN, BigGAN | **95.10** | **99.12** |

### 4. Calibration & Robustness

- **Expected Calibration Error (ECE)**: `0.0486` (reliable probabilistic predictions).
- **Brier Score**: `0.0955`.
- **JPEG Compression Stability**: Preserves `97.0%–98.0%` accuracy across JPEG qualities `Q40` to `Q95`.

---

## 📦 Model Zoo & Pretrained Weights

Verified 20-epoch production weights are publicly hosted on Hugging Face Hub:

| Model Variant | Parameters | File Size | Hugging Face Checkpoint Link | Local Destination |
| :--- | :---: | :---: | :--- | :--- |
| **MFFT-Tiny** | 372K | ~4.4 MB | [`MohsinElis/mfft-checkpoints/runs/mfft_tiny/checkpoints/best.pt`](https://huggingface.co/MohsinElis/mfft-checkpoints/resolve/main/runs/mfft_tiny/checkpoints/best.pt) | `model/checkpoints/production/tiny.pt` |
| **MFFT-Base** | 1.62M | ~18.7 MB | [`MohsinElis/mfft-checkpoints/runs/mfft_base/checkpoints/best.pt`](https://huggingface.co/MohsinElis/mfft-checkpoints/resolve/main/runs/mfft_base/checkpoints/best.pt) | `model/checkpoints/production/base.pt` |
| **MFFT-Large** | 6.30M | ~72.2 MB | [`MohsinElis/mfft-checkpoints/runs/mfft_large/checkpoints/best.pt`](https://huggingface.co/MohsinElis/mfft-checkpoints/resolve/main/runs/mfft_large/checkpoints/best.pt) | `model/checkpoints/production/large.pt` |

To download all production checkpoints with a single command (no authentication required):

```bash
python scripts/download_production_models.py
```

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/MIHMahmudEli/mfft.git
cd mfft

# Install dependencies
pip install -r model/requirements.txt
```

### 2. Python Inference

```python
import torch
from PIL import Image
import torchvision.transforms as T
from model.src.model import build_mfft

# 1. Build model architecture (tiny, base, or large)
model = build_mfft(variant="large", img_size=384, num_classes=2)

# 2. Load verified production weights
checkpoint = torch.load("model/checkpoints/production/large.pt", map_location="cpu", weights_only=False)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# 3. Preprocessing (aspect-ratio preserving resize + center crop)
def preprocess(img: Image.Image, img_size: int = 384):
    w, h = img.size
    target = img_size + 16
    if w < h:
        new_w, new_h = target, max(target, int(h * target / w))
    else:
        new_h, new_w = target, max(target, int(w * target / h))
    img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
    left = max(0, (new_w - img_size) // 2)
    top = max(0, (new_h - img_size) // 2)
    img = img.crop((left, top, left + img_size, top + img_size))
    
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(img).unsqueeze(0)

image = Image.open("sample.jpg").convert("RGB")
tensor = preprocess(image)

with torch.no_grad():
    outputs = model(tensor, return_features=True)
    logits = outputs["logits"] if isinstance(outputs, dict) else outputs
    probs = torch.softmax(logits, dim=-1)

ai_probability = probs[0, 1].item()
print(f"Prediction: {'AI-Generated' if ai_probability > 0.5 else 'Authentic'}")
print(f"Confidence: {ai_probability * 100:.2f}% AI Probability")
```

---

## 🌐 Live Demonstrations & Services

- **Web Application**: [https://imageverify-ai.vercel.app/](https://imageverify-ai.vercel.app/) — Interactive OLED interface with real-time laser scan animation and multi-model switching.
- **REST Inference API**: [https://mfft-detection-api.onrender.com/](https://mfft-detection-api.onrender.com/) — Production FastAPI microservice supporting single and batch image classification.
- **Hugging Face Checkpoints**: [https://huggingface.co/MohsinElis/mfft-checkpoints](https://huggingface.co/MohsinElis/mfft-checkpoints) — Machine-readable GPU training and evaluation checkpoints.
- **Master Manifest**: [https://huggingface.co/MohsinElis/mfft-master-manifest](https://huggingface.co/MohsinElis/mfft-master-manifest) — Frozen 120,000-image dataset split manifest.

---

## 🛠️ Repository Structure

```
.
├── model/
│   ├── src/
│   │   ├── model.py                  # Core MFFT architecture & Frequency-Guided Attention (FGA)
│   │   ├── baselines.py              # Implementations of 10 baseline models
│   │   ├── dataset.py                # Dataset loader with FFT pre-filtering and caching
│   │   ├── train.py                  # Distributed AMP training engine
│   │   ├── evaluate.py               # Evaluation metrics (AUC, F1, specificity, precision)
│   │   ├── calibration.py            # Temperature scaling, ECE, and reliability metrics
│   │   ├── robustness.py             # Perturbation analysis (JPEG, blur, downscale)
│   │   ├── stats.py                  # Bootstrap confidence intervals and McNemar tests
│   │   ├── logo_eval.py              # Leave-One-Generator-Out cross-generator evaluation
│   │   └── visualize.py              # Publication figure generators
│   ├── checkpoints/
│   │   └── production/               # Pretrained weights (tiny.pt, base.pt, large.pt)
│   └── requirements.txt
├── api/                              # High-throughput FastAPI inference server
│   ├── main.py                       # REST endpoints & CORS configuration
│   ├── model_server.py               # Model loading & aspect-preserving inference
│   └── schemas.py                    # Pydantic request/response schemas
├── web/                              # Modern Next.js 14 web application
│   ├── app/                          # App router and UI pages
│   ├── components/                   # React components (laser scan, result cards)
│   └── lib/                          # API client and TypeScript definitions
├── scripts/
│   ├── download_production_models.py # One-command checkpoint downloader
│   ├── test_deployment_e2e.py        # End-to-end API verification suite
│   └── deploy_hf_space.py            # Hugging Face Space packager
├── kaggle_mfft_pipeline.ipynb        # Autonomous training and evaluation pipeline
├── compose.yaml                      # Multi-container local/cloud deployment
├── Dockerfile.render                 # Production container for cloud inference
├── render.yaml                       # Infrastructure-as-code specification
├── CITATION.cff                      # Zenodo DOI citation metadata
├── .zenodo.json                      # Zenodo archival configuration
├── DEPLOYMENT.md                     # Deployment and operations manual
└── LICENSE                           # MIT License
```

---

## 📜 Citation

If you use this codebase, models, or research artifacts in your research, please cite:

```bibtex
@software{hossain2026mfft,
  author       = {Mohsin Ibna Hossain and MD. Muttakin and Mohammad Mahmudul Hasan},
  title        = {MFFT: Multi-Frequency Fusion Transformer for Explainable AI-Generated Image Detection},
  year         = {2026},
  publisher    = {Zenodo},
  version      = {v1.0.0},
  doi          = {10.5281/zenodo.23339178},
  url          = {https://github.com/MIHMahmudEli/mfft}
}
```

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
