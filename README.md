# MFFT: Multi-Frequency Fusion Transformer for Explainable AI-Generated Image Detection

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24225137.svg)](https://doi.org/10.5281/zenodo.24225137)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg?style=flat&logo=pytorch)](https://pytorch.org/)
[![Hugging Face Models](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Checkpoints-yellow.svg)](https://huggingface.co/MohsinElis/mfft-checkpoints)
[![Hugging Face Manifest](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Master%20Manifest-blue.svg)](https://huggingface.co/MohsinElis/mfft-master-manifest)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Live Demo](https://img.shields.io/badge/Demo-ImageVerify%20AI-000000.svg?logo=vercel)](https://imageverify-ai.vercel.app/)
[![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97%20Space-MFFT%20API-blue.svg)](https://huggingface.co/spaces/MohsinElis/mfft-api)
[![API Status](https://img.shields.io/badge/API-Render%20Online-46E3B7.svg?logo=render)](https://mfft-detection-api.onrender.com/)

Official open-source PyTorch implementation, verified pretrained models, evaluation pipelines, and reproducibility artifacts for the **Multi-Frequency Fusion Transformer (MFFT)**, an explainable vision architecture designed to detect AI-generated imagery and deepfakes across radial Fourier frequency sub-bands using depthwise separable convolutions and Frequency-Guided Attention (FGA).

> **Associated Journal Publication**:  
> *"MFFT: Multi-Frequency Fusion Transformer for AI-Generated Image Detection"*  
> **Mohsin Ibna Hossain, MD. Muttakin, and Mohammad Mahmudul Hasan**  
> *Elsevier: Machine Learning with Applications (MLWA)*

---

## 🌟 Key Highlights

- **Frequency-Domain Forensic Representation**: Decomposes 2D Fast Fourier Transform (FFT) spectra into discrete non-overlapping radial annular sub-bands (Low, Mid, High) satisfying an exact reconstruction property ($\sum_{k=1}^K B_k = X$), isolating transposed convolution upsampling grids and spectral roll-off anomalies that spatial-only models miss.
- **Frequency-Guided Attention (FGA)**: An inductive physical gating mechanism that dynamically modulates inter-band cross-attention according to the physical logarithmic Fourier energy distribution of the input image, regularizing the transformer against semantic shortcut learning.
- **Pareto-Optimal Parameter Efficiency**:
  - **MFFT-Tiny** (372K params, 1.49 MB FP32): **98.59% accuracy**, **99.88% AUC-ROC** (~3.2 ms GPU latency). Approaches within 0.72%–0.92% of heavy supervised CNNs while using 10.8× to 63.5× fewer parameters, and outperforms large Vision Transformers (ViT-B/16 at 93.13%, CLIP ViT-B/32 at 85.76%) by **+5.46% to +12.83%**.
  - **MFFT-Base** (1.62M params, 6.49 MB FP32): **98.37% accuracy**, **99.87% AUC-ROC** (5-seed replication average: **98.58% ± 0.13%** accuracy, **99.89% ± 0.02%** AUC-ROC).
  - **MFFT-Large** (6.30M params, 25.21 MB FP32): **98.11% accuracy**, **99.68% AUC-ROC** (extended 4-band spectral configuration).
- **Frozen Master Benchmark**: Evaluated on a canonical, balanced dataset of **120,000 images** (84K train / 18K val / 18K test) anchored by SHA-256 hash `e8db1a7249ef22fbf9fcb6e6659e9ecc5ebbc5d81430a268fce26546f23bdde2` on [MohsinElis/mfft-master-manifest](https://huggingface.co/MohsinElis/mfft-master-manifest).
- **Cross-Generator Generalization**: Evaluated under Leave-One-Generator-Out (LOGO) cross-generator validation across latent diffusion (SDXL: 97.80%), GANs (95.10%), and facial deepfakes (98.09%).
- **Calibrated & Robust**: Low Expected Calibration Error ($\text{ECE} = 0.0486$, Brier score = 0.0145) and stable detection under lossy JPEG compression (97.0%–98.0% accuracy from $Q=95$ down to $Q=40$).

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
       │     Cross-Attention Band Fusion (CAF)           │
       │   - Learns inter-band Query-Key dependencies   │
       │   - Modulated by Parseval Fourier Energy (FGA)  │
       └──────────────────────┬──────────────────────────┘
                              │
                              ▼
       ┌─────────────────────────────────────────────────┐
       │              MLP Classification Head            │
       └──────────────────────┬──────────────────────────┘
                              │
               ┌──────────────┴──────────────┐
               ▼                             ▼
       Binary Classification         Forensic Spectral
       (Authentic vs. AI)            Attribution Heatmap
```

---

## 📊 Benchmark Results

All metrics below are computed on the canonical held-out test set of **18,000 unseen images** (50% authentic / 50% AI across 12 diverse generator shards) and verified directly from evaluation artifacts hosted on [MohsinElis/mfft-checkpoints](https://huggingface.co/MohsinElis/mfft-checkpoints).

### 1. Proposed MFFT Family vs. Ten Baseline Models

> **Note on Accuracy vs. Parameter Efficiency**: Deeply parameterized supervised models (EfficientNet-B0 at 99.51%, ResNet-50 at 99.38%, Swin-T at 99.31%) attain marginally higher raw test accuracy (+0.72% to +0.92%). However, MFFT-Tiny approaches within 0.72%–0.92% while cutting model parameters by **10.8× to 63.5×** (requiring only 1.49 MB storage) and substantially outperforms standard Vision Transformers (ViT-B/16 by +5.46%, CLIP ViT-B/32 by +12.83%).

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

Evaluated across 5 independent random training seeds on the frozen 18,000-image test set:

| Run Identifier | Seed | Test Acc (%) | AUC-ROC (%) | Precision (%) | Recall (%) | F1-Score (%) | Loss |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Run 1 (Canonical / Seed 0) | 0 | 98.37 | 99.87 | 98.59 | 98.13 | 98.36 | 0.2258 |
| Run 2 (Seed 42) | 42 | 98.53 | 99.89 | 99.07 | 97.98 | 98.52 | 0.2241 |
| Run 3 (Seed 999) | 999 | 98.75 | 99.88 | 99.34 | 98.16 | 98.74 | 0.2214 |
| Run 4 (Seed 1234) | 1234 | 98.65 | 99.92 | 98.56 | 98.74 | 98.65 | 0.2209 |
| Run 5 (Seed 2026) | 2026 | 98.62 | 99.90 | 98.71 | 98.53 | 98.62 | 0.2216 |
| **Five-Seed Mean ± Std** | — | **98.58 ± 0.13** | **99.89 ± 0.02** | **98.85 ± 0.33** | **98.31 ± 0.31** | **98.58 ± 0.13** | **0.2228 ± 0.002** |

### 3. Cross-Generator Generalization (LOGO Protocol)

In Leave-One-Generator-Out (LOGO) evaluations, the designated generator family is completely excluded from training and paired with strictly disjoint, unseen authentic real photographs:

| Held-Out Family | Generator Sources | Test Acc (%) | AUC-ROC (%) | Precision (%) | Recall (%) | F1-Score (%) | Specificity (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Latent Diffusion (SDXL)** | Stable Diffusion XL | **97.80** | **97.33** | 83.31 | 81.52 | 82.40 | 98.90 |
| **GAN Architectures** | StyleGAN, ProGAN | **95.10** | **96.46** | 97.69 | 84.84 | 90.81 | 99.20 |
| **Facial Deepfakes** | DFDC, FaceForensics++ | **98.09** | **99.69** | 90.90 | 98.05 | 94.34 | 98.10 |

### 4. Calibration & JPEG Robustness

- **Expected Calibration Error (ECE)**: `0.0486` (4.86% error across 10 confidence bins).
- **Brier Score**: `0.0145` (mean squared probability error).
- **JPEG Robustness**: Accuracy remains stable within a 1.0% envelope across compression quality factors:
  - $Q = 95$: **97.5%**
  - $Q = 80$: **97.5%**
  - $Q = 60$: **98.0%**
  - $Q = 40$: **97.0%**

---

## 📦 Model Zoo & Pretrained Weights

Verified 20-epoch production weights are publicly hosted on Hugging Face Hub:

| Model Variant | Trainable Parameters | Model Size (FP32) | Checkpoint File Size | Hugging Face Hub Checkpoint Link | Local Destination |
| :--- | :---: | :---: | :---: | :--- | :--- |
| **MFFT-Tiny** | 372,450 | 1.49 MB | ~4.4 MB | [`runs/mfft_tiny/checkpoints/best.pt`](https://huggingface.co/MohsinElis/mfft-checkpoints/resolve/main/runs/mfft_tiny/checkpoints/best.pt) | `model/checkpoints/production/tiny.pt` |
| **MFFT-Base** | 1,622,690 | 6.49 MB | ~18.7 MB | [`runs/mfft_base/checkpoints/best.pt`](https://huggingface.co/MohsinElis/mfft-checkpoints/resolve/main/runs/mfft_base/checkpoints/best.pt) | `model/checkpoints/production/base.pt` |
| **MFFT-Large** | 6,301,250 | 25.21 MB | ~72.2 MB | [`runs/mfft_large/checkpoints/best.pt`](https://huggingface.co/MohsinElis/mfft-checkpoints/resolve/main/runs/mfft_large/checkpoints/best.pt) | `model/checkpoints/production/large.pt` |

To download all production checkpoints with a single command (no authentication required):

```bash
python scripts/download_production_models.py
```

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the official public repository
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
model = build_mfft(variant="base")

# 2. Load verified production weights
checkpoint = torch.load("model/checkpoints/production/base.pt", map_location="cpu", weights_only=False)
raw_state = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint

# Remap training cluster state keys to local model definition
state_dict = {}
for k, v in raw_state.items():
    nk = k.replace(".stages.", ".blocks.").replace("fga.gate.", "freq_guided_attn.freq_gate.")
    if "num_batches_tracked" not in nk:
        state_dict[nk] = v

model.load_state_dict(state_dict, strict=False)
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
    logits = model(tensor)
    probs = torch.softmax(logits, dim=-1)

ai_probability = probs[0, 1].item()
print(f"Prediction: {'AI-Generated' if ai_probability > 0.5 else 'Authentic'}")
print(f"Confidence: {ai_probability * 100:.2f}% AI Probability")
```

---

## 🌐 Live Demonstrations & Services

- **Web Application**: [https://imageverify-ai.vercel.app/](https://imageverify-ai.vercel.app/) — Interactive interface with real-time laser scan animation and multi-model switching.
- **REST Inference API**: [https://mfft-detection-api.onrender.com/](https://mfft-detection-api.onrender.com/) — Production FastAPI microservice supporting single and batch image classification.
- **Hugging Face Checkpoints**: [https://huggingface.co/MohsinElis/mfft-checkpoints](https://huggingface.co/MohsinElis/mfft-checkpoints) — Machine-readable GPU training and evaluation checkpoints.
- **Master Manifest**: [https://huggingface.co/MohsinElis/mfft-master-manifest](https://huggingface.co/MohsinElis/mfft-master-manifest) — Frozen 120,000-image dataset split manifest.
- **Canonical Results Registry**: [`results/canonical_results_registry.json`](results/canonical_results_registry.json) — Full machine-readable experimental records across all 32 experimental conditions.

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
│   └── requirements.txt
├── api/                              # High-throughput FastAPI inference server
│   ├── main.py                       # REST endpoints & CORS configuration
│   ├── model_server.py               # Model loading & aspect-preserving inference
│   ├── schemas.py                    # Pydantic request/response schemas
│   ├── requirements.txt              # API dependencies
│   └── Dockerfile                    # Container definition for FastAPI
├── web/                              # Modern Next.js 14 web application
│   ├── app/                          # App router and UI pages
│   ├── components/                   # React components (laser scan, result cards)
│   └── lib/                          # API client and TypeScript definitions
├── scripts/
│   ├── download_production_models.py # One-command checkpoint downloader from Hugging Face
│   ├── test_deployment_e2e.py        # End-to-end API verification suite
│   ├── test_results_registry_validation.py # Registry validation test suite
│   └── deploy_hf_space.py            # Hugging Face Space packager
├── results/
│   └── canonical_results_registry.json # Comprehensive machine-readable experimental registry
├── pipelines/
│   ├── kaggle_mfft_pipeline.ipynb    # Autonomous training and evaluation pipeline
│   └── kaggle_mfft_pipeline.py       # Standalone pipeline export & verification script
├── compose.yaml                      # Multi-container orchestration (api + web + nginx)
├── nginx.conf                        # Reverse proxy configuration
├── Dockerfile.api                    # Standalone API container
├── Dockerfile.web                    # Standalone Web container
├── Dockerfile.render                 # Production container for cloud inference
├── render.yaml                       # Infrastructure-as-code specification
├── CITATION.cff                      # Zenodo DOI citation metadata
├── .zenodo.json                      # Zenodo archival configuration
└── LICENSE                           # MIT License
```

---

## 📜 Citation

If you use this codebase, models, or research artifacts in your research, please cite:

```bibtex
@article{hossain2026mfft,
  title={MFFT: Multi-Frequency Fusion Transformer for AI-Generated Image Detection},
  author={Hossain, Mohsin Ibna and Muttakin, MD. and Hasan, Mohammad Mahmudul},
  journal={Machine Learning with Applications},
  year={2026},
  publisher={Elsevier},
  note={Official Checkpoints: \url{https://huggingface.co/MohsinElis/mfft-checkpoints}}
}

@software{hossain2026mfft_code,
  author       = {Hossain, Mohsin Ibna and Muttakin, MD. and Hasan, Mohammad Mahmudul},
  title        = {MFFT: Multi-Frequency Fusion Transformer for Explainable AI-Generated Image Detection},
  year         = {2026},
  publisher    = {Zenodo},
  version      = {v1.0.0},
  doi          = {10.5281/zenodo.24225137},
  url          = {https://github.com/MIHMahmudEli/mfft}
}
```

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
