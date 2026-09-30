# MFFT: Multi-Frequency Fusion Transformer for AI-Generated Image Detection

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg?style=flat&logo=pytorch)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Paper](https://img.shields.io/badge/Paper-In%20Submission-brightgreen.svg)]()
[![Live Demo](https://img.shields.io/badge/Demo-ImageVerify%20AI-000000.svg?logo=vercel)](https://imageverify-ai.vercel.app/)
[![API Status](https://img.shields.io/badge/API-Render%20Online-46E3B7.svg?logo=render)](https://mfft-detection-api.onrender.com/)

Official PyTorch implementation of **Multi-Frequency Fusion Transformer (MFFT)**, an explainable deep architecture designed to detect AI-generated and deepfake images by analyzing localized artifacts across radial Fourier frequency bands.

---

## 🌟 Key Highlights

- **Frequency-Domain Forensic Representation**: Extracts radial Fourier frequency bands (Low, Mid, High) to uncover subtle generative grid discrepancies and spectral phase disruptions invisible in RGB pixel space.
- **Frequency-Guided Cross-Attention (FGA)**: Cross-attends spatial features guided by frequency magnitude weights, directing the transformer's receptive field to artifact-dense frequency bands.
- **Compact & High-Accuracy**:
  - **MFFT-Tiny** (372K params): **98.59% accuracy**, **99.88% AUC-ROC** (<15ms latency).
  - **MFFT-Base** (1.62M params): **98.58% accuracy**, **99.89% AUC-ROC** (balanced production default).
  - **MFFT-Large** (6.30M params): **98.11% accuracy**, **99.68% AUC-ROC** (maximum forensic capacity).
- **Outperforms 10 Competitive Baselines**: Evaluated against vision transformers, vision-language models, CNNs, and frequency baselines (CLIP, Swin-T, ViT-B/16, DeiT-S, EfficientNet-B0, ResNet-50, FreqDetect).
- **Inherent Explainability**: Generates high-frequency spatial anomaly heatmaps and per-band contribution vectors with every inference.
- **Robust Against Degradation**: Evaluated on JPEG compression (Q30–Q90), Gaussian blurring, and downscale-upscale perturbations.

---

## 🔬 Architecture Overview

```
Input Image (384x384)
         │
         ▼
┌────────────────────────────────────────────────────────┐
│     Fourier Radial Decomposition (Low / Mid / High)     │
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
       │   - Learns spectral phase & magnitude weights   │
       │   - Discovers spatial artifact correspondences  │
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

Evaluated on the canonical frozen test benchmark of **18,000 held-out images** (balanced authentic vs. AI generators including Stable Diffusion, Midjourney, DALL-E 3, SDXL, BigGAN, and CelebDF):

### 1. Proposed MFFT Variants vs. Baselines

| Model | Parameters | Test Acc (%) | AUC-ROC (%) | Recall (%) | Specificity (%) | F1-Score (%) |
|---|---|---|---|---|---|---|
| **MFFT-Tiny (Ours)** | **372K** | **98.59** | **99.88** | 98.42 | 98.76 | **98.59** |
| **MFFT-Base (Ours)** | **1.62M** | **98.58** | **99.89** | 98.51 | 98.65 | **98.58** |
| **MFFT-Large (Ours)** | **6.30M** | **98.11** | **99.68** | 97.41 | 98.80 | **98.09** |
| CLIP (ViT-B/32) | 87.9M | 95.83 | 99.12 | 95.20 | 96.46 | 95.81 |
| Swin-T | 27.5M | 95.42 | 98.96 | 94.88 | 95.96 | 95.40 |
| ViT-B/16 | 86.1M | 95.10 | 98.82 | 94.60 | 95.60 | 95.08 |
| DeiT-Small | 22.1M | 94.76 | 98.54 | 93.90 | 95.62 | 94.73 |
| ResNet-50 | 23.5M | 94.65 | 98.45 | 94.12 | 95.18 | 94.63 |
| EfficientNet-B0 | 4.0M | 94.20 | 98.12 | 93.55 | 94.85 | 94.18 |
| ResNet-18 | 11.2M | 93.85 | 97.90 | 93.10 | 94.60 | 93.82 |
| LightViT | 5.1M | 92.40 | 96.85 | 91.80 | 93.00 | 92.38 |
| SimpleCNN | 423K | 89.15 | 94.50 | 88.20 | 90.10 | 89.12 |
| FreqDetect | 14.7K | 82.40 | 89.20 | 81.10 | 83.70 | 82.35 |

---

## 🚀 Quick Start

### 1. Installation

```bash
git clone https://github.com/MIHMahmudEli/mfft.git
cd mfft

# Install dependencies
pip install -r model/requirements.txt
```

### 2. Pretrained Model Loading

```python
import torch
from model.src.model import build_mfft

# Instantiate MFFT architecture (tiny, base, or large)
model = build_mfft(variant="base", img_size=384, num_classes=2)

# Load verified 20-epoch production weights
checkpoint = torch.load("model/checkpoints/production/base.pt", map_location="cpu", weights_only=False)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("MFFT model loaded successfully!")
```

### 3. Running Inference on an Image

```python
from PIL import Image
import torchvision.transforms as T

# Standard preprocessing
transform = T.Compose([
    T.Resize((384, 384)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

image = Image.open("sample.jpg").convert("RGB")
tensor = transform(image).unsqueeze(0)

with torch.no_grad():
    outputs = model(tensor, return_features=True)
    logits = outputs["logits"] if isinstance(outputs, dict) else outputs
    probs = torch.softmax(logits, dim=-1)
    
ai_probability = probs[0, 1].item()
print(f"AI-Generated Probability: {ai_probability * 100:.2f}%")
```

---

## 🛠️ Repository Structure

```
.
├── model/
│   ├── src/
│   │   ├── model.py              # Core MFFT architecture & Frequency-Guided Attention
│   │   ├── baselines.py          # 10 competitive baseline architectures
│   │   ├── dataset.py            # Dataset loading, Fourier pre-filtering, and augmentations
│   │   ├── train.py              # Multi-GPU AMP training engine
│   │   ├── evaluate.py           # Evaluation metrics (AUC-ROC, F1, specificity)
│   │   ├── robustness.py         # JPEG, blur, and resolution perturbation curves
│   │   ├── calibration.py        # Temperature scaling, ECE, and reliability diagrams
│   │   ├── stats.py              # Bootstrap confidence intervals & McNemar tests
│   │   └── visualize.py          # Publication-ready figures and error distributions
│   ├── checkpoints/
│   │   └── production/           # Verified 20-epoch weights (tiny, base, large)
│   └── requirements.txt
├── api/                          # High-throughput FastAPI inference server
│   ├── main.py
│   ├── model_server.py
│   └── schemas.py
├── web/                          # Modern Next.js 14 web application
│   ├── app/
│   ├── components/
│   └── public/samples/
├── paper/
│   ├── mfft_manuscript.tex       # Full IEEE-formatted manuscript source
│   ├── result/tables/            # Benchmark CSV/LaTeX tables
│   └── result/figures/           # 300 DPI publication figures
├── compose.yaml                  # Multi-container production deployment
├── Dockerfile.render             # Ultra-lightweight CPU container for cloud deployment
├── DEPLOYMENT.md                 # Production deployment & operations guide
└── LICENSE                       # MIT License
```

---

## 🌐 Live Demonstrations

- **Web Application**: [https://imageverify-ai.vercel.app/](https://imageverify-ai.vercel.app/) (OLED Black UI with real-time laser scan animation)
- **Inference API**: [https://mfft-detection-api.onrender.com/](https://mfft-detection-api.onrender.com/) (FastAPI REST service)
- **Hugging Face Space**: [https://huggingface.co/spaces/MohsinElis/mfft-api](https://huggingface.co/spaces/MohsinElis/mfft-api)

---

## 📜 Citation

If you find this work or codebase useful for your research, please cite:

```bibtex
@article{hasan2026mfft,
  title={MFFT: Multi-Frequency Fusion Transformer for Explainable AI-Generated Image Detection},
  author={Hasan, Mohammad Mahmudul},
  journal={arXiv preprint},
  year={2026}
}
```

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
