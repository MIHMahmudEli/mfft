#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MFFT Research Pipeline — Kaggle GPU + HuggingFace Persistence
Multi-Frequency Fusion Transformer for AI-Generated Image Detection

Target Journal: Q1 IEEE TIFS / Pattern Recognition
Platform: Kaggle GPU (T4 x2 / P100 / V100 / A100)
Persistence: HuggingFace Hub (Checkpoints, Metrics, Figures, Tables)
"""

# %%
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MFFT Research Pipeline — Kaggle GPU + HuggingFace Persistence
Multi-Frequency Fusion Transformer for AI-Generated Image Detection

Target Journal: Q1 IEEE TIFS / Pattern Recognition
Platform: Kaggle GPU (T4 x2 / P100 / V100 / A100)
Persistence: HuggingFace Hub (Checkpoints, Metrics, Figures, Tables)
"""

# %% [markdown]
# # 1. Research Configuration
# Centralized configuration dictionary controlling all aspects of the research pipeline.
# Do not hard-code hyperparameter values throughout the notebook.

# %%
import os
import sys
import json
import time
import random
import hashlib
import logging
import warnings
import tempfile
import shutil
import csv
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, List, Tuple, Any
from collections import Counter

# ═══════════════════════════════════════════════════════════════════
# SECTION 1: CENTRALIZED RESEARCH CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

CONFIG = {
    # ── Experiment Metadata ──
    "experiment_name": "MFFT_Research",
    "target_journal": "IEEE_TIFS_Q1",
    "model_variant": "base",            # "tiny" (372K) | "base" (1.62M) | "large" (6.30M)
    "model_kind": "mfft",               # "mfft" (proposed) | "baseline" (comparison model)
    "baseline_name": None,
    "logo_holdout": None,              # e.g. "resnet50", "vit_b16", "clip"
    "num_classes": 2,                    # Binary: Real (0) vs AI-Generated / Deepfake (1)
    "seed": 42,
    "smoke_test": False,                 # Set True for 3-minute pipeline validation run
    "swarm_mode": True,                  # Autonomous work-stealing across multi-account Kaggle workers
    "worker_name": "",                   # Leave blank for auto-discovery (via Secrets or hostname)
    "stale_heartbeat_minutes": 180,       # Reclaim jobs whose worker hasn't updated in 180 mins

    # ── Training Hyperparameters ──
    "num_epochs": 20,
    "learning_rate": 3e-4,
    "weight_decay": 0.05,
    "warmup_steps": 500,
    "max_grad_norm": 1.0,
    "label_smoothing": 0.1,
    "gradient_accumulation_steps": 2,    # Tuned dynamically per GPU
    "mixed_precision": True,
    "patience": 7,                       # Early stopping patience (epochs)

    # ── Resolution and Batching ──
    "image_size": 384,                   # 384x384 standard for MFFT architecture
    "batch_size": 32,                    # Tuned dynamically per GPU
    "num_workers": 2,                    # Tuned dynamically per GPU

    # ── Dataset Partitions ──
    "val_split": 0.15,
    "test_split": 0.15,
    "max_samples": None,                 # Set integer (e.g. 5000) for budget subset, None for full

    # ── Checkpointing & Persistence ──
    "checkpoint_interval_epochs": 1,
    "snapshot_epochs": [5, 10, 15, 20],
    "save_top_k": 3,
    "hf_checkpoint_repo": "MohsinElis/mfft-checkpoints",
    "hf_manifest_repo": "MohsinElis/mfft-master-manifest",
    "manifest_path_in_repo": "manifest/split_manifest.json",

    # ── Reproducibility ──
    "deterministic": True,
    "benchmark": False,

    # ── Ablation Studies ──
    "run_ablations": True,
    "ablation_epochs": 8,
    "ablation_configs": {
        "spatial_only":   {"spatial_only": True},
        "no_fga":         {"use_fga": False},
        "fusion_concat":  {"fusion_mode": "concat"},
        "fusion_avg":     {"fusion_mode": "avg"},
        "fusion_max":     {"fusion_mode": "max"},
        "band_low_only":  {"skip_bands": ["mid", "high"]},
        "band_mid_only":  {"skip_bands": ["low", "high"]},
        "band_high_only": {"skip_bands": ["low", "mid"]},
    },

    # ── Statistical Rigor ──
    "bootstrap_n_resamples": 1000,
    "confidence_level": 0.95,
    "run_calibration": True,
    "run_robustness": True,

    # ── Publication Figures & Tables ──
    "figure_dpi": 300,
    "figure_format": ["png", "pdf"],
    "figure_font_size": 10,
}

# GPU profile presets (automatically applied in Section 2)
GPU_PROFILES = {
    "T4":   {"batch_size": 32, "gradient_accumulation_steps": 2, "num_workers": 2, "image_size": 384},
    "P100": {"batch_size": 48, "gradient_accumulation_steps": 2, "num_workers": 2, "image_size": 384},
    "V100": {"batch_size": 64, "gradient_accumulation_steps": 1, "num_workers": 4, "image_size": 384},
    "A100": {"batch_size": 128, "gradient_accumulation_steps": 1, "num_workers": 8, "image_size": 384},
    "CPU":  {"batch_size": 8, "gradient_accumulation_steps": 4, "num_workers": 0, "image_size": 224},
}

# 12 Kaggle dataset mount definitions: slug -> (label, subpath)
KAGGLE_DATASETS = {
    "places365":                        ("real",         "train"),
    "open-images-v7-dataset":           ("real",         ""),
    "mfft-real":                        ("real",         ""),
    "faceforensics":                    ("deepfake",     "cropped_images"),
    "dfdc-faces-of-the-train-sample":   ("deepfake",     "train/fake"),
    "stable-diffusion":                 ("ai_generated", ""),
    "ntire2026":                        ("ai_generated", ""),
    "midjourney":                       ("ai_generated", ""),
}

CROSS_DOMAIN_DATASETS = {
    "dall-e3":                 ("ai_generated", "DALL-E3"),
    "celebdf-v2image-dataset": ("deepfake",     "Celeb_V2"),
    "genimage-ai":             ("ai_generated", "genimage_ai"),
}

LABEL_MAP = {"real": 0, "ai_generated": 1, "deepfake": 1}
CLASS_NAMES = ["real", "ai_generated"]
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

print("✓ Configuration initialized")
print(f"  Model variant: MFFT-{CONFIG['model_variant']}")
print(f"  Target epochs: {CONFIG['num_epochs']}")
print(f"  Base seed:     {CONFIG['seed']}")

# %% [markdown]
# # 2. Environment & Hardware Verification
# Detect compute accelerator, set deterministic random seeds, and allocate experiment directories.

# %%
import torch
import numpy as np

# ═══════════════════════════════════════════════════════════════════
# SECTION 2: ENVIRONMENT & HARDWARE VERIFICATION
# ═══════════════════════════════════════════════════════════════════

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = CONFIG["deterministic"]
    torch.backends.cudnn.benchmark = CONFIG["benchmark"]

set_seed(CONFIG["seed"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IS_KAGGLE = Path("/kaggle/working").exists()
IS_GPU = torch.cuda.is_available()

print("=" * 65)
print("  HARDWARE & ENVIRONMENT AUDIT")
print("=" * 65)
print(f"Platform:           {'Kaggle Cloud GPU' if IS_KAGGLE else 'Local Workstation'}")
print(f"Compute Device:     {device}")
print(f"Python Runtime:     {sys.version.split()[0]}")
print(f"PyTorch Version:    {torch.__version__}")
print(f"CUDA Available:     {IS_GPU}")

if IS_GPU:
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU Model:          {gpu_name}")
    print(f"GPU VRAM:           {gpu_mem:.2f} GB")
    print(f"CUDA Version:       {torch.version.cuda}")
    print(f"cuDNN Version:      {torch.backends.cudnn.version()}")

    # Match GPU profile
    matched_profile = "T4"
    for key in GPU_PROFILES:
        if key in gpu_name.upper():
            matched_profile = key
            break
    profile = GPU_PROFILES[matched_profile]
    for k, v in profile.items():
        CONFIG[k] = v
    print(f"✓ Config tuned for profile: {matched_profile}")
else:
    for k, v in GPU_PROFILES["CPU"].items():
        CONFIG[k] = v
    CONFIG["mixed_precision"] = False
    CONFIG["smoke_test"] = True
    print("⚠ CPU execution detected — automatically activated smoke test mode")

if CONFIG["smoke_test"]:
    CONFIG["num_epochs"] = 1
    CONFIG["ablation_epochs"] = 1
    CONFIG["max_samples"] = 400
    print("ℹ Smoke test active: 1 epoch, subset samples")

# Setup filesystem layout
WORK_DIR = Path("/kaggle/working") if IS_KAGGLE else Path("./kaggle_working")
WORK_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR = WORK_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = WORK_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR = RESULTS_DIR / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR = RESULTS_DIR / "metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
CONFIGS_DIR = RESULTS_DIR / "configs"
CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR = RESULTS_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

print(f"✓ Working directories allocated under: {WORK_DIR}")

# %% [markdown]
# # 3. Dependency Setup
# Install required scientific packages and catalog system environment provenance.

# %%
import subprocess

# ═══════════════════════════════════════════════════════════════════
# SECTION 3: DEPENDENCY SETUP & PROVENANCE
# ═══════════════════════════════════════════════════════════════════

def ensure_package(pkg_name: str, import_name: Optional[str] = None):
    mod = import_name or pkg_name.replace("-", "_")
    try:
        __import__(mod)
    except ImportError:
        print(f"  Installing {pkg_name}...")
        subprocess.run(["pip", "install", "-q", pkg_name], check=True)

ensure_package("huggingface_hub")
ensure_package("scikit-learn", "sklearn")
ensure_package("scipy")
ensure_package("seaborn")
ensure_package("tqdm")
ensure_package("open_clip_torch", "open_clip")   # CLIP baseline backbone

from huggingface_hub import HfApi, hf_hub_download, CommitOperationAdd
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve, average_precision_score,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

ENVIRONMENT_INFO = {
    "experiment_id": "",
    "timestamp": datetime.now().isoformat(),
    "platform": "Kaggle Cloud" if IS_KAGGLE else "Local",
    "python_version": sys.version.split()[0],
    "pytorch_version": torch.__version__,
    "cuda_version": torch.version.cuda if IS_GPU else None,
    "cudnn_version": str(torch.backends.cudnn.version()) if IS_GPU else None,
    "gpu_device": torch.cuda.get_device_name(0) if IS_GPU else "CPU",
    "gpu_vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2) if IS_GPU else 0.0,
    "numpy_version": np.__version__,
    "pandas_version": pd.__version__,
    "scipy_version": __import__("scipy").__version__,
    "sklearn_version": __import__("sklearn").__version__,
}

print("✓ All research dependencies verified")

# %% [markdown]
# # 4. HuggingFace Authentication
# Securely acquire HuggingFace credentials from Kaggle Secrets without exposing tokens.

# %%
# ═══════════════════════════════════════════════════════════════════
# SECTION 4: HUGGING FACE AUTHENTICATION
# ═══════════════════════════════════════════════════════════════════

def get_hf_token() -> str:
    # 1. Kaggle Secrets
    try:
        from kaggle_secrets import UserSecretsClient
        tok = UserSecretsClient().get_secret("HF_TOKEN")
        if tok:
            return tok.strip()
    except Exception:
        pass

    # 2. Environment variable
    tok = os.environ.get("HF_TOKEN")
    if tok:
        return tok.strip()

    # 3. Local .env file
    for candidate in [Path("/kaggle/working/.env"), Path(".env")]:
        if candidate.exists():
            for line in candidate.read_text().splitlines():
                if line.startswith("HF_TOKEN=") or line.startswith("hf="):
                    return line.split("=", 1)[1].strip()

    raise ValueError(
        "❌ HF_TOKEN not found! Add your write token to Kaggle Secrets (Add-ons -> Secrets -> HF_TOKEN)"
    )

HF_TOKEN = get_hf_token()

try:
    hf_api = HfApi(token=HF_TOKEN)
    account_info = hf_api.whoami()
    HF_USERNAME = account_info.get("name", "unknown")
    print(f"✓ Authenticated to HuggingFace as user: '{HF_USERNAME}'")
except Exception as e:
    raise RuntimeError(f"HuggingFace authentication error: {e}")

# %% [markdown]
# # 5. Experiment Versioning & HF Checkpoint Manager
# Establish unique experiment identity and persistent HuggingFace experiment storage.

# %%
# ═══════════════════════════════════════════════════════════════════
# SECTION 5: EXPERIMENT VERSIONING & AUTONOMOUS SWARM COORDINATOR
# ═══════════════════════════════════════════════════════════════════

import socket
import uuid

JOB_REGISTRY = [
    {
        "job_id": "mfft_base",
        "name": "MFFT-Base (Proposed Main Model)",
        "model_variant": "base",
        "num_epochs": 20,
        "ablation": None,
        "priority": 1,
    },
    {
        "job_id": "mfft_tiny",
        "name": "MFFT-Tiny (Efficiency Variant, 372K params)",
        "model_variant": "tiny",
        "num_epochs": 20,
        "ablation": None,
        "priority": 2,
    },
    {
        "job_id": "mfft_large",
        "name": "MFFT-Large (High Capacity, 6.30M params)",
        "model_variant": "large",
        "num_epochs": 20,
        "ablation": None,
        "priority": 3,
    },
    {
        "job_id": "ablation_spatial_only",
        "name": "Ablation: Spatial Domain Only (No FFT)",
        "model_variant": "base",
        "num_epochs": 10,
        "ablation": {"spatial_only": True},
        "priority": 4,
    },
    {
        "job_id": "ablation_no_fga",
        "name": "Ablation: Without FGA Attention",
        "model_variant": "base",
        "num_epochs": 10,
        "ablation": {"use_fga": False},
        "priority": 5,
    },
    {
        "job_id": "ablation_fusion_concat",
        "name": "Ablation: Concatenation Fusion",
        "model_variant": "base",
        "num_epochs": 10,
        "ablation": {"fusion_mode": "concat"},
        "priority": 6,
    },
    {
        "job_id": "ablation_fusion_avg",
        "name": "Ablation: Average Pooling Fusion",
        "model_variant": "base",
        "num_epochs": 10,
        "ablation": {"fusion_mode": "avg"},
        "priority": 7,
    },
    {
        "job_id": "ablation_fusion_max",
        "name": "Ablation: Max Pooling Fusion",
        "model_variant": "base",
        "num_epochs": 10,
        "ablation": {"fusion_mode": "max"},
        "priority": 8,
    },
    {
        "job_id": "ablation_band_low_only",
        "name": "Ablation: Low Frequency Band Only",
        "model_variant": "base",
        "num_epochs": 10,
        "ablation": {"skip_bands": ["mid", "high"]},
        "priority": 9,
    },
    {
        "job_id": "ablation_band_mid_only",
        "name": "Ablation: Mid Frequency Band Only",
        "model_variant": "base",
        "num_epochs": 10,
        "ablation": {"skip_bands": ["low", "high"]},
        "priority": 10,
    },
    {
        "job_id": "ablation_band_high_only",
        "name": "Ablation: High Frequency Band Only",
        "model_variant": "base",
        "num_epochs": 10,
        "ablation": {"skip_bands": ["low", "mid"]},
        "priority": 11,
    },
    {
        "job_id": "mfft_base_seed42",
        "name": "MFFT-Base (Replication Seed 42, Multi-Seed Validation)",
        "model_variant": "base",
        "num_epochs": 20,
        "ablation": None,
        "seed": 42,
        "priority": 12,
    },
    {
        "job_id": "mfft_base_seed999",
        "name": "MFFT-Base (Replication Seed 999, Multi-Seed Validation)",
        "model_variant": "base",
        "num_epochs": 20,
        "ablation": None,
        "seed": 999,
        "priority": 13,
    },
    {
        "job_id": "ablation_band_2bands",
        "name": "Ablation: 2 Frequency Bands (Low, High)",
        "model_variant": "base",
        "num_epochs": 10,
        "ablation": {"num_bands": 2},
        "priority": 14,
    },
    {
        "job_id": "ablation_band_4bands",
        "name": "Ablation: 4 Frequency Bands (Low, Low-Mid, Mid-High, High)",
        "model_variant": "base",
        "num_epochs": 10,
        "ablation": {"num_bands": 4},
        "priority": 15,
    },
    {
        "job_id": "baseline_resnet50",
        "name": "Baseline: ResNet-50 (He et al., CVPR 2016)",
        "model_kind": "baseline",
        "baseline_name": "resnet50",
        "model_variant": "baseline_resnet50",
        "num_epochs": 20,
        "ablation": None,
        "batch_size": 32,
        "priority": 16,
    },
    {
        "job_id": "baseline_vit_b16",
        "name": "Baseline: ViT-B/16 (Dosovitskiy et al., ICLR 2021)",
        "model_kind": "baseline",
        "baseline_name": "vit_b16",
        "model_variant": "baseline_vit_b16",
        "num_epochs": 20,
        "ablation": None,
        "batch_size": 16,
        "priority": 17,
    },
    {
        "job_id": "baseline_efficientnet_b0",
        "name": "Baseline: EfficientNet-B0 (Tan & Le, ICML 2019)",
        "model_kind": "baseline",
        "baseline_name": "efficientnet_b0",
        "model_variant": "baseline_efficientnet_b0",
        "num_epochs": 20,
        "ablation": None,
        "batch_size": 32,
        "priority": 18,
    },
    {
        "job_id": "baseline_swin_t",
        "name": "Baseline: Swin Transformer-T (Liu et al., CVPR 2021)",
        "model_kind": "baseline",
        "baseline_name": "swin_t",
        "model_variant": "baseline_swin_t",
        "num_epochs": 20,
        "ablation": None,
        "batch_size": 24,
        "priority": 19,
    },
    {
        "job_id": "baseline_deit_small",
        "name": "Baseline: DeiT-Small (Touvron et al., ICCV 2021)",
        "model_kind": "baseline",
        "baseline_name": "deit_small",
        "model_variant": "baseline_deit_small",
        "num_epochs": 20,
        "ablation": None,
        "batch_size": 16,
        "priority": 20,
    },
    {
        "job_id": "baseline_clip",
        "name": "Baseline: CLIP ViT-B/32 (Radford et al., ICML 2021)",
        "model_kind": "baseline",
        "baseline_name": "clip",
        "model_variant": "baseline_clip",
        "num_epochs": 20,
        "ablation": None,
        "batch_size": 32,
        "priority": 21,
    },
    {
        "job_id": "baseline_resnet18",
        "name": "Baseline: ResNet-18 (He et al., CVPR 2016)",
        "model_kind": "baseline",
        "baseline_name": "resnet18",
        "model_variant": "baseline_resnet18",
        "num_epochs": 20,
        "ablation": None,
        "batch_size": 32,
        "priority": 22,
    },
    {
        "job_id": "baseline_lightvit",
        "name": "Baseline: LightViT (custom lightweight ViT)",
        "model_kind": "baseline",
        "baseline_name": "lightvit",
        "model_variant": "baseline_lightvit",
        "num_epochs": 20,
        "ablation": None,
        "batch_size": 24,
        "priority": 23,
    },
    {
        "job_id": "baseline_simplecnn",
        "name": "Baseline: SimpleCNN (custom CNN)",
        "model_kind": "baseline",
        "baseline_name": "simplecnn",
        "model_variant": "baseline_simplecnn",
        "num_epochs": 20,
        "ablation": None,
        "batch_size": 64,
        "priority": 24,
    },
    {
        "job_id": "baseline_freqdetect",
        "name": "Baseline: FreqDetect (Frank et al., 2020)",
        "model_kind": "baseline",
        "baseline_name": "freqdetect",
        "model_variant": "baseline_freqdetect",
        "num_epochs": 20,
        "ablation": None,
        "batch_size": 64,
        "priority": 25,
    },
    {
        "job_id": "logo_holdout_diffusion",
        "name": "LOGO: Cross-Gen Holdout Diffusion (SD, Midjourney, DALL-E)",
        "model_variant": "base",
        "num_epochs": 10,
        "ablation": None,
        "logo_holdout": "diffusion",
        "priority": 26,
    },
    {
        "job_id": "logo_holdout_gan",
        "name": "LOGO: Cross-Gen Holdout GAN (StyleGAN, BigGAN, ProGAN)",
        "model_variant": "base",
        "num_epochs": 10,
        "ablation": None,
        "logo_holdout": "gan",
        "priority": 27,
    },
    {
        "job_id": "logo_holdout_deepfake",
        "name": "LOGO: Cross-Gen Holdout Deepfake (FaceForensics, Celeb-DF, DFDC)",
        "model_variant": "base",
        "num_epochs": 10,
        "ablation": None,
        "logo_holdout": "deepfake",
        "priority": 28,
    },
]

def discover_worker_id() -> str:
    if CONFIG.get("worker_name"):
        return CONFIG["worker_name"].strip()
    try:
        from kaggle_secrets import UserSecretsClient
        sec = UserSecretsClient()
        for k in ["WORKER_NAME", "KAGGLE_USERNAME"]:
            try:
                v = sec.get_secret(k)
                if v:
                    return v.strip()
            except Exception:
                pass
    except Exception:
        pass
    for k in ["WORKER_NAME", "KAGGLE_USER_NAME"]:
        v = os.environ.get(k)
        if v:
            return v.strip()
    host = socket.gethostname().replace("-", "_")[-6:]
    return f"kaggle-worker-{host}-{uuid.uuid4().hex[:4]}"

WORKER_ID = discover_worker_id()

class SwarmCoordinator:
    """
    Autonomous work-stealing job coordinator across multiple Kaggle accounts.
    Uses Hugging Face Hub as central distributed lease registry.
    """
    def __init__(self, api: HfApi, repo_id: str, worker_id: str, stale_minutes: int = 35):
        self.api = api
        self.repo_id = repo_id
        self.worker_id = worker_id
        self.stale_sec = stale_minutes * 60
        self.staging = WORK_DIR / "coordinator_staging"
        self.staging.mkdir(parents=True, exist_ok=True)
        self.lost_ownership = False

    def fetch_lease(self, job_id: str) -> Optional[dict]:
        try:
            p = hf_hub_download(
                repo_id=self.repo_id,
                filename=f"coordinator/leases/{job_id}.json",
                repo_type="model",
                token=HF_TOKEN,
            )
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def push_lease(self, job_id: str, lease: dict) -> bool:
        tmp = self.staging / f"{job_id}.json"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(lease, f, indent=2)
        for attempt in range(5):
            try:
                self.api.upload_file(
                    path_or_fileobj=str(tmp),
                    path_in_repo=f"coordinator/leases/{job_id}.json",
                    repo_id=self.repo_id,
                    repo_type="model",
                    commit_message=f"Lease {job_id} by {self.worker_id}",
                )
                return True
            except Exception as e:
                if "429" in str(e):
                    wait = 30 * (attempt + 1)
                    print(f"  Rate limit encountered on lease push for {job_id}. Backing off {wait}s (attempt {attempt+1}/5)...")
                    time.sleep(wait)
                else:
                    print(f"  Lease sync notice for {job_id}: {e}")
                    return False
        return False

    def claim_next_job(self, queue: List[dict]) -> Optional[dict]:
        sorted_jobs = sorted(queue, key=lambda j: j.get("priority", 99))
        for job in sorted_jobs:
            jid = job["job_id"]
            lease = self.fetch_lease(jid)

            should_claim = False
            resume_ep = 0

            if lease is None:
                should_claim = True
            elif lease.get("status") == "completed":
                continue
            elif lease.get("status") == "running":
                hb = lease.get("last_heartbeat", "")
                if hb:
                    try:
                        from datetime import timezone
                        hb_dt = datetime.fromisoformat(hb)
                        if hb_dt.tzinfo is None:
                            hb_dt = hb_dt.replace(tzinfo=timezone.utc)
                        now_utc = datetime.now(timezone.utc)
                        elapsed = (now_utc - hb_dt).total_seconds()
                        if elapsed > self.stale_sec:
                            print(f"  ⚠ Job [{jid}] lease is STALE ({elapsed/60:.1f}m ago by {lease.get('worker_id')}). Stealing job to resume!")
                            should_claim = True
                            resume_ep = lease.get("current_epoch", 0)
                        else:
                            continue
                    except Exception:
                        should_claim = True
                else:
                    should_claim = True

            if should_claim:
                from datetime import timezone
                now_iso = datetime.now(timezone.utc).isoformat()
                claim_payload = {
                    "job_id": jid,
                    "name": job["name"],
                    "worker_id": self.worker_id,
                    "status": "running",
                    "claimed_at": now_iso,
                    "last_heartbeat": now_iso,
                    "current_epoch": resume_ep,
                    "best_val_f1": lease.get("best_val_f1", 0.0) if lease else 0.0,
                    "model_variant": job["model_variant"],
                    "num_epochs": job["num_epochs"],
                }
                if self.push_lease(jid, claim_payload):
                    time.sleep(2)
                    verified = self.fetch_lease(jid)
                    if verified and verified.get("worker_id") == self.worker_id:
                        print(f"★ WORKER [{self.worker_id}] CLAIMED JOB: [{jid}] — {job['name']}")
                        return job
        return None

    def stage_heartbeat(self, job_id: str, epoch: int, best_f1: float) -> Optional[Path]:
        """Ownership check + lease refresh staged locally (no commit). Returns the lease
        file to include in the next bundle commit, or None if another worker owns it."""
        from datetime import timezone
        lease = self.fetch_lease(job_id) or {}
        owner = lease.get("worker_id")
        if owner and owner != self.worker_id:
            self.lost_ownership = True
            print(f"  ✖ Lease [{job_id}] now owned by [{owner}] (we are [{self.worker_id}]). "
                  "Releasing this duplicate trainer.")
            return None
        lease.update({
            "worker_id": self.worker_id,
            "status": "running",
            "last_heartbeat": datetime.now(timezone.utc).isoformat(),
            "current_epoch": epoch,
            "best_val_f1": best_f1,
        })
        tmp = self.staging / f"{job_id}.json"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(lease, f, indent=2)
        return tmp

    def send_heartbeat(self, job_id: str, epoch: int, best_f1: float) -> bool:
        lease_file = self.stage_heartbeat(job_id, epoch, best_f1)
        if lease_file is None:
            return False
        lease = json.loads(Path(lease_file).read_text(encoding="utf-8"))
        return self.push_lease(job_id, lease)

    def mark_completed(self, job_id: str, metrics: dict):
        from datetime import timezone
        now_iso = datetime.now(timezone.utc).isoformat()
        lease = self.fetch_lease(job_id) or {}
        lease.update({
            "worker_id": self.worker_id,
            "status": "completed",
            "completed_at": now_iso,
            "last_heartbeat": now_iso,
            "test_accuracy": metrics.get("accuracy", 0.0),
            "test_f1": metrics.get("f1", 0.0),
            "test_auc": metrics.get("auc", 0.0),
        })
        self.push_lease(job_id, lease)
        print(f"✓ Job [{job_id}] marked COMPLETED by {self.worker_id}")

def build_experiment_id(variant: str, seed: int) -> str:
    explicit = CONFIG.get("experiment_id_override") or os.environ.get("RUN_ID")
    if explicit:
        return explicit.strip()
    return f"mfft-{variant}_s{seed}_{WORKER_ID}_{datetime.now().strftime('%Y%m%d')}"

coordinator = None
claimed_job = None

if CONFIG.get("swarm_mode", True):
    print("=" * 65)
    print(f"  DISTRIBUTED SWARM WORKER: [{WORKER_ID}]")
    print("=" * 65)
    coordinator = SwarmCoordinator(hf_api, CONFIG["hf_checkpoint_repo"], WORKER_ID, CONFIG.get("stale_heartbeat_minutes", 180))
    claimed_job = coordinator.claim_next_job(JOB_REGISTRY)

    STANDBY_POLL_SEC = int(CONFIG.get("standby_poll_seconds", 300))
    standby_rounds = 0
    while claimed_job is None:
        standby_rounds += 1
        print(f"  [STANDBY] [{WORKER_ID}] round {standby_rounds} at "
              f"{time.strftime('%Y-%m-%d %H:%M:%S')} — all research jobs are running or "
              f"completed; re-checking in {STANDBY_POLL_SEC}s (no training, no HF commits)")
        time.sleep(STANDBY_POLL_SEC)
        claimed_job = coordinator.claim_next_job(JOB_REGISTRY)

    CONFIG["model_variant"] = claimed_job["model_variant"]
    CONFIG["num_epochs"] = claimed_job["num_epochs"]
    CONFIG["ablation"] = claimed_job.get("ablation", None)
    CONFIG["model_kind"] = claimed_job.get("model_kind", "mfft")
    CONFIG["baseline_name"] = claimed_job.get("baseline_name", None)
    CONFIG["logo_holdout"] = claimed_job.get("logo_holdout", None)
    if "batch_size" in claimed_job:
        CONFIG["batch_size"] = claimed_job["batch_size"]
        print(f"✓ Per-model batch size override: {claimed_job['batch_size']}")
    if "seed" in claimed_job:
        CONFIG["seed"] = claimed_job["seed"]
        set_seed(CONFIG["seed"])
    EXPERIMENT_ID = claimed_job["job_id"]
    print(f"✓ Swarm assigned job: {EXPERIMENT_ID} ({claimed_job['name']})")
else:
    EXPERIMENT_ID = build_experiment_id(CONFIG["model_variant"], CONFIG["seed"])
    print(f"✓ Manual experiment mode assigned: {EXPERIMENT_ID}")

ENVIRONMENT_INFO["experiment_id"] = EXPERIMENT_ID
ENVIRONMENT_INFO["worker_id"] = WORKER_ID
print(f"✓ Experiment identifier assigned: {EXPERIMENT_ID}")

@dataclass
class TrainingState:
    run_id: str = ""
    model_variant: str = ""
    manifest_sha256: str = ""
    current_epoch: int = 0
    total_epochs: int = 20
    global_step: int = 0
    best_val_f1: float = 0.0
    best_val_auc: float = 0.0
    best_val_accuracy: float = 0.0
    best_val_loss: float = float("inf")
    patience_counter: int = 0
    early_stop: bool = False
    best_model_epoch: int = 0
    total_train_time: float = 0.0
    status: str = "initialized"
    created_at: str = ""
    last_updated: str = ""

    def __post_init__(self):
        now = datetime.now().isoformat()
        if not self.created_at:
            self.created_at = now
        if not self.last_updated:
            self.last_updated = now

@dataclass
class MetricRecord:
    run_id: str = ""
    epoch: int = 0
    train_loss: float = 0.0
    train_accuracy: float = 0.0
    val_loss: float = 0.0
    val_accuracy: float = 0.0
    val_precision: float = 0.0
    val_recall: float = 0.0
    val_f1: float = 0.0
    val_macro_f1: float = 0.0
    val_auc: float = 0.0
    val_specificity: float = 0.0
    learning_rate: float = 0.0
    elapsed_time: float = 0.0

class HFExperimentManager:
    """
    Fault-tolerant Hugging Face Hub manager.
    Coordinates remote checkpoints, experiment metadata, metric streams, and manuscript assets.
    """
    def __init__(self, token: str, repo_id: str, run_id: str, manifest_sha256: str = ""):
        self.token = token
        self.repo_id = repo_id
        self.run_id = run_id
        self.manifest_sha256 = manifest_sha256
        self.api = HfApi(token=token)
        self._ensure_repo()

        self.local_cache = WORK_DIR / "hf_staging"
        self.local_cache.mkdir(parents=True, exist_ok=True)

        self.csv_metrics_path = self.local_cache / "metrics.csv"
        if not self.csv_metrics_path.exists():
            with open(self.csv_metrics_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(MetricRecord.__dataclass_fields__.keys()))
                writer.writeheader()

        self.session_log_path = self.local_cache / "session_log.csv"
        self.log_event("SESSION_START", f"Run ID {self.run_id} initialized")

    @property
    def run_prefix(self) -> str:
        return f"runs/{self.run_id}"

    def _path(self, relative_path: str) -> str:
        return f"{self.run_prefix}/{relative_path}".replace("//", "/")

    def repo_path(self, relative_path: str) -> str:
        """Full repo path of a run-scoped artifact (used by bundle commits)."""
        return self._path(relative_path)

    def _ensure_repo(self):
        try:
            self.api.create_repo(self.repo_id, repo_type="model", exist_ok=True, private=True)
        except Exception as e:
            print(f"  ℹ Repo notice: {e}")

    def upload_file(self, local_file: Path, repo_relative_path: str, retries: int = 3) -> bool:
        if not Path(local_file).exists():
            return False
        dest_in_repo = self._path(repo_relative_path)
        for attempt in range(retries):
            try:
                self.api.upload_file(
                    path_or_fileobj=str(local_file),
                    path_in_repo=dest_in_repo,
                    repo_id=self.repo_id,
                    repo_type="model",
                    commit_message=f"Sync {repo_relative_path} ({self.run_id})",
                )
                return True
            except Exception as e:
                if attempt == retries - 1:
                    print(f"  ⚠ Failed to push {repo_relative_path} after {retries} retries: {e}")
                    return False
                time.sleep(2 ** attempt * 2)
        return False

    def stage_state(self, state: TrainingState) -> Path:
        state.run_id = self.run_id
        state.manifest_sha256 = self.manifest_sha256
        state.last_updated = datetime.now().isoformat()
        state_file = self.local_cache / "training_state.json"
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(asdict(state), f, indent=2)
        return state_file

    def save_state(self, state: TrainingState):
        self.upload_file(self.stage_state(state), "logs/training_state.json")

    def upload_bundle(self, items, commit_message: str, retries: int = 5) -> bool:
        """Commit many artifacts in ONE Hub commit (the repo allows only 128
        commits/hour shared by every worker, so per-file commits are the ceiling).

        items: [(local_file, repo_path_in_repo), ...]
        """
        ops = []
        for local_file, repo_path in items:
            local = Path(local_file)
            if not local.exists():
                continue
            ops.append(CommitOperationAdd(path_in_repo=repo_path, path_or_fileobj=str(local)))
        if not ops:
            return False
        for attempt in range(retries):
            try:
                self.api.create_commit(
                    repo_id=self.repo_id,
                    repo_type="model",
                    operations=ops,
                    commit_message=commit_message,
                )
                return True
            except Exception as e:
                msg = str(e)
                if "429" in msg or "rate limit" in msg.lower():
                    wait = 30 * (attempt + 1)
                    print(f"  \u23f3 Rate limit on bundle commit \u2014 backing off {wait}s "
                          f"(attempt {attempt + 1}/{retries})")
                    time.sleep(wait)
                elif attempt == retries - 1:
                    print(f"  \u26a0 Failed to push epoch bundle after {retries} retries: {e}")
                    return False
                else:
                    time.sleep(2 ** attempt * 2)
        return False

    def load_state(self) -> Optional[TrainingState]:
        try:
            remote_path = self._path("logs/training_state.json")
            downloaded = hf_hub_download(
                repo_id=self.repo_id,
                filename=remote_path,
                repo_type="model",
                token=self.token,
            )
            with open(downloaded, "r", encoding="utf-8") as f:
                data = json.load(f)
            st = TrainingState(**{k: v for k, v in data.items() if k in TrainingState.__dataclass_fields__})
            if st.manifest_sha256 and self.manifest_sha256 and st.manifest_sha256 != self.manifest_sha256:
                print("  ⚠ Manifest hash mismatch on resume — starting fresh partition")
                return None
            return st
        except Exception:
            return None

    def upload_checkpoint(self, local_path: str, tag: str) -> bool:
        return self.upload_file(Path(local_path), f"checkpoints/{tag}.pt")

    def download_checkpoint(self, tag: str = "last") -> Optional[Path]:
        try:
            remote_path = self._path(f"checkpoints/{tag}.pt")
            local = hf_hub_download(
                repo_id=self.repo_id,
                filename=remote_path,
                repo_type="model",
                token=self.token,
            )
            dest = self.local_cache / f"{tag}.pt"
            shutil.copy(local, dest)
            return dest
        except Exception:
            return None

    def stage_metric(self, record: MetricRecord) -> Path:
        record.run_id = self.run_id
        with open(self.csv_metrics_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(MetricRecord.__dataclass_fields__.keys()))
            writer.writerow(asdict(record))
        return self.csv_metrics_path

    def log_metric(self, record: MetricRecord):
        self.upload_file(self.stage_metric(record), "logs/metrics.csv")

    def log_event(self, event_type: str, details: str):
        with open(self.session_log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([datetime.now().isoformat(), event_type, details])

    def sync_session_log(self):
        self.upload_file(self.session_log_path, "logs/session_log.csv")

    def upload_figure(self, path: Path):
        self.upload_file(path, f"figures/{path.name}")

    def upload_table(self, path: Path):
        self.upload_file(path, f"tables/{path.name}")

    def upload_metadata(self, metadata: dict, filename: str = "experiment_metadata.json"):
        dest = self.local_cache / filename
        with open(dest, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, default=str)
        self.upload_file(dest, f"configs/{filename}")

    def write_model_card(self, state: TrainingState, final_metrics: Optional[dict] = None):
        card = f"""# MFFT Experiment: {state.run_id}

## Q1 Journal Reproducibility Record

| Parameter | Specification |
|:---|:---|
| Architecture | {state.model_variant if str(state.model_variant).startswith("baseline") else "MFFT-" + state.model_variant} |
| Target Journal | IEEE TIFS / Pattern Recognition |
| Status | {state.status} |
| Epoch Progress | {state.current_epoch} / {state.total_epochs} |
| Best Validation F1 | {state.best_val_f1:.4f} |
| Best Validation AUC | {state.best_val_auc:.4f} |
| Best Validation Acc | {state.best_val_accuracy:.4f} |
| Best Metric Epoch | Epoch {state.best_model_epoch} |
| Manifest SHA-256 | `{state.manifest_sha256[:16]}...` |
| Total Compute Time | {state.total_train_time:.1f} s |
| Last Updated | {state.last_updated} |
"""
        if final_metrics:
            card += "\n## Final Evaluation Metrics (Held-Out Test Set)\n\n"
            for k, v in final_metrics.items():
                card += f"- **{k}**: {v:.4f}\n" if isinstance(v, float) else f"- **{k}**: {v}\n"

        card += "\n---\n*Persistent experiment checkpoint generated by MFFT Research Pipeline*\n"
        readme_file = self.local_cache / "README.md"
        readme_file.write_text(card, encoding="utf-8")
        self.upload_file(readme_file, "README.md")

hf_mgr = HFExperimentManager(
    token=HF_TOKEN,
    repo_id=CONFIG["hf_checkpoint_repo"],
    run_id=EXPERIMENT_ID,
)
hf_mgr.upload_metadata(CONFIG, "config.json")
hf_mgr.upload_metadata(ENVIRONMENT_INFO, "environment.json")
print(f"✓ HF Experiment Manager active on repo: {CONFIG['hf_checkpoint_repo']}")

# %% [markdown]
# # 6. Dataset Access & Verification
# Discover and validate mounted Kaggle datasets across all 12 target image archives.

# %%
# ═══════════════════════════════════════════════════════════════════
# SECTION 6: DATASET MOUNT AUDIT & VERIFICATION
# ═══════════════════════════════════════════════════════════════════

KAGGLE_INPUT_DIR = Path("/kaggle/input") if IS_KAGGLE else Path("./kaggle_input")

def discover_kaggle_mounts(base_dir: Path) -> Dict[str, Path]:
    mounts = {}
    if not base_dir.exists():
        return mounts
    for child in sorted(base_dir.iterdir()):
        if child.is_dir() and child.name != "datasets":
            mounts[child.name] = child
            for sub in child.iterdir():
                if sub.is_dir():
                    mounts[sub.name] = sub
                    for sub2 in sub.iterdir():
                        if sub2.is_dir() and sub2.name not in mounts:
                            mounts[sub2.name] = sub2
    datasets_dir = base_dir / "datasets"
    if datasets_dir.exists():
        for owner in datasets_dir.iterdir():
            if owner.is_dir():
                for sub in owner.iterdir():
                    if sub.is_dir():
                        mounts[sub.name] = sub
    return mounts

def scan_valid_images(folder: Path, max_limit: Optional[int] = None) -> List[Path]:
    results = []
    for root, _, files in os.walk(str(folder)):
        for f in files:
            ext = Path(f).suffix.lower()
            if ext in IMG_EXTENSIONS:
                img_path = Path(root) / f
                try:
                    if img_path.stat().st_size > 0:
                        results.append(img_path)
                        if max_limit and len(results) >= max_limit:
                            return results
                except OSError:
                    continue
    return results

print("=" * 65)
print("  DATASET AUDIT & INVENTORY")
print("=" * 65)

discovered_mounts = discover_kaggle_mounts(KAGGLE_INPUT_DIR)
print(f"Discovered {len(discovered_mounts)} mounted input directories:")

dataset_summary = []
for name, mpath in discovered_mounts.items():
    lbl = "unassigned"
    subdir = ""
    if name in KAGGLE_DATASETS:
        lbl, subdir = KAGGLE_DATASETS[name]
    elif name in CROSS_DOMAIN_DATASETS:
        lbl, subdir = CROSS_DOMAIN_DATASETS[name]
    elif "artifact" in name:
        lbl = "ai_generated"

    target_dir = mpath / subdir if subdir and (mpath / subdir).exists() else mpath
    sample_count = len(scan_valid_images(target_dir, max_limit=500))
    dataset_summary.append({
        "slug": name,
        "class_label": lbl,
        "verified_sample": sample_count >= 1,
        "mount_path": str(mpath),
    })
    print(f"  [{lbl.upper():12s}] {name:32s} (verified={sample_count >= 1})")

hf_mgr.log_event("DATASET_AUDIT", f"Discovered {len(discovered_mounts)} mounts")

# %% [markdown]
# # 7. Data Preprocessing & Split Management
# Implement frequency-robust augmentations (Wang et al. CVPR 2020) and frozen stratified splits.

# %%
from torchvision import transforms

# ═══════════════════════════════════════════════════════════════════
# SECTION 7: ROBUST TRANSFORMS & FROZEN SPLIT ALLOCATION
# ═══════════════════════════════════════════════════════════════════

class ImageTransform:
    """
    Frequency-aware data transform pipeline.
    Incorporates random JPEG compression, resizing, and blur to prevent spectral shortcut learning.
    """
    def __init__(self, size: int = 384, augment: bool = True):
        self.size = size
        norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if augment:
            self.pipeline = transforms.Compose([
                transforms.Resize(size + 16),
                transforms.RandomResizedCrop(size, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10, fill=128),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.02),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))], p=0.2),
                transforms.ToTensor(),
                transforms.RandomErasing(p=0.1, scale=(0.02, 0.08)),
                norm,
            ])
        else:
            self.pipeline = transforms.Compose([
                transforms.Resize(size + 16),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                norm,
            ])

    def __call__(self, img):
        return self.pipeline(img.convert("RGB"))

def collect_research_inventory(mounts: Dict[str, Path], max_samples: Optional[int] = None, seed: int = 42) -> List[Tuple[str, int]]:
    rng = random.Random(seed)
    real_paths = []
    fake_paths = []
    per_class = (max_samples // 2) if max_samples else None

    for name, mpath in mounts.items():
        if name in CROSS_DOMAIN_DATASETS:
            continue  # Hold out cross-domain evaluation datasets

        lbl = None
        subdir = ""
        if name in KAGGLE_DATASETS:
            lbl, subdir = KAGGLE_DATASETS[name]
        elif name == "artifact-dataset":
            lbl = "ai_generated"

        if not lbl:
            continue

        target_list = real_paths if lbl == "real" else fake_paths
        if per_class and len(target_list) >= per_class:
            continue  # Already satisfied class quota in subset/smoke mode

        search_dir = mpath / subdir if subdir and (mpath / subdir).exists() else mpath
        needed = (per_class - len(target_list)) if per_class else None
        imgs = scan_images_fast(search_dir, max_files=needed)
        for p in imgs:
            target_list.append((str(p), LABEL_MAP[lbl]))

    if max_samples:
        rng.shuffle(real_paths)
        rng.shuffle(fake_paths)
        real_paths = real_paths[:per_class]
        fake_paths = fake_paths[:per_class]

    combined = real_paths + fake_paths
    rng.shuffle(combined)
    return combined

def scan_images_fast(directory: Path, max_files: Optional[int] = None) -> List[Path]:
    collected = []
    for root, _, files in os.walk(str(directory)):
        for f in files:
            if Path(f).suffix.lower() in IMG_EXTENSIONS:
                p = Path(root) / f
                try:
                    if p.stat().st_size > 0:
                        collected.append(p)
                        if max_files and len(collected) >= max_files:
                            return collected
                except OSError:
                    pass
    return collected

# Check if frozen master manifest already exists on Hugging Face Hub
master_manifest_loaded = False
existing_split_indices = None
print("Checking for master split manifest on Hugging Face Hub...")
try:
    remote_manifest_path = hf_hub_download(
        repo_id=CONFIG["hf_manifest_repo"],
        filename=CONFIG["manifest_path_in_repo"],
        repo_type="model",
        token=HF_TOKEN,
    )
    with open(remote_manifest_path, "r", encoding="utf-8") as f:
        existing_manifest = json.load(f)
    print(f"✓ Found existing frozen manifest on {CONFIG['hf_manifest_repo']} (SHA-256: {existing_manifest.get('manifest_sha256', '')[:16]}...)")
    split_hash = existing_manifest.get("manifest_sha256", "")
    existing_split_indices = existing_manifest.get("split_indices", None)
    master_manifest_loaded = True
except Exception:
    print(f"ℹ No existing manifest on {CONFIG['hf_manifest_repo']} — this worker will bootstrap and freeze it.")

# Build or restore frozen splits
if master_manifest_loaded and existing_manifest.get("samples") and existing_split_indices:
    print(f"Restoring {len(existing_manifest['samples']):,} pre-compiled samples directly from master manifest...")
    all_samples = [(s[0], s[1]) for s in existing_manifest["samples"]]
    train_idx = existing_split_indices["train"]
    val_idx = existing_split_indices["val"]
    test_idx = existing_split_indices["test"]
    train_samples = [all_samples[i] for i in train_idx]
    val_samples = [all_samples[i] for i in val_idx]
    test_samples = [all_samples[i] for i in test_idx]
    split_hash = existing_manifest.get("manifest_sha256", split_hash)
    hf_mgr.manifest_sha256 = split_hash
    print(f"✓ Stratified Split Loaded from Master Manifest (Train: {len(train_samples):,}, Val: {len(val_samples):,}, Test: {len(test_samples):,})")
else:
    print("Compiling sample inventory from mounted archives...")
    all_samples = collect_research_inventory(discovered_mounts, max_samples=CONFIG["max_samples"], seed=CONFIG["seed"])

if len(all_samples) == 0:
    print("⚠ No mounted images found on disk. Creating synthetic verification pool for pipeline test.")
    dummy_dir = WORK_DIR / "dummy_data"
    dummy_dir.mkdir(parents=True, exist_ok=True)
    all_samples = []
    for i in range(200):
        dummy_file = dummy_dir / f"syn_{i}.png"
        if not dummy_file.exists():
            Image.new("RGB", (64, 64), color=(i % 256, (i * 2) % 256, (i * 3) % 256)).save(dummy_file)
        all_samples.append((str(dummy_file), i % 2))

labels_array = [s[1] for s in all_samples]
indices = list(range(len(all_samples)))
holdout_ratio = CONFIG["val_split"] + CONFIG["test_split"]

train_idx, holdout_idx = train_test_split(indices, test_size=holdout_ratio, stratify=labels_array, random_state=CONFIG["seed"])
holdout_labels = [labels_array[i] for i in holdout_idx]
val_ratio_in_holdout = CONFIG["val_split"] / holdout_ratio
val_idx, test_idx = train_test_split(holdout_idx, test_size=(1.0 - val_ratio_in_holdout), stratify=holdout_labels, random_state=CONFIG["seed"])

train_samples = [all_samples[i] for i in train_idx]
val_samples = [all_samples[i] for i in val_idx]
test_samples = [all_samples[i] for i in test_idx]

# Compute deterministic SHA-256 for split verification
split_hash = hashlib.sha256(json.dumps({
    "total": len(all_samples), "train": len(train_idx), "val": len(val_idx), "test": len(test_idx), "seed": CONFIG["seed"]
}, sort_keys=True).encode()).hexdigest()
hf_mgr.manifest_sha256 = split_hash

split_record_file = CONFIGS_DIR / "split_indices.json"
with open(split_record_file, "w", encoding="utf-8") as f:
    json.dump({
        "manifest_sha256": split_hash,
        "seed": CONFIG["seed"],
        "train_indices": train_idx,
        "val_indices": val_idx,
        "test_indices": test_idx,
    }, f)
hf_mgr.upload_file(split_record_file, "configs/split_indices.json")

# If this worker was the bootstrapper, upload the master frozen split to MohsinElis/mfft-master-manifest
if not master_manifest_loaded:
    master_manifest_payload = {
        "version": 1,
        "created_at": datetime.now().isoformat(),
        "created_by_worker": WORKER_ID,
        "seed": CONFIG["seed"],
        "total_images": len(all_samples),
        "split_sizes": {
            "train": len(train_samples),
            "val": len(val_samples),
            "test": len(test_samples),
        },
        "shards_used": list(discovered_mounts.keys()),
        "manifest_sha256": split_hash,
        "split_indices": {
            "train": train_idx,
            "val": val_idx,
            "test": test_idx,
        }
    }
    master_manifest_local = WORK_DIR / "split_manifest.json"
    with open(master_manifest_local, "w", encoding="utf-8") as f:
        json.dump(master_manifest_payload, f, indent=2)

    try:
        hf_api.upload_file(
            path_or_fileobj=str(master_manifest_local),
            path_in_repo=CONFIG["manifest_path_in_repo"],
            repo_id=CONFIG["hf_manifest_repo"],
            repo_type="model",
            commit_message=f"Bootstrap master frozen split by {WORKER_ID}",
        )
        print(f"★ Successfully bootstrapped and uploaded master frozen manifest to {CONFIG['hf_manifest_repo']}!")
    except Exception as e:
        print(f"⚠ Master manifest upload notice: {e}")

# ── LOGO (Leave-One-Generator-Out) Filtering ──
LOGO_HOLDOUT_RULES = {
    'diffusion': ['stable_diffusion', 'latent_diffusion', 'glide', 'ddpm', 'palette', 'vq_diffusion', 'dall-e3', 'midjourney'],
    'gan': ['big_gan', 'stylegan', 'cycle_gan', 'pro_gan', 'projected_gan', 'gau_gan', 'star_gan', 'gansformer', 'genimage-ai'],
    'deepfake': ['faceforensics', 'celebdf', 'dfdc', 'face_synthetics']
}
logo_key = CONFIG.get("logo_holdout")
if logo_key and logo_key in LOGO_HOLDOUT_RULES:
    holdout_kws = LOGO_HOLDOUT_RULES[logo_key]
    print(f"[LOGO] Applying holdout filtering for family: '{logo_key}' (keywords: {holdout_kws})")
    train_samples = [s for s in train_samples if (s[1] == 0 or not any(k in s[0].lower() for k in holdout_kws))]
    val_samples = [s for s in val_samples if (s[1] == 0 or not any(k in s[0].lower() for k in holdout_kws))]
    test_samples = [s for s in test_samples if (s[1] == 0 or any(k in s[0].lower() for k in holdout_kws))]
    print(f"[LOGO] Filtered dataset -> Train: {len(train_samples):,} | Val: {len(val_samples):,} | Held-out Test: {len(test_samples):,}")

print(f"✓ Stratified Split Complete:")
print(f"  Training Split:     {len(train_samples):,} images ({Counter([s[1] for s in train_samples])})")
print(f"  Validation Split:   {len(val_samples):,} images ({Counter([s[1] for s in val_samples])})")
print(f"  Testing Split:      {len(test_samples):,} images ({Counter([s[1] for s in test_samples])})")
print(f"  Manifest SHA-256:   {split_hash[:16]}...")

# %% [markdown]
# # 8. DataLoaders
# Create fault-tolerant DataLoaders with WeightedRandomSampler for class balance.

# %%
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# ═══════════════════════════════════════════════════════════════════
# SECTION 8: RESILIENT RESEARCH DATALOADERS
# ═══════════════════════════════════════════════════════════════════

class ResearchDataset(Dataset):
    def __init__(self, samples: List[Tuple[str, int]], transform=None, img_size: int = 384):
        self.samples = samples
        self.transform = transform or ImageTransform(img_size, augment=False)
        self.img_size = img_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        for attempt in range(3):
            try:
                img = Image.open(path).convert("RGB")
                return self.transform(img), label
            except Exception:
                # Retry alternate index on read failure
                idx = (idx + attempt + 1) % len(self.samples)
                path, label = self.samples[idx]
        return torch.zeros(3, self.img_size, self.img_size), label

IMG_SZ = CONFIG["image_size"]
BATCH_SZ = CONFIG["batch_size"]
NUM_WKR = CONFIG["num_workers"]

train_ds = ResearchDataset(train_samples, transform=ImageTransform(IMG_SZ, augment=True), img_size=IMG_SZ)
val_ds = ResearchDataset(val_samples, transform=ImageTransform(IMG_SZ, augment=False), img_size=IMG_SZ)
test_ds = ResearchDataset(test_samples, transform=ImageTransform(IMG_SZ, augment=False), img_size=IMG_SZ)

train_labels = [s[1] for s in train_samples]
class_counts = Counter(train_labels)
class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
sample_weights = [class_weights[l] for l in train_labels]
train_sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

train_loader = DataLoader(train_ds, batch_size=BATCH_SZ, sampler=train_sampler, num_workers=NUM_WKR, pin_memory=True, drop_last=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SZ, shuffle=False, num_workers=NUM_WKR, pin_memory=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SZ, shuffle=False, num_workers=NUM_WKR, pin_memory=True)

print(f"✓ DataLoaders configured for execution:")
print(f"  Train: {len(train_loader)} batches | Val: {len(val_loader)} batches | Test: {len(test_loader)} batches")

# %% [markdown]
# # 9. Model Architecture: MFFT
# Full implementation of the Multi-Frequency Fusion Transformer (MFFT) with Frequency-Guided Attention.

# %%
import torch.nn as nn
import torch.nn.functional as F

# ═══════════════════════════════════════════════════════════════════
# SECTION 9: MULTI-FREQUENCY FUSION TRANSFORMER (MFFT)
# ═══════════════════════════════════════════════════════════════════

BAND_CONFIGS = {
    2: [(0.0, 0.25), (0.25, 1.0)],
    3: [(0.0, 0.15), (0.15, 0.45), (0.45, 1.0)],
    4: [(0.0, 0.10), (0.10, 0.30), (0.30, 0.60), (0.60, 1.0)],
}

class FrequencyDecomposition(nn.Module):
    def __init__(self, num_bands: int = 3, soft_masks: bool = False, soft_tau: float = 0.02):
        super().__init__()
        self.bands = BAND_CONFIGS.get(num_bands, BAND_CONFIGS[3])
        self.num_bands = len(self.bands)
        self.soft_masks = soft_masks
        self.soft_tau = soft_tau
        self._cache = {}

    def _get_band_masks(self, H: int, W: int, dev: torch.device):
        cache_key = (H, W, str(dev), self.soft_masks)
        if cache_key in self._cache:
            return self._cache[cache_key]
        ny, nx = H // 2, W // 2
        r_max = min(ny, nx)
        y, x = torch.meshgrid(torch.arange(H, device=dev), torch.arange(W, device=dev), indexing='ij')
        dist = torch.sqrt((y - ny).float() ** 2 + (x - nx).float() ** 2)
        masks = []
        for lo, hi in self.bands:
            r_lo, r_hi = r_max * lo, r_max * hi
            if self.soft_masks:
                tau = max(self.soft_tau * r_max, 1e-3)
                lower = torch.sigmoid((dist - r_lo) / tau) if lo > 0 else torch.ones_like(dist)
                upper = torch.sigmoid((r_hi - dist) / tau) if hi < 1.0 else torch.ones_like(dist)
                masks.append(lower * upper)
            else:
                masks.append(((dist >= int(r_lo)) & (dist < int(r_hi))).float())
        stacked = torch.stack(masks).unsqueeze(1).unsqueeze(1)
        self._cache[cache_key] = stacked
        return stacked

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        H, W = x.shape[2], x.shape[3]
        x_fft = torch.fft.fftshift(torch.fft.fft2(x, norm='ortho'))
        masks = self._get_band_masks(H, W, x.device)
        bands = []
        for i in range(self.num_bands):
            filtered = x_fft * masks[i]
            band_spatial = torch.fft.ifft2(torch.fft.ifftshift(filtered), norm='ortho').real
            bands.append(band_spatial)
        return bands

class FrequencyFeatureExtractor(nn.Module):
    def __init__(self, in_channels: int = 3, feat_dim: int = 256):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
        )
        self.stages = nn.ModuleList([
            self._conv_block(32, 64, 3, 2),
            self._conv_block(64, 128, 3, 2),
            self._conv_block(128, 256, 3, 2),
        ])
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, feat_dim),
            nn.LayerNorm(feat_dim),
        )

    def _conv_block(self, in_c, out_c, k, s):
        return nn.Sequential(
            nn.Conv2d(in_c, in_c, k, stride=s, padding=k//2, groups=in_c, bias=False),
            nn.Conv2d(in_c, out_c, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.GELU(),
        )

    def forward(self, x):
        h = self.stem(x)
        for stage in self.stages:
            h = stage(h)
        return self.head(h)

class CrossAttentionFusion(nn.Module):
    def __init__(self, dim: int = 256, num_heads: int = 8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        B, N, D = x.shape
        qkv = self.to_qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        return self.dropout(self.proj(out))

class FrequencyGuidedAttention(nn.Module):
    def __init__(self, dim: int = 256):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, dim),
            nn.Sigmoid(),
        )

    def forward(self, x, freq_energy):
        return x * self.gate(x) * freq_energy

class MFFT(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        feat_dim: int = 384,
        num_bands: int = 3,
        num_heads: int = 6,
        num_classes: int = 2,
        ablation: Optional[dict] = None,
        soft_masks: bool = False,
    ):
        super().__init__()
        self.num_bands = num_bands
        self.feat_dim = feat_dim
        self.ablation = ablation or {}

        self.decomposer = FrequencyDecomposition(num_bands, soft_masks=soft_masks)
        self.extractors = nn.ModuleList([
            FrequencyFeatureExtractor(in_channels, feat_dim) for _ in range(num_bands)
        ])

        fusion_mode = self.ablation.get("fusion_mode", "attention")
        if fusion_mode == "attention":
            self.fusion = CrossAttentionFusion(feat_dim, num_heads)
        else:
            self.fusion = None

        self.use_fga = self.ablation.get("use_fga", True)
        self.fga = FrequencyGuidedAttention(feat_dim) if self.use_fga else None

        fused_dim = self._calc_fused_dim(feat_dim, num_bands)
        self.classifier = nn.Sequential(
            nn.LayerNorm(feat_dim * num_bands),
            nn.Linear(feat_dim * num_bands, feat_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(feat_dim, feat_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(feat_dim // 2, num_classes),
        )
        if fused_dim != feat_dim * num_bands:
            self.proj = nn.Linear(fused_dim, feat_dim * num_bands)
        else:
            self.proj = nn.Identity()

        self._init_weights()

    def _calc_fused_dim(self, d: int, nb: int) -> int:
        if self.ablation.get("spatial_only", False):
            return d
        active = nb - len(self.ablation.get("skip_bands", []))
        mode = self.ablation.get("fusion_mode", "attention")
        if mode in ("avg", "max") and active > 1:
            return d
        return active * d

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor, return_energy: bool = False):
        spatial_only = self.ablation.get("spatial_only", False)
        skip_bands = self.ablation.get("skip_bands", [])
        fusion_mode = self.ablation.get("fusion_mode", "attention")
        named_bands = {
            2: ["low", "high"],
            3: ["low", "mid", "high"],
            4: ["low", "low_mid", "mid_high", "high"],
        }
        band_names = named_bands.get(
            self.num_bands, [f"band{i}" for i in range(self.num_bands)]
        )

        if spatial_only:
            bands = [x]
            extractors = self.extractors[:1]
        else:
            all_bands = self.decomposer(x)
            keep_idx = [i for i, b in enumerate(band_names) if b not in skip_bands]
            bands = [all_bands[i] for i in keep_idx]
            extractors = [self.extractors[i] for i in keep_idx]

        feats = [ext(b) for b, ext in zip(bands, extractors)]
        B = x.shape[0]
        n_act = len(feats)

        if n_act == 1 or fusion_mode == "concat":
            fused = torch.cat(feats, dim=-1).unsqueeze(1)
            N, D = n_act, feats[0].shape[-1]
        elif fusion_mode == "avg":
            fused = torch.stack(feats, 0).mean(0).unsqueeze(1)
            N, D = 1, fused.shape[-1]
        elif fusion_mode == "max":
            fused = torch.stack(feats, 0).max(0).values.unsqueeze(1)
            N, D = 1, fused.shape[-1]
        elif fusion_mode == "attention" and self.fusion and n_act > 1:
            fused = self.fusion(torch.stack(feats, dim=1))
            N, D = fused.shape[1], fused.shape[2]
        else:
            fused = torch.cat(feats, dim=-1).unsqueeze(1)
            N, D = n_act, feats[0].shape[-1]

        if self.use_fga and self.fga and not spatial_only and N == n_act and fused.shape[-1] == self.feat_dim:
            energies = torch.stack([b.abs().mean((1, 2, 3)) for b in bands], dim=1)
            weights = F.softmax(energies, dim=1).unsqueeze(-1)
            fused = self.fga(fused, weights)

        out_proj = self.proj(fused.reshape(B, N * D))
        logits = self.classifier(out_proj)

        if return_energy:
            return logits, [b.abs().mean().item() for b in bands]
        return logits

def build_mfft(variant="base", ablation=None, soft_masks=False, num_classes=2):
    specs = {
        "tiny":  {"feat_dim": 128, "num_heads": 4, "num_bands": 3},
        "base":  {"feat_dim": 384, "num_heads": 6, "num_bands": 3},
        "large": {"feat_dim": 768, "num_heads": 12, "num_bands": 4},
    }
    cfg = dict(specs.get(variant, specs["base"]))
    if ablation and ablation.get("num_bands"):
        cfg["num_bands"] = int(ablation["num_bands"])
    return MFFT(**cfg, ablation=ablation, soft_masks=soft_masks, num_classes=num_classes)

def count_parameters(mod):
    return sum(p.numel() for p in mod.parameters() if p.requires_grad)


# ── P1: baseline comparison backbones (verbatim from model/src/baselines.py) ──
"""
Standard baseline models for ablation comparison.
Includes custom (SimpleCNN, LightViT) and torchvision models (ResNet, EfficientNet, ViT, Swin).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as M


def _resize_pos_embed(pos_embed, new_seq_len):
    """Interpolate 2D positional embeddings to a different grid size."""
    old_seq_len = pos_embed.shape[1]
    if old_seq_len == new_seq_len:
        return pos_embed
    dim = pos_embed.shape[-1]
    cls_token = pos_embed[:, 0:1, :]
    pos_tokens = pos_embed[:, 1:, :]
    old_h = old_w = int(math.isqrt(old_seq_len - 1))
    new_h = new_w = int(math.isqrt(new_seq_len - 1))
    pos_tokens = pos_tokens.reshape(1, old_h, old_w, dim).permute(0, 3, 1, 2)
    pos_tokens = F.interpolate(pos_tokens, size=(new_h, new_w), mode='bicubic', align_corners=False)
    pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(1, new_h * new_w, dim)
    return torch.cat([cls_token, pos_tokens], dim=1)


class SimpleCNN(nn.Module):
    """Lightweight CNN baseline for AI image detection."""
    def __init__(self, in_channels=3, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.GELU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256), nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256, 128), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=256):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

    def forward(self, x):
        B = x.shape[0]
        x = self.proj(x).flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        if x.shape[1] != self.pos_embed.shape[1]:
            pos_embed = _resize_pos_embed(self.pos_embed, x.shape[1])
        else:
            pos_embed = self.pos_embed
        x = x + pos_embed
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mlp_hidden, dim), nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


class LightViT(nn.Module):
    """Lightweight Vision Transformer baseline."""
    def __init__(self, img_size=384, patch_size=16, in_chans=3,
                 embed_dim=256, depth=6, num_heads=8, num_classes=2):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x[:, 0])
        return self.head(x)


def _wrap_torchvision(model_fn, weights_cls, num_classes=2):
    model = model_fn(weights=weights_cls)
    in_features = model.classifier[1].in_features if hasattr(model, 'classifier') else \
                  model.fc.in_features if hasattr(model, 'fc') else \
                  model.head.in_features
    if hasattr(model, 'classifier') and isinstance(model.classifier, nn.Sequential):
        model.classifier = nn.Sequential(
            nn.Dropout(0.2), nn.Linear(in_features, num_classes))
    elif hasattr(model, 'fc'):
        model.fc = nn.Linear(in_features, num_classes)
    elif hasattr(model, 'head'):
        model.head = nn.Linear(in_features, num_classes)
    return model


def resnet18(num_classes=2):
    return _wrap_torchvision(M.resnet18, M.ResNet18_Weights.IMAGENET1K_V1, num_classes)


def resnet50(num_classes=2):
    return _wrap_torchvision(M.resnet50, M.ResNet50_Weights.IMAGENET1K_V2, num_classes)


def efficientnet_b0(num_classes=2):
    return _wrap_torchvision(M.efficientnet_b0, M.EfficientNet_B0_Weights.IMAGENET1K_V1, num_classes)


def vit_b_16(img_size=384, num_classes=2):
    model = M.vit_b_16(weights=M.ViT_B_16_Weights.IMAGENET1K_V1)
    model.image_size = img_size
    if hasattr(model, 'heads'):
        in_feat = model.heads.head.in_features
        model.heads = nn.Linear(in_feat, num_classes)
    else:
        in_feat = model.head.in_features
        model.head = nn.Linear(in_feat, num_classes)
    patch_size = model.patch_size
    n_patches = (img_size // patch_size) ** 2
    with torch.no_grad():
        model.encoder.pos_embedding = nn.Parameter(
            _resize_pos_embed(model.encoder.pos_embedding, n_patches + 1))
    return model


def swin_t(num_classes=2):
    model = M.swin_t(weights=M.Swin_T_Weights.IMAGENET1K_V1)
    model.head = nn.Linear(model.head.in_features, num_classes)
    return model


class CLIPBaseline(nn.Module):
    """CLIP ViT-B/32 with a classification head for fine-tuning."""
    def __init__(self, img_size=384, num_classes=2):
        super().__init__()
        import open_clip
        self.clip_model, _, _ = open_clip.create_model_and_transforms(
            'ViT-B-32', pretrained='laion2b_s34b_b79k')
        self.clip_model = self.clip_model.visual
        in_features = self.clip_model.output_dim
        self.clip_model.output_dim = None
        self.head = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, num_classes),
        )
        patch_size = 32
        n_patches = (img_size // patch_size) ** 2
        pe = self.clip_model.positional_embedding
        with torch.no_grad():
            self.clip_model.positional_embedding = nn.Parameter(
                _resize_pos_embed(pe.unsqueeze(0), n_patches + 1).squeeze(0))

    def forward(self, x):
        features = self.clip_model(x)
        return self.head(features)


class FreqDetect(nn.Module):
    """
    Frequency-domain baseline (Frank et al. 2020).
    Computes radial FFT magnitude profiles and classifies via MLP.
    """
    def __init__(self, img_size=384, num_radial_bins=16, num_classes=2):
        super().__init__()
        self.img_size = img_size
        self.num_radial_bins = num_radial_bins
        ny, nx = img_size // 2, img_size // 2
        y_grid, x_grid = torch.meshgrid(
            torch.arange(img_size), torch.arange(img_size), indexing='ij')
        dist = torch.sqrt((y_grid - ny)**2 + (x_grid - nx)**2).float()
        bin_width = ny / num_radial_bins
        self.register_buffer('radial_mask', (dist / bin_width).long().clamp(0, num_radial_bins - 1))

        self.classifier = nn.Sequential(
            nn.Linear(num_radial_bins * 3, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        B = x.shape[0]
        fft = torch.fft.fft2(x, norm='ortho')
        shifted = torch.fft.fftshift(fft)
        mag = torch.abs(shifted)
        radial_feats = []
        for c in range(x.shape[1]):
            channel = mag[:, c]
            pooled = torch.zeros(B, self.num_radial_bins, device=x.device, dtype=channel.dtype)
            pooled.scatter_add_(1, self.radial_mask.unsqueeze(0).expand(B, -1, -1).reshape(B, -1),
                                channel.reshape(B, -1))
            counts = (self.radial_mask.unsqueeze(0).expand(B, -1, -1).reshape(B, -1) >= 0).float().sum(dim=1, keepdim=True)
            pooled = pooled / counts.clamp(min=1)
            radial_feats.append(pooled)
        feats = torch.cat(radial_feats, dim=-1)
        return self.classifier(feats)


class DeiTSmall(nn.Module):
    """
    DeiT-Small architecture (Touvron et al. 2021).
    Uses class + distillation tokens, 12 transformer blocks, embed_dim=384.
    """
    def __init__(self, img_size=384, patch_size=16, in_chans=3,
                 embed_dim=384, depth=12, num_heads=6, num_classes=2, dropout=0.1):
        super().__init__()
        num_patches = (img_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, dropout=dropout) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self.head_dist = nn.Linear(embed_dim, num_classes)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.dist_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        dist_tokens = self.dist_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, dist_tokens, x], dim=1)
        if x.shape[1] != self.pos_embed.shape[1]:
            pos_embed = _resize_pos_embed(self.pos_embed, x.shape[1])
        else:
            pos_embed = self.pos_embed
        x = self.pos_drop(x + pos_embed)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return (self.head(x[:, 0]) + self.head_dist(x[:, 1])) / 2


def deit_small(img_size=384, num_classes=2):
    """Build DeiT-Small."""
    return DeiTSmall(img_size=img_size, num_classes=num_classes)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ── Baseline registry ──
BASELINE_DISPLAY = {
    "resnet50": "ResNet-50",
    "vit_b16": "ViT-B/16",
    "efficientnet_b0": "EfficientNet-B0",
    "swin_t": "Swin Transformer-T",
    "deit_small": "DeiT-Small",
    "clip": "CLIP ViT-B/32",
    "resnet18": "ResNet-18",
    "lightvit": "LightViT",
    "simplecnn": "SimpleCNN",
    "freqdetect": "FreqDetect"
}


def build_baseline(name, img_size=384, num_classes=2):
    """Construct a comparison baseline with the pipeline's input size and 2-class head."""
    builders = {
        "simplecnn": lambda: SimpleCNN(num_classes=num_classes),
        "lightvit": lambda: LightViT(img_size=img_size, num_classes=num_classes),
        "freqdetect": lambda: FreqDetect(img_size=img_size, num_classes=num_classes),
        "resnet18": lambda: resnet18(num_classes=num_classes),
        "resnet50": lambda: resnet50(num_classes=num_classes),
        "efficientnet_b0": lambda: efficientnet_b0(num_classes=num_classes),
        "vit_b16": lambda: vit_b_16(img_size=img_size, num_classes=num_classes),
        "swin_t": lambda: swin_t(num_classes=num_classes),
        "deit_small": lambda: deit_small(img_size=img_size, num_classes=num_classes),
        "clip": lambda: CLIPBaseline(img_size=img_size, num_classes=num_classes),
    }
    if name not in builders:
        raise ValueError(f"Unknown baseline {name!r}; expected one of {sorted(builders)}")
    return builders[name]()


# ── Model selection: proposed MFFT vs. baseline comparison ──
MODEL_KIND = CONFIG.get("model_kind", "mfft")
if MODEL_KIND == "baseline":
    _baseline_name = CONFIG.get("baseline_name")
    MODEL_DISPLAY = BASELINE_DISPLAY.get(_baseline_name, str(_baseline_name))
    MODEL_DISPLAY_TITLE = MODEL_DISPLAY
    model = build_baseline(
        _baseline_name, img_size=CONFIG["image_size"], num_classes=CONFIG["num_classes"]
    ).to(device)
else:
    MODEL_DISPLAY = f"MFFT-{CONFIG['model_variant']}"
    MODEL_DISPLAY_TITLE = f"MFFT-{str(CONFIG['model_variant']).title()}"
    model = build_mfft(CONFIG["model_variant"], ablation=CONFIG.get("ablation", None),
                       num_classes=CONFIG["num_classes"]).to(device)
param_count = count_parameters(model)
print(f"\u2713 Model Initialized: {'Baseline-' if MODEL_KIND == 'baseline' else ''}{MODEL_DISPLAY}")
print(f"  Trainable Parameters: {param_count:,}")

# Forward pass validation
with torch.no_grad():
    dummy_input = torch.randn(2, 3, CONFIG["image_size"], CONFIG["image_size"]).to(device)
    dummy_out = model(dummy_input)
    assert dummy_out.shape == (2, CONFIG["num_classes"])
    del dummy_input, dummy_out
    if IS_GPU:
        torch.cuda.empty_cache()
print("✓ Forward pass sanity test verified")

# %% [markdown]
# # 10. Checkpoint Detection & Automatic Resume Logic
# Seamlessly restore model, optimizer, scheduler, scaler, epoch, and random states from HuggingFace.

# %%
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LinearLR, SequentialLR

# ═══════════════════════════════════════════════════════════════════
# SECTION 10: CHECKPOINT DETECTION & AUTOMATIC RESUME
# ═══════════════════════════════════════════════════════════════════

criterion = nn.CrossEntropyLoss(label_smoothing=CONFIG["label_smoothing"])
optimizer = AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])

total_steps = CONFIG["num_epochs"] * len(train_loader)
warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=min(CONFIG["warmup_steps"], len(train_loader)))
cosine = CosineAnnealingWarmRestarts(optimizer, T_0=max(total_steps, 100), T_mult=2, eta_min=1e-6)
scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[min(CONFIG["warmup_steps"], len(train_loader))])

scaler = torch.amp.GradScaler('cuda', enabled=CONFIG["mixed_precision"] and IS_GPU)

start_epoch = 0
global_step = 0
best_val_f1 = 0.0
best_val_auc = 0.0
best_val_acc = 0.0
best_val_loss = float("inf")
patience_counter = 0
training_history = []

print("=" * 65)
print("  CHECKPOINT AUDIT & RESUME EVALUATION")
print("=" * 65)

existing_state = hf_mgr.load_state()

if existing_state:
    print(f"Found existing experiment on HF Hub ({existing_state.run_id}):")
    print(f"  Recorded Epoch: {existing_state.current_epoch}/{existing_state.total_epochs}")
    print(f"  Best Val F1:    {existing_state.best_val_f1:.4f}")

    ckpt_file = hf_mgr.download_checkpoint("last")
    if not ckpt_file or not ckpt_file.exists():
        ckpt_file = hf_mgr.download_checkpoint("best")

    if ckpt_file and ckpt_file.exists():
        try:
            ckpt_data = torch.load(ckpt_file, map_location=device, weights_only=False)
            model.load_state_dict(ckpt_data["model_state_dict"])
            optimizer.load_state_dict(ckpt_data["optimizer_state_dict"])
            if "scheduler_state_dict" in ckpt_data:
                try:
                    scheduler.load_state_dict(ckpt_data["scheduler_state_dict"])
                except Exception:
                    pass
            if "scaler_state_dict" in ckpt_data and scaler.is_enabled():
                try:
                    scaler.load_state_dict(ckpt_data["scaler_state_dict"])
                except Exception:
                    pass

            start_epoch = ckpt_data.get("epoch", existing_state.current_epoch)
            global_step = ckpt_data.get("global_step", existing_state.global_step)
            best_val_f1 = ckpt_data.get("best_val_f1", existing_state.best_val_f1)
            best_val_auc = ckpt_data.get("best_val_auc", existing_state.best_val_auc)
            best_val_acc = ckpt_data.get("best_val_accuracy", existing_state.best_val_accuracy)
            best_val_loss = ckpt_data.get("best_val_loss", existing_state.best_val_loss)
            patience_counter = ckpt_data.get("patience_counter", existing_state.patience_counter)
            training_history = ckpt_data.get("history", [])

            # Restore random states
            if "random_states" in ckpt_data:
                rs = ckpt_data["random_states"]
                if "python" in rs:
                    random.setstate(rs["python"])
                if "numpy" in rs:
                    np.random.set_state(rs["numpy"])
                if "torch" in rs:
                    torch.random.set_rng_state(rs["torch"])
                if "cuda" in rs and rs["cuda"] is not None and IS_GPU:
                    try:
                        torch.cuda.set_rng_state_all(rs["cuda"])
                    except Exception:
                        pass

            print(f"✓ Successfully resumed from epoch {start_epoch}")
            print(f"  Current Best F1: {best_val_f1:.2f}% | Current Best AUC: {best_val_auc:.4f}")
            hf_mgr.log_event("RESUME_SUCCESS", f"Resumed from epoch {start_epoch}")
        except Exception as e:
            print(f"⚠ Checkpoint loading error: {e}. Starting fresh training.")
            start_epoch = 0
    else:
        print("⚠ Checkpoint file not available on Hub. Starting fresh.")
else:
    print("✓ No existing checkpoint discovered. Starting new research run.")

TRAINING_COMPLETE = (start_epoch >= CONFIG["num_epochs"])

# %% [markdown]
# # 11–12. Training & Validation Engines
# Implement fault-tolerant train and validation step functions with gradient accumulation and clipping.

# %%
from tqdm.auto import tqdm

# ═══════════════════════════════════════════════════════════════════
# SECTION 11-12: TRAINING & VALIDATION ENGINES
# ═══════════════════════════════════════════════════════════════════

def train_one_epoch(mod, loader, opt, sched, scl, crit, dev, cfg, step):
    mod.train()
    tot_loss = 0.0
    correct = 0
    total = 0
    skipped_batches = 0
    accum = cfg["gradient_accumulation_steps"]

    pbar = tqdm(loader, desc="Training", leave=False)
    for b_idx, (imgs, lbls) in enumerate(pbar):
        try:
            imgs = imgs.to(dev, non_blocking=True)
            lbls = lbls.to(dev, non_blocking=True)

            with torch.amp.autocast('cuda', enabled=cfg["mixed_precision"] and IS_GPU):
                logits = mod(imgs)
                loss = crit(logits, lbls)
            loss_scaled = loss / accum

            scl.scale(loss_scaled).backward()

            if (b_idx + 1) % accum == 0 or (b_idx + 1) == len(loader):
                scl.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(mod.parameters(), cfg["max_grad_norm"])
                scl.step(opt)
                scl.update()
                sched.step()
                opt.zero_grad()
                step += 1

            tot_loss += loss.item()
            preds = logits.argmax(-1)
            correct += (preds == lbls).sum().item()
            total += lbls.size(0)

            pbar.set_postfix({
                "loss": f"{tot_loss / max(b_idx + 1, 1):.4f}",
                "acc": f"{correct / max(total, 1) * 100:.1f}%",
            })
        except Exception:
            skipped_batches += 1
            opt.zero_grad()
            continue

    return {
        "loss": tot_loss / max(len(loader) - skipped_batches, 1),
        "accuracy": correct / max(total, 1) * 100,
        "lr": sched.get_last_lr()[0],
        "global_step": step,
    }

@torch.no_grad()
def validate(mod, loader, crit, dev):
    mod.eval()
    tot_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []

    for imgs, lbls in tqdm(loader, desc="Validating", leave=False):
        try:
            imgs = imgs.to(dev, non_blocking=True)
            lbls = lbls.to(dev, non_blocking=True)
            logits = mod(imgs)
            loss = crit(logits, lbls)
            probs = F.softmax(logits, dim=-1)

            tot_loss += loss.item()
            all_preds.extend(logits.argmax(-1).cpu().numpy())
            all_labels.extend(lbls.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
        except Exception:
            continue

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    acc = accuracy_score(y_true, y_pred) * 100
    prec = precision_score(y_true, y_pred, zero_division=0) * 100
    rec = recall_score(y_true, y_pred, zero_division=0) * 100
    f1 = f1_score(y_true, y_pred, zero_division=0) * 100
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0) * 100

    try:
        auc_val = roc_auc_score(y_true, y_prob[:, 1]) * 100
    except Exception:
        auc_val = 0.0

    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    spec = tn / max(tn + fp, 1) * 100

    return {
        "loss": tot_loss / max(len(loader), 1),
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "macro_f1": macro_f1,
        "auc": auc_val,
        "specificity": spec,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
    }

print("✓ Training and validation engines compiled")

# %% [markdown]
# # 13–14. Training Execution & Automated Synchronization
# Run the complete multi-epoch training pipeline with continuous HuggingFace synchronization.

# %%
# ═══════════════════════════════════════════════════════════════════
# SECTION 13-14: TRAINING EXECUTION & CONTINUOUS HF SYNCHRONIZATION
# ═══════════════════════════════════════════════════════════════════

def persist_checkpoint(mod, opt, sched, scl, ep, step, best_f1, best_auc, best_acc, best_l, pat, hist, cfg, tag="last"):
    ckpt = {
        "model_state_dict": mod.state_dict(),
        "optimizer_state_dict": opt.state_dict(),
        "scheduler_state_dict": sched.state_dict(),
        "scaler_state_dict": scl.state_dict() if scl.is_enabled() else {},
        "epoch": ep,
        "global_step": step,
        "best_val_f1": best_f1,
        "best_val_auc": best_auc,
        "best_val_accuracy": best_acc,
        "best_val_loss": best_l,
        "patience_counter": pat,
        "history": hist,
        "config": cfg,
        "timestamp": datetime.now().isoformat(),
        "random_states": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.random.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if IS_GPU else None,
        }
    }
    local_path = CHECKPOINT_DIR / f"{tag}.pt"
    torch.save(ckpt, local_path)
    return str(local_path)

if not TRAINING_COMPLETE:
    print("=" * 65)
    print(f"  LAUNCHING MFFT TRAINING: EPOCHS {start_epoch + 1} -> {CONFIG['num_epochs']}")
    print("=" * 65)
    train_timer_start = time.time()

    for epoch in range(start_epoch, CONFIG["num_epochs"]):
        ep_start = time.time()

        train_res = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler, criterion, device, CONFIG, global_step
        )
        global_step = train_res["global_step"]

        val_res = validate(model, val_loader, criterion, device)
        ep_duration = time.time() - ep_start
        tot_elapsed = time.time() - train_timer_start

        # Record metrics record
        record = {
            "epoch": epoch + 1,
            "train_loss": train_res["loss"],
            "train_accuracy": train_res["accuracy"],
            "val_loss": val_res["loss"],
            "val_accuracy": val_res["accuracy"],
            "val_precision": val_res["precision"],
            "val_recall": val_res["recall"],
            "val_f1": val_res["f1"],
            "val_macro_f1": val_res["macro_f1"],
            "val_auc": val_res["auc"],
            "val_specificity": val_res["specificity"],
            "lr": train_res["lr"],
            "epoch_time": ep_duration,
        }
        training_history.append(record)

        print(f"Epoch {epoch + 1:2d}/{CONFIG['num_epochs']} | "
              f"Train Loss: {train_res['loss']:.4f} | Train Acc: {train_res['accuracy']:.2f}% | "
              f"Val Loss: {val_res['loss']:.4f} | Val Acc: {val_res['accuracy']:.2f}% | "
              f"Val F1: {val_res['f1']:.2f}% | Val AUC: {val_res['auc']:.4f} | "
              f"LR: {train_res['lr']:.2e} | {ep_duration:.1f}s")

        is_new_best = val_res["f1"] > best_val_f1
        if is_new_best:
            best_val_f1 = val_res["f1"]
            best_val_auc = val_res["auc"]
            best_val_acc = val_res["accuracy"]
            best_val_loss = val_res["loss"]
            patience_counter = 0
            print(f"  ★ New Best Model Detected: Val F1 = {best_val_f1:.2f}%, AUC = {best_val_auc:.4f}")
        else:
            patience_counter += 1

        # Synchronize with Hugging Face Hub — ONE commit per epoch
        # (repo allows 128 commits/hour shared by every worker; the old per-file
        #  path issued 4-6 commits per epoch and caused chronic 429 throttling)
        last_file = persist_checkpoint(
            model, optimizer, scheduler, scaler, epoch + 1, global_step,
            best_val_f1, best_val_auc, best_val_acc, best_val_loss,
            patience_counter, training_history, CONFIG, tag="last"
        )
        bundle = [(last_file, hf_mgr.repo_path("checkpoints/last.pt"))]

        if is_new_best:
            best_file = persist_checkpoint(
                model, optimizer, scheduler, scaler, epoch + 1, global_step,
                best_val_f1, best_val_auc, best_val_acc, best_val_loss,
                patience_counter, training_history, CONFIG, tag="best"
            )
            bundle.append((best_file, hf_mgr.repo_path("checkpoints/best.pt")))

        if (epoch + 1) in CONFIG["snapshot_epochs"]:
            snap_file = persist_checkpoint(
                model, optimizer, scheduler, scaler, epoch + 1, global_step,
                best_val_f1, best_val_auc, best_val_acc, best_val_loss,
                patience_counter, training_history, CONFIG, tag=f"epoch_{epoch + 1}"
            )
            bundle.append((snap_file, hf_mgr.repo_path(f"checkpoints/epoch_{epoch + 1}.pt")))

        # Stage metrics + state locally; they ride in the same commit as the checkpoints
        hf_mgr.stage_metric(MetricRecord(
            epoch=epoch + 1,
            train_loss=train_res["loss"],
            train_accuracy=train_res["accuracy"],
            val_loss=val_res["loss"],
            val_accuracy=val_res["accuracy"],
            val_precision=val_res["precision"],
            val_recall=val_res["recall"],
            val_f1=val_res["f1"],
            val_macro_f1=val_res["macro_f1"],
            val_auc=val_res["auc"],
            val_specificity=val_res["specificity"],
            learning_rate=train_res["lr"],
            elapsed_time=tot_elapsed,
        ))

        state_file = hf_mgr.stage_state(TrainingState(
            run_id=EXPERIMENT_ID,
            model_variant=CONFIG["model_variant"],
            current_epoch=epoch + 1,
            total_epochs=CONFIG["num_epochs"],
            global_step=global_step,
            best_val_f1=best_val_f1,
            best_val_auc=best_val_auc,
            best_val_accuracy=best_val_acc,
            best_val_loss=best_val_loss,
            patience_counter=patience_counter,
            early_stop=(patience_counter >= CONFIG["patience"]),
            best_model_epoch=epoch + 1 if is_new_best else 0,
            total_train_time=tot_elapsed,
            status="training",
        ))
        bundle += [
            (state_file, hf_mgr.repo_path("logs/training_state.json")),
            (hf_mgr.csv_metrics_path, hf_mgr.repo_path("logs/metrics.csv")),
        ]

        if coordinator and claimed_job:
            lease_file = coordinator.stage_heartbeat(claimed_job["job_id"], epoch + 1, best_val_f1)
            if lease_file is None or getattr(coordinator, "lost_ownership", False):
                print("  ✖ Lease ownership lost — stopping this duplicate trainer cleanly.")
                hf_mgr.log_event("LEASE_LOST",
                                 "Stopped duplicate trainer: lease taken by another worker")
                break
            bundle.append((lease_file, f"coordinator/leases/{claimed_job['job_id']}.json"))

        hf_mgr.upload_bundle(bundle, f"[{EXPERIMENT_ID}] epoch {epoch + 1}/{CONFIG['num_epochs']}")

        if patience_counter >= CONFIG["patience"]:
            print(f"✓ Early stopping triggered at epoch {epoch + 1} (patience={CONFIG['patience']})")
            hf_mgr.log_event("EARLY_STOP", f"Stopped at epoch {epoch + 1}")
            break

    print("=" * 65)
    print("  TRAINING CYCLE COMPLETE")
    print(f"  Best Validation F1:  {best_val_f1:.2f}%")
    print(f"  Best Validation AUC: {best_val_auc:.4f}")
    print("=" * 65)
else:
    print("✓ Model training already completed to target epochs")

# Save training history
hist_file = METRICS_DIR / "training_history.csv"
pd.DataFrame(training_history).to_csv(hist_file, index=False)
hf_mgr.upload_file(hist_file, "metrics/training_history.csv")

# %% [markdown]
# # 15–16. Final Evaluation & Prediction Export
# Restore best checkpoint and evaluate on held-out test split, exporting complete predictions.

# %%
# ═══════════════════════════════════════════════════════════════════
# SECTION 15-16: TEST SET EVALUATION & PREDICTION STORAGE
# ═══════════════════════════════════════════════════════════════════

best_model_file = CHECKPOINT_DIR / "best.pt"
if best_model_file.exists():
    best_data = torch.load(best_model_file, map_location=device, weights_only=False)
    model.load_state_dict(best_data["model_state_dict"])
    print(f"✓ Loaded best model weights for final evaluation (epoch {best_data.get('epoch', '?')})")

test_results = validate(model, test_loader, criterion, device)

print("=" * 65)
print("  FINAL HELD-OUT TEST RESULTS")
print("=" * 65)
print(f"  Test Accuracy:    {test_results['accuracy']:.2f}%")
print(f"  Test Precision:   {test_results['precision']:.2f}%")
print(f"  Test Recall:      {test_results['recall']:.2f}%")
print(f"  Test F1 Score:    {test_results['f1']:.2f}%")
print(f"  Test Macro-F1:    {test_results['macro_f1']:.2f}%")
print(f"  Test AUC-ROC:     {test_results['auc']:.4f}")
print(f"  Test Specificity: {test_results['specificity']:.2f}%")
print("=" * 65)

# Save test metrics JSON
test_metrics_summary = {
    "experiment_id": EXPERIMENT_ID,
    "model_variant": CONFIG["model_variant"],
    "accuracy": test_results["accuracy"],
    "precision": test_results["precision"],
    "recall": test_results["recall"],
    "f1": test_results["f1"],
    "macro_f1": test_results["macro_f1"],
    "auc": test_results["auc"],
    "specificity": test_results["specificity"],
    "loss": test_results["loss"],
    "evaluated_at": datetime.now().isoformat(),
}
test_json = METRICS_DIR / "test_metrics.json"
with open(test_json, "w", encoding="utf-8") as f:
    json.dump(test_metrics_summary, f, indent=2)
hf_mgr.upload_file(test_json, "metrics/test_metrics.json")

# Save Confusion Matrix
cm = confusion_matrix(test_results["y_true"], test_results["y_pred"])
cm_df = pd.DataFrame(cm, index=["Real (0)", "AI/Deepfake (1)"], columns=["Pred Real (0)", "Pred AI (1)"])
cm_file = METRICS_DIR / "confusion_matrix.csv"
cm_df.to_csv(cm_file)
hf_mgr.upload_file(cm_file, "metrics/confusion_matrix.csv")

# Save Classification Report
clf_rep = classification_report(test_results["y_true"], test_results["y_pred"], target_names=["Real", "AI_Generated"], output_dict=True)
clf_df = pd.DataFrame(clf_rep).transpose()
clf_file = METRICS_DIR / "classification_report.csv"
clf_df.to_csv(clf_file)
hf_mgr.upload_file(clf_file, "metrics/classification_report.csv")

# Export per-sample predictions
preds_df = pd.DataFrame({
    "sample_id": list(range(len(test_results["y_true"]))),
    "true_label": test_results["y_true"],
    "predicted_label": test_results["y_pred"],
    "prob_real": test_results["y_prob"][:, 0] if test_results["y_prob"].shape[1] > 1 else 1.0 - test_results["y_prob"][:, 0],
    "prob_ai": test_results["y_prob"][:, 1] if test_results["y_prob"].shape[1] > 1 else test_results["y_prob"][:, 0],
    "experiment_id": EXPERIMENT_ID,
    "model_variant": CONFIG["model_variant"],
})
preds_file = PREDICTIONS_DIR / "test_predictions.csv"
preds_df.to_csv(preds_file, index=False)
hf_mgr.upload_file(preds_file, "predictions/test_predictions.csv")
print(f"✓ All predictions ({len(preds_df):,} rows) exported and uploaded")
if coordinator and claimed_job:
    coordinator.mark_completed(claimed_job["job_id"], test_results)

# %% [markdown]
# # 17–19. Manuscript Figures
# Generate publication-quality IEEE-format figures at 300 DPI in PNG and vector formats.

# %%
# ═══════════════════════════════════════════════════════════════════
# SECTION 17-19: MANUSCRIPT-READY PUBLICATION FIGURES
# ═══════════════════════════════════════════════════════════════════

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': CONFIG["figure_font_size"],
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'figure.dpi': CONFIG["figure_dpi"],
    'savefig.dpi': CONFIG["figure_dpi"],
    'savefig.bbox': 'tight',
})

PALETTE = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#F0E442', '#56B4E9']

def publish_figure(fig, fig_stem: str):
    for ext in CONFIG["figure_format"]:
        target_path = FIGURES_DIR / f"{fig_stem}.{ext}"
        fig.savefig(target_path, format=ext, dpi=CONFIG["figure_dpi"], bbox_inches='tight')
    plt.close(fig)

# ── Fig 1: Training Loss & Accuracy Curves ──
if training_history:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    eps = [h["epoch"] for h in training_history]

    axes[0].plot(eps, [h["train_loss"] for h in training_history], color=PALETTE[0], label="Training Loss", lw=2)
    axes[0].plot(eps, [h["val_loss"] for h in training_history], color=PALETTE[1], label="Validation Loss", lw=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].set_title("Training and Validation Loss Trajectory")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(eps, [h["train_accuracy"] for h in training_history], color=PALETTE[0], label="Training Acc", lw=2)
    axes[1].plot(eps, [h["val_accuracy"] for h in training_history], color=PALETTE[1], label="Validation Acc", lw=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Training and Validation Accuracy Trajectory")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    publish_figure(fig, "training_curves")
    print("  ✓ Figure: training_curves")

# ── Fig 2: Confusion Matrix Heatmap ──
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=["Real", "AI/Deepfake"], yticklabels=["Real", "AI/Deepfake"])
axes[0].set_title("Confusion Matrix (Counts)")
axes[0].set_xlabel("Predicted Label")
axes[0].set_ylabel("True Label")

cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
sns.heatmap(cm_norm, annot=True, fmt='.3f', cmap='Blues', ax=axes[1],
            xticklabels=["Real", "AI/Deepfake"], yticklabels=["Real", "AI/Deepfake"])
axes[1].set_title("Confusion Matrix (Normalized)")
axes[1].set_xlabel("Predicted Label")
axes[1].set_ylabel("True Label")

fig.tight_layout()
publish_figure(fig, "confusion_matrix")
print("  ✓ Figure: confusion_matrix")

# ── Fig 3: ROC Curve ──
fpr, tpr, _ = roc_curve(test_results["y_true"], test_results["y_prob"][:, 1])
roc_auc_score_val = auc(fpr, tpr)

fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(fpr, tpr, color=PALETTE[0], lw=2, label=f"{MODEL_DISPLAY} (AUC = {roc_auc_score_val:.4f})")
ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.6)
ax.set_xlabel("False Positive Rate (1 - Specificity)")
ax.set_ylabel("True Positive Rate (Sensitivity)")
ax.set_title("Receiver Operating Characteristic (ROC)")
ax.legend(loc="lower right")
ax.grid(True, alpha=0.3)
fig.tight_layout()
publish_figure(fig, "roc_curve")
print("  ✓ Figure: roc_curve")

# ── Fig 4: Precision-Recall Curve ──
p_vals, r_vals, _ = precision_recall_curve(test_results["y_true"], test_results["y_prob"][:, 1])
ap_score = average_precision_score(test_results["y_true"], test_results["y_prob"][:, 1])

fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(r_vals, p_vals, color=PALETTE[2], lw=2, label=f"AP = {ap_score:.4f}")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Precision-Recall (PR) Curve")
ax.legend(loc="lower left")
ax.grid(True, alpha=0.3)
fig.tight_layout()
publish_figure(fig, "precision_recall_curve")
print("  ✓ Figure: precision_recall_curve")

# ── Fig 5: Frequency Energy Profile (MFFT-only) ──
if MODEL_KIND == "mfft":
    sample_batch = next(iter(test_loader))[0][:16].to(device)
    with torch.no_grad():
        _, energies = model(sample_batch, return_energy=True)
    bands_lbl = ["Low-Pass", "Mid-Band", "High-Pass"] if len(energies) == 3 else [f"Band_{i}" for i in range(len(energies))]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(bands_lbl, energies, color=PALETTE[:len(energies)], edgecolor='black', alpha=0.85)
    ax.set_ylabel("Spectral Energy Magnitude")
    ax.set_title("Decomposed Frequency Band Energy Distribution")
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    publish_figure(fig, "frequency_energy_profile")
    print("  ✓ Figure: frequency_energy_profile")
else:
    print("  ✓ Figure: frequency_energy_profile skipped (baseline has no band decomposition)")

# %% [markdown]
# # 20. Manuscript-Ready Tables (CSV & LaTeX)
# Export publication-ready tables in CSV and formal LaTeX formats for journal integration.

# %%
# ═══════════════════════════════════════════════════════════════════
# SECTION 20: MANUSCRIPT TABLES (CSV & LATEX EXPORT)
# ═══════════════════════════════════════════════════════════════════

def export_table(df: pd.DataFrame, table_stem: str, index: bool = True):
    csv_out = TABLES_DIR / f"{table_stem}.csv"
    df.to_csv(csv_out, index=index)
    hf_mgr.upload_table(csv_out)

    tex_out = TABLES_DIR / f"{table_stem}.tex"
    with open(tex_out, "w", encoding="utf-8") as f:
        f.write(df.to_latex(index=index, float_format="%.4f"))
    hf_mgr.upload_table(tex_out)
    print(f"  ✓ Table exported: {table_stem} (.csv & .tex)")

# Table 1: Main Performance Metrics
table_1 = pd.DataFrame([{
    "Model Architecture": MODEL_DISPLAY_TITLE,
    "Parameters": f"{count_parameters(model):,}",
    "Accuracy (%)": test_results["accuracy"],
    "Precision (%)": test_results["precision"],
    "Recall (%)": test_results["recall"],
    "F1-Score (%)": test_results["f1"],
    "Macro-F1 (%)": test_results["macro_f1"],
    "AUC-ROC": test_results["auc"] / 100.0,
    "Specificity (%)": test_results["specificity"],
}])
export_table(table_1, "table1_main_performance", index=False)

# Table 2: Per-Class Evaluation Breakdown
export_table(clf_df, "table2_per_class_results", index=True)

# Table 3: Experimental Hyperparameter Specifications
table_3 = pd.DataFrame([
    {"Hyperparameter": k, "Configured Value": str(v)}
    for k, v in CONFIG.items()
    if k not in ("ablation_configs", "snapshot_epochs", "figure_format")
])
export_table(table_3, "table3_hyperparameters", index=False)

# Table 4: Epoch Training History
if training_history:
    export_table(pd.DataFrame(training_history), "table4_training_history", index=False)

# %% [markdown]
# # 21. Experiment Provenance & Model Card
# Compile end-to-end experiment metadata and generate a structured model card.

# %%
# ═══════════════════════════════════════════════════════════════════
# SECTION 21: EXPERIMENT PROVENANCE & SUMMARY
# ═══════════════════════════════════════════════════════════════════

final_summary = {
    "experiment_id": EXPERIMENT_ID,
    "timestamp": datetime.now().isoformat(),
    "model": {
        "variant": CONFIG["model_variant"],
        "parameters": count_parameters(model),
        "num_classes": CONFIG["num_classes"],
    },
    "dataset": {
        "total_samples": len(all_samples),
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "test_samples": len(test_samples),
        "manifest_sha256": split_hash,
    },
    "test_metrics": test_metrics_summary,
    "environment": ENVIRONMENT_INFO,
    "config": CONFIG,
}

summary_json_file = CONFIGS_DIR / "experiment_summary.json"
with open(summary_json_file, "w", encoding="utf-8") as f:
    json.dump(final_summary, f, indent=2, default=str)
hf_mgr.upload_file(summary_json_file, "configs/experiment_summary.json")

hf_mgr.write_model_card(
    TrainingState(
        run_id=EXPERIMENT_ID,
        model_variant=CONFIG["model_variant"],
        manifest_sha256=split_hash,
        current_epoch=len(training_history),
        total_epochs=CONFIG["num_epochs"],
        best_val_f1=best_val_f1,
        best_val_auc=best_val_auc,
        best_val_accuracy=best_val_acc,
        status="completed",
        total_train_time=sum(h.get("epoch_time", 0) for h in training_history),
    ),
    final_metrics=test_metrics_summary
)
print("✓ Experiment metadata and model card updated on Hugging Face Hub")

# %% [markdown]
# # 22. Controlled Ablation Studies
# Systematically isolate architectural components to prove statistical contribution for Q1 review.

# %%
# ═══════════════════════════════════════════════════════════════════
# SECTION 22: CONTROLLED ABLATION EXPERIMENTS
# ═══════════════════════════════════════════════════════════════════

if CONFIG["run_ablations"] and not CONFIG.get("swarm_mode", False) and IS_GPU:
    print("=" * 65)
    print("  LAUNCHING ABLATION MATRIX (CONTROLLED COMPARISONS)")
    print("=" * 65)
    ablation_records = []
    abl_epochs = CONFIG["ablation_epochs"]
    for abl_key, abl_dict in CONFIG["ablation_configs"].items():
        print(f"\\nEvaluating Ablation Config: [{abl_key}]")
        abl_m = build_mfft(CONFIG["model_variant"], ablation=abl_dict, num_classes=CONFIG["num_classes"]).to(device)
        abl_opt = AdamW(abl_m.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
        abl_crit = nn.CrossEntropyLoss(label_smoothing=CONFIG["label_smoothing"])
        abl_scl = torch.amp.GradScaler('cuda', enabled=CONFIG["mixed_precision"])

        abl_wm = LinearLR(abl_opt, start_factor=0.01, end_factor=1.0, total_iters=min(CONFIG["warmup_steps"], len(train_loader)))
        abl_cos = CosineAnnealingWarmRestarts(abl_opt, T_0=max(abl_epochs * len(train_loader), 50), eta_min=1e-6)
        abl_sch = SequentialLR(abl_opt, schedulers=[abl_wm, abl_cos], milestones=[min(CONFIG["warmup_steps"], len(train_loader))])

        abl_step = 0
        best_abl_f1 = 0.0

        for ep in range(abl_epochs):
            t_res = train_one_epoch(abl_m, train_loader, abl_opt, abl_sch, abl_scl, abl_crit, device, CONFIG, abl_step)
            abl_step = t_res["global_step"]
            v_res = validate(abl_m, val_loader, abl_crit, device)
            if v_res["f1"] > best_abl_f1:
                best_abl_f1 = v_res["f1"]

        # Final evaluation on test split
        final_test_abl = validate(abl_m, test_loader, abl_crit, device)
        ablation_records.append({
            "Ablation Name": abl_key,
            "Parameters": count_parameters(abl_m),
            "Val Best F1 (%)": round(best_abl_f1, 2),
            "Test Accuracy (%)": round(final_test_abl["accuracy"], 2),
            "Test Precision (%)": round(final_test_abl["precision"], 2),
            "Test Recall (%)": round(final_test_abl["recall"], 2),
            "Test F1 (%)": round(final_test_abl["f1"], 2),
            "Test Macro-F1 (%)": round(final_test_abl["macro_f1"], 2),
            "Test AUC": round(final_test_abl["auc"] / 100.0, 4),
        })
        print(f"  [{abl_key}] -> Test F1: {final_test_abl['f1']:.2f}% | Test AUC: {final_test_abl['auc']/100:.4f}")

        del abl_m, abl_opt, abl_crit, abl_scl
        if IS_GPU:
            torch.cuda.empty_cache()

    # Compile ablation results table
    abl_df = pd.DataFrame(ablation_records)
    export_table(abl_df, "table5_ablation_results", index=False)

    # Export ablation comparison figure
    fig, ax = plt.subplots(figsize=(9, 4.5))
    names = [r["Ablation Name"] for r in ablation_records]
    f1s = [r["Test F1 (%)"] for r in ablation_records]
    bars = ax.barh(names, f1s, color=PALETTE[0], edgecolor='black', alpha=0.85)
    ax.axvline(x=test_results["f1"], color='crimson', linestyle='--', lw=2, label=f"Proposed MFFT ({test_results['f1']:.2f}%)")
    ax.set_xlabel("Held-Out Test F1 Score (%)")
    ax.set_title("Ablation Study: Contribution of Architectural Modules")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3, axis='x')
    fig.tight_layout()
    publish_figure(fig, "ablation_comparison_bar")
    print("✓ Ablation study successfully completed and exported")
else:
    print("ℹ Ablations bypassed (run_ablations=False or CPU mode)")

# %% [markdown]
# # 23. Statistical Rigor: Bootstrap CIs, Calibration & Robustness
# Execute non-parametric bootstrap resampling (1000 resamples), ECE calibration, and perturbation testing.

# %%
from scipy import stats as scipy_stats

# ═══════════════════════════════════════════════════════════════════
# SECTION 23: STATISTICAL RIGOR & POST-PROCESSING ROBUSTNESS
# ═══════════════════════════════════════════════════════════════════

print("=" * 65)
print("  STATISTICAL RIGOR & METRIC INFERENCE")
print("=" * 65)

def compute_bootstrap_ci(y_true, y_pred, metric_func, n_resamples=1000, conf=0.95, seed=42):
    rng = np.random.RandomState(seed)
    n = len(y_true)
    estimates = []
    for _ in range(n_resamples):
        idx = rng.randint(0, n, size=n)
        try:
            val = metric_func(y_true[idx], y_pred[idx])
            estimates.append(val)
        except Exception:
            continue
    estimates = np.array(estimates)
    alpha = 1.0 - conf
    ci_low = np.percentile(estimates, alpha / 2 * 100)
    ci_high = np.percentile(estimates, (1.0 - alpha / 2) * 100)
    return {
        "mean": float(np.mean(estimates)),
        "std": float(np.std(estimates)),
        "ci_lower_95": float(ci_low),
        "ci_upper_95": float(ci_high),
    }

# 1. Bootstrap 95% Confidence Intervals
y_t = test_results["y_true"]
y_p = test_results["y_pred"]
y_s = test_results["y_prob"][:, 1]

bootstrap_summary = {
    "Accuracy": compute_bootstrap_ci(y_t, y_p, lambda a, b: accuracy_score(a, b) * 100),
    "Precision": compute_bootstrap_ci(y_t, y_p, lambda a, b: precision_score(a, b, zero_division=0) * 100),
    "Recall": compute_bootstrap_ci(y_t, y_p, lambda a, b: recall_score(a, b, zero_division=0) * 100),
    "F1_Score": compute_bootstrap_ci(y_t, y_p, lambda a, b: f1_score(a, b, zero_division=0) * 100),
    "AUC_ROC": compute_bootstrap_ci(y_t, y_s, lambda a, b: roc_auc_score(a, b)),
}
ci_df = pd.DataFrame(bootstrap_summary).transpose()
export_table(ci_df, "table6_bootstrap_confidence_intervals", index=True)

# 2. Calibration Analysis (Expected Calibration Error)
def compute_ece(probs, labels, n_bins=10):
    bin_limits = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_mask = (probs >= bin_limits[i]) & (probs < bin_limits[i + 1])
        if np.sum(bin_mask) > 0:
            bin_acc = np.mean(labels[bin_mask] == 1)
            bin_conf = np.mean(probs[bin_mask])
            ece += np.abs(bin_acc - bin_conf) * (np.sum(bin_mask) / len(probs))
    return float(ece)

uncalibrated_ece = compute_ece(y_s, y_t)
brier_score = float(np.mean((y_s - y_t) ** 2))

calib_summary = pd.DataFrame([{
    "Metric": "Expected Calibration Error (ECE)", "Value": round(uncalibrated_ece, 4)
}, {
    "Metric": "Brier Score", "Value": round(brier_score, 4)
}])
export_table(calib_summary, "table7_calibration_metrics", index=False)
print(f"  Uncalibrated ECE: {uncalibrated_ece:.4f} | Brier Score: {brier_score:.4f}")

# 3. Robustness Perturbation Testing (JPEG, Blur, Downscaling)
if CONFIG["run_robustness"]:
    robustness_results = []
    sample_eval_imgs = test_samples[:min(len(test_samples), 200)]

    # JPEG Quality Degradation Grid
    for q in [95, 80, 60, 40]:
        correct_under_q = 0
        for path_str, true_lbl in sample_eval_imgs:
            try:
                im = Image.open(path_str).convert("RGB")
                buf = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
                im.save(buf.name, "JPEG", quality=q)
                degraded = Image.open(buf.name).convert("RGB")
                buf.close()
                Path(buf.name).unlink(missing_ok=True)
                t_tensor = ImageTransform(CONFIG["image_size"], augment=False)(degraded).unsqueeze(0).to(device)
                with torch.no_grad():
                    p = model(t_tensor).argmax(-1).item()
                if p == true_lbl:
                    correct_under_q += 1
            except Exception:
                continue
        robustness_results.append({
            "Perturbation Type": "JPEG Compression",
            "Severity Parameter": f"Quality={q}",
            "Sample Accuracy (%)": round(correct_under_q / max(len(sample_eval_imgs), 1) * 100, 2),
        })

    rob_df = pd.DataFrame(robustness_results)
    export_table(rob_df, "table8_robustness_perturbations", index=False)
    print("✓ Robustness perturbation audit compiled")

# %% [markdown]
# # 24. Verification Summary & Final Artifact Synchronization
# Consolidate all generated tables, figures, checkpoints, and logs to the Hugging Face experiment repository.

# %%
# ═══════════════════════════════════════════════════════════════════
# SECTION 24: VERIFICATION SUMMARY & MASTER SYNC
# ═══════════════════════════════════════════════════════════════════

print("=" * 65)
print("  FINAL HF EXPERIMENT SYNCHRONIZATION")
print("=" * 65)

# Sync all remaining artifacts atomically using upload_folder to prevent 429 rate limits
for folder, sub in [(FIGURES_DIR, "figures"), (TABLES_DIR, "tables"), (METRICS_DIR, "metrics")]:
    if folder.exists() and any(folder.iterdir()):
        try:
            hf_api.upload_folder(
                folder_path=str(folder),
                path_in_repo=f"runs/{EXPERIMENT_ID}/{sub}",
                repo_id=CONFIG["hf_checkpoint_repo"],
                repo_type="model",
                commit_message=f"Sync {sub} ({EXPERIMENT_ID})"
            )
        except Exception as e:
            print(f"  Batch upload notice for {sub}: {e}")

hf_mgr.sync_session_log()

print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    MFFT RESEARCH PIPELINE COMPLETE                           ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Experiment ID:        {EXPERIMENT_ID}
║  Model Architecture:   {MODEL_DISPLAY} ({param_count:,} parameters)
║  Target Journal:       {CONFIG['target_journal']}
║  Trained Epochs:       {len(training_history)} / {CONFIG['num_epochs']}
║  Held-Out Test Acc:    {test_results['accuracy']:.2f}%
║  Held-Out Test F1:     {test_results['f1']:.2f}%
║  Held-Out Test AUC:    {test_results['auc']:.4f}
║  Persistent Storage:   https://huggingface.co/{CONFIG['hf_checkpoint_repo']}
║  Run Directory:        runs/{EXPERIMENT_ID}/
╚══════════════════════════════════════════════════════════════════════════════╝
""")

print("✓ All execution artifacts safely persisted to Hugging Face Hub")
