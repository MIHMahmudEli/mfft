"""Leave-one-generator-out (LOGO) evaluation -- the cross-generator
generalization protocol promised in the paper (Section VIII, item 1),
following the GenImage benchmark design.

Two entry points:

1. make_logo_manifests(...)  -- writes train/test manifest CSVs for one
   held-out generator: the train manifest contains all real images plus
   fakes from every OTHER generator; the test manifest contains the
   held-out generator's fakes plus an equal-size random sample of real
   images never placed in that train manifest.

2. evaluate_per_generator(...) -- given a trained model, reports accuracy
   per generator family on a test manifest (used both for the LOGO test
   sets and for the per-generator breakdown table).

Usage (from model/):
    from src.logo_eval import make_logo_manifests, evaluate_per_generator, FAKE_GENERATORS
    for gen in FAKE_GENERATORS:
        make_logo_manifests("dataset/metadata/clean_metadata.csv", gen,
                            out_dir="dataset/metadata/logo")
"""
import csv
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import accuracy_score, roc_auc_score

from .dataset import ImageTransform

# Generator values as they appear in clean_metadata.csv's `generator` column.
FAKE_GENERATORS = [
    "BigGAN", "Glide", "Stable Diffusion", "DALL-E3", "Midjourney",
    "Celeb-DF", "FaceForensics", "DFDC", "unknown",
]


def _load_manifest(manifest_csv: str) -> pd.DataFrame:
    required = {"filename", "label", "generator"}
    df = pd.read_csv(manifest_csv, dtype=str, low_memory=False,
                     usecols=lambda c: c in required)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Manifest {manifest_csv} missing columns: {missing}")
    return df


def make_logo_manifests(
    manifest_csv: str,
    holdout_generator: str,
    out_dir: str,
    real_test_size: Optional[int] = None,
    seed: int = 42,
) -> Dict[str, str]:
    """Write LOGO train/test manifests for one held-out generator.

    Returns {"train": path, "test": path}.
    """
    df = _load_manifest(manifest_csv)
    real = df[df["label"] == "real"]
    fake = df[df["label"] != "real"]

    held = fake[fake["generator"] == holdout_generator]
    if len(held) == 0:
        raise ValueError(
            f"No images for generator '{holdout_generator}'. "
            f"Available: {sorted(fake['generator'].dropna().unique())}"
        )
    other_fake = fake[fake["generator"] != holdout_generator]

    rng = np.random.default_rng(seed)
    n_real_test = real_test_size or min(len(held), len(real) // 10)
    real_test_idx = rng.choice(real.index, size=n_real_test, replace=False)
    real_test = real.loc[real_test_idx]
    real_train = real.drop(index=real_test_idx)

    train_df = pd.concat([real_train, other_fake], ignore_index=True)
    test_df = pd.concat([real_test, held], ignore_index=True)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    slug = holdout_generator.replace(" ", "_").replace("-", "_").lower()
    paths = {
        "train": str(out / f"logo_train_wo_{slug}.csv"),
        "test": str(out / f"logo_test_{slug}.csv"),
    }
    train_df.to_csv(paths["train"], index=False)
    test_df.to_csv(paths["test"], index=False)
    print(f"[LOGO {holdout_generator}] train={len(train_df)} "
          f"(real {len(real_train)}, fake {len(other_fake)}), "
          f"test={len(test_df)} (real {len(real_test)}, fake {len(held)})")
    return paths


@torch.no_grad()
def evaluate_per_generator(
    model,
    manifest_csv: str,
    images_root: str,
    device: str = "cuda",
    size: int = 384,
    batch_size: int = 32,
    max_per_generator: Optional[int] = None,
    out_csv: Optional[str] = None,
) -> List[dict]:
    """Per-generator accuracy (and overall AUC) on a manifest."""
    model.eval()
    df = _load_manifest(manifest_csv)
    transform = ImageTransform(size=size, augment=False)
    root = Path(images_root)

    df = df.copy()
    df["group"] = df.apply(
        lambda r: "real" if r["label"] == "real" else str(r["generator"]), axis=1)

    rows = []
    all_true, all_score = [], []
    for group, gdf in df.groupby("group"):
        if max_per_generator and len(gdf) > max_per_generator:
            gdf = gdf.sample(max_per_generator, random_state=42)

        y_true, y_score = [], []
        batch, labels = [], []

        def _flush():
            nonlocal batch, labels
            if not batch:
                return
            x = torch.stack(batch).to(device)
            probs = F.softmax(model(x), dim=-1)[:, 1]
            y_score.extend(probs.cpu().tolist())
            y_true.extend(labels)
            batch, labels = [], []

        for _, r in gdf.iterrows():
            p = root / str(r["filename"]).replace("\\", "/")
            if not p.exists():
                continue
            try:
                img = Image.open(p).convert("RGB")
            except Exception:
                continue
            batch.append(transform(img))
            labels.append(0 if r["label"] == "real" else 1)
            if len(batch) >= batch_size:
                _flush()
        _flush()

        if not y_true:
            continue
        y_true_a = np.array(y_true)
        y_pred = (np.array(y_score) >= 0.5).astype(int)
        acc = accuracy_score(y_true_a, y_pred) * 100
        rows.append({"generator": group, "n": len(y_true_a),
                     "accuracy": round(float(acc), 2)})
        all_true.extend(y_true)
        all_score.extend(y_score)
        print(f"{group:>20}: acc={acc:.2f}%  (n={len(y_true_a)})")

    if len(np.unique(all_true)) > 1:
        overall_auc = roc_auc_score(all_true, all_score)
        rows.append({"generator": "OVERALL_AUC", "n": len(all_true),
                     "accuracy": round(float(overall_auc), 4)})
        print(f"{'overall AUC':>20}: {overall_auc:.4f}")

    if out_csv:
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["generator", "n", "accuracy"])
            w.writeheader()
            w.writerows(rows)
        print(f"Saved per-generator table to {out_csv}")
    return rows
