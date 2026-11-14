"""Robustness evaluation: accuracy/AUC under JPEG recompression,
downscaling, and Gaussian blur -- the degradation curves promised in
the paper's full-scale protocol (Section VIII, item 3).

Usage (from model/):
    from src.robustness import run_robustness_suite
    df = run_robustness_suite(model, test_dataset, device="cuda",
                              out_csv="robustness.csv")

`test_dataset` must be an AIDetectionDataset (or anything exposing
`.samples` as a list of (path, label)); perturbations are applied to the
raw image before the standard eval transform so the pipeline matches
deployment conditions.
"""
import io
import csv
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
from sklearn.metrics import accuracy_score, roc_auc_score

from .dataset import ImageTransform


# ------------------------------------------------------------------
# Perturbations (applied to PIL images, before the eval transform)
# ------------------------------------------------------------------
def jpeg_compress(img: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def downscale(img: Image.Image, factor: float) -> Image.Image:
    """Downscale by `factor` then upscale back (bilinear), simulating
    thumbnail/re-post pipelines."""
    w, h = img.size
    small = img.resize((max(1, int(w * factor)), max(1, int(h * factor))),
                       Image.BILINEAR)
    return small.resize((w, h), Image.BILINEAR)


def gaussian_blur(img: Image.Image, sigma: float) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(radius=sigma))


DEFAULT_GRID = (
    [("clean", "none", None, lambda im: im)]
    + [("jpeg", f"q{q}", q, (lambda q_: lambda im: jpeg_compress(im, q_))(q))
       for q in (95, 80, 60, 40, 30)]
    + [("downscale", f"x{f}", f, (lambda f_: lambda im: downscale(im, f_))(f))
       for f in (0.75, 0.5, 0.25)]
    + [("blur", f"s{s}", s, (lambda s_: lambda im: gaussian_blur(im, s_))(s))
       for s in (0.5, 1.0, 2.0)]
)


@torch.no_grad()
def _evaluate(model, samples, perturb, transform, device, batch_size=32,
              max_samples: Optional[int] = None):
    model.eval()
    if max_samples:
        samples = samples[:max_samples]

    y_true, y_score = [], []
    batch, batch_labels = [], []

    def _flush():
        nonlocal batch, batch_labels
        if not batch:
            return
        x = torch.stack(batch).to(device)
        logits = model(x)
        probs = F.softmax(logits, dim=-1)[:, 1]
        y_score.extend(probs.cpu().tolist())
        y_true.extend(batch_labels)
        batch, batch_labels = [], []

    for path, label in samples:
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            continue
        batch.append(transform(perturb(img)))
        batch_labels.append(label)
        if len(batch) >= batch_size:
            _flush()
    _flush()

    y_true = np.array(y_true)
    y_score = np.array(y_score)
    y_pred = (y_score >= 0.5).astype(int)
    return {
        "n": len(y_true),
        "accuracy": float(accuracy_score(y_true, y_pred)) * 100,
        "auc": float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else float("nan"),
    }


def run_robustness_suite(
    model,
    test_dataset,
    device: str = "cuda",
    size: int = 384,
    batch_size: int = 32,
    max_samples: Optional[int] = None,
    grid=DEFAULT_GRID,
    out_csv: Optional[str] = None,
) -> List[dict]:
    """Evaluate `model` on `test_dataset` under every perturbation in the
    grid. Returns a list of row dicts; optionally writes them to CSV."""
    transform = ImageTransform(size=size, augment=False)
    rows = []
    for family, level_name, level, perturb in grid:
        metrics = _evaluate(model, list(test_dataset.samples), perturb,
                            transform, device, batch_size, max_samples)
        row = {"perturbation": family, "level": level_name,
               "param": level, **metrics}
        rows.append(row)
        print(f"{family:>10} {level_name:>6}: acc={metrics['accuracy']:.2f}%  "
              f"auc={metrics['auc']:.4f}  (n={metrics['n']})")

    if out_csv:
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved robustness table to {out_csv}")
    return rows
