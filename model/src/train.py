import os
import sys
import json
import time
import math
import random
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LinearLR, SequentialLR

from .model import build_mfft, count_parameters, MFFTWithExplainability
from .dataset import create_dataloaders, AIDetectionDataset
from .config import Config


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Trainer:
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")

        set_seed(config.training.seed)

        self.model = build_mfft(config.training.model_variant)
        self.model = self.model.to(self.device)
        print(f"Model parameters: {count_parameters(self.model):,}")

        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=config.training.label_smoothing
        )

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.training.lr,
            weight_decay=config.training.weight_decay,
        )

        self._build_scheduler()
        self.scaler = GradScaler(enabled=config.training.mixed_precision and torch.cuda.is_available())

        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.global_step = 0
        self.best_val_acc = 0.0
        self.best_val_loss = float("inf")

    def _build_scheduler(self):
        cfg = self.config.training
        warmup = LinearLR(
            self.optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=cfg.warmup_steps,
        )
        cosine = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=cfg.epochs * 100,
            T_mult=2,
            eta_min=1e-6,
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup, cosine],
            milestones=[cfg.warmup_steps],
        )

    def train_epoch(self, train_loader) -> dict:
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        skipped = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            try:
                images = images.to(self.device)
                labels = labels.to(self.device)

                with autocast(enabled=self.config.training.mixed_precision and torch.cuda.is_available()):
                    logits = self.model(images)
                    loss = self.criterion(logits, labels)

                loss = loss / self.config.training.gradient_accumulation_steps

                self.scaler.scale(loss).backward()

                if (batch_idx + 1) % self.config.training.gradient_accumulation_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.training.max_grad_norm,
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1

                total_loss += loss.item() * self.config.training.gradient_accumulation_steps
                preds = logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
            except Exception as e:
                skipped += 1
                self.optimizer.zero_grad()
                if skipped <= 3:
                    print(f"  Warning: skipping bad batch {batch_idx}: {e}")

        if skipped > 0:
            print(f"  Skipped {skipped} bad batches this epoch")

        return {
            "loss": total_loss / max(len(train_loader) - skipped, 1),
            "acc": correct / max(total, 1) * 100,
            "lr": self.scheduler.get_last_lr()[0],
        }

    @torch.no_grad()
    def validate(self, val_loader) -> dict:
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        all_probs = []

        for images, labels in val_loader:
            try:
                images = images.to(self.device)
                labels = labels.to(self.device)

                logits = self.model(images)
                loss = self.criterion(logits, labels)

                probs = F.softmax(logits, dim=-1)
                preds = logits.argmax(dim=-1)

                total_loss += loss.item()
                correct += (preds == labels).sum().item()
                total += labels.size(0)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
            except Exception as e:
                print(f"  Warning: skipping bad val batch: {e}")
                continue

        accuracy = correct / total * 100

        tn = sum(1 for p, l in zip(all_preds, all_labels) if p == 0 and l == 0)
        fp = sum(1 for p, l in zip(all_preds, all_labels) if p == 1 and l == 0)
        fn = sum(1 for p, l in zip(all_preds, all_labels) if p == 0 and l == 1)
        tp = sum(1 for p, l in zip(all_preds, all_labels) if p == 1 and l == 1)

        precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        specificity = tn / (tn + fp) * 100 if (tn + fp) > 0 else 0

        return {
            "loss": total_loss / len(val_loader),
            "acc": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "specificity": specificity,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        }

    def save_checkpoint(self, metrics: dict, is_best: bool = False):
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config.to_dict(),
            "global_step": self.global_step,
            "metrics": metrics,
            "timestamp": datetime.now().isoformat(),
        }

        path = self.output_dir / f"step_{self.global_step}.pt"
        torch.save(checkpoint, path)
        print(f"  Checkpoint saved: {path}")

        if is_best:
            best_path = self.output_dir / "best.pt"
            torch.save(checkpoint, best_path)
            print(f"  Best model saved: {best_path}")

        old_checkpoints = sorted(self.output_dir.glob("step_*.pt"))
        while len(old_checkpoints) > self.config.training.save_top_k:
            old_checkpoints[0].unlink()
            old_checkpoints = old_checkpoints[1:]

        with open(self.output_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

    def train(self, train_loader, val_loader):
        print(f"\n{'='*60}")
        print(f"Training MFFT-{self.config.training.model_variant}")
        print(f"{'='*60}")
        print(f"Epochs: {self.config.training.epochs}")
        print(f"Batch size: {self.config.training.batch_size}")
        print(f"Learning rate: {self.config.training.lr}")
        print(f"Image size: {self.config.training.image_size}")
        print(f"{'='*60}\n")

        start_time = time.time()

        for epoch in range(1, self.config.training.epochs + 1):
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.validate(val_loader)

            elapsed = time.time() - start_time
            print(f"Epoch {epoch:3d}/{self.config.training.epochs} | "
                  f"Train Loss: {train_metrics['loss']:.4f} | "
                  f"Train Acc: {train_metrics['acc']:.2f}% | "
                  f"Val Loss: {val_metrics['loss']:.4f} | "
                  f"Val Acc: {val_metrics['acc']:.2f}% | "
                  f"F1: {val_metrics['f1']:.2f} | "
                  f"LR: {train_metrics['lr']:.2e} | "
                  f"Time: {elapsed:.0f}s")

            metrics = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_acc": train_metrics["acc"],
                "val_loss": val_metrics["loss"],
                "val_acc": val_metrics["acc"],
                "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"],
                "val_f1": val_metrics["f1"],
                "val_specificity": val_metrics["specificity"],
                "confusion_matrix": {
                    "tp": val_metrics["tp"],
                    "tn": val_metrics["tn"],
                    "fp": val_metrics["fp"],
                    "fn": val_metrics["fn"],
                },
                "learning_rate": train_metrics["lr"],
                "elapsed_seconds": elapsed,
            }

            is_best = val_metrics["acc"] > self.best_val_acc
            if is_best:
                self.best_val_acc = val_metrics["acc"]
                self.best_val_loss = val_metrics["loss"]

            if epoch % 5 == 0 or is_best or epoch == self.config.training.epochs:
                self.save_checkpoint(metrics, is_best=is_best)

        print(f"\n{'='*60}")
        print(f"Training complete! Best val acc: {self.best_val_acc:.2f}%")
        print(f"Total time: {elapsed:.0f}s")
        print(f"{'='*60}")

        return metrics


def main():
    config = Config()
    config.output_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "checkpoints"
    )
    os.makedirs(config.output_dir, exist_ok=True)

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    metadata_paths = cfg.dataset.metadata_paths
    existing_metadata = [p for p in metadata_paths if os.path.exists(p)]
    print(f"Found {len(existing_metadata)} metadata files: {existing_metadata}")

    train_loader, val_loader = create_dataloaders(
        root_dir=root,
        metadata_paths=existing_metadata,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        size=config.training.image_size,
        val_split=config.dataset.val_split,
        undersample=config.dataset.undersample,
    )

    trainer = Trainer(config)
    trainer.train(train_loader, val_loader)


if __name__ == "__main__":
    main()
