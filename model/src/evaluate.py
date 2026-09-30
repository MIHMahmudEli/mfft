import torch
import torch.nn.functional as F
import numpy as np
import json
from pathlib import Path
from typing import List, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, auc,
)
from .model import build_mfft, MFFTWithExplainability
from .dataset import AIDetectionDataset, ImageTransform
from torch.utils.data import DataLoader
import pandas as pd


class Evaluator:
    def __init__(self, checkpoint_path: str, variant: str = "base"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")

        self.model = build_mfft(variant).to(self.device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        print(f"Loaded checkpoint: {checkpoint_path}")
        print(f"  Global step: {checkpoint.get('global_step', 'N/A')}")
        print(f"  Best val acc: {checkpoint.get('metrics', {}).get('val_acc', 'N/A')}")

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> dict:
        all_preds = []
        all_labels = []
        all_probs = []
        all_logits = []

        for images, labels in loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            logits = self.model(images)
            probs = F.softmax(logits, dim=-1)

            all_preds.extend(logits.argmax(dim=-1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_logits.extend(logits.cpu().numpy())

        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)
        y_prob = np.array(all_probs)[:, 1]

        cm = confusion_matrix(y_true, y_pred)

        results = {
            "accuracy": float(accuracy_score(y_true, y_pred)) * 100,
            "precision": float(precision_score(y_true, y_pred)) * 100,
            "recall": float(recall_score(y_true, y_pred)) * 100,
            "f1": float(f1_score(y_true, y_pred)) * 100,
            "specificity": float(cm[0, 0] / (cm[0, 0] + cm[0, 1])) * 100,
            "auc_roc": float(roc_auc_score(y_true, y_prob)) * 100,
            "confusion_matrix": {
                "tn": int(cm[0, 0]), "fp": int(cm[0, 1]),
                "fn": int(cm[1, 0]), "tp": int(cm[1, 1]),
            },
            "total_samples": len(y_true),
        }

        return results

    @torch.no_grad()
    def evaluate_by_category(
        self, loader: DataLoader, categories: List[str]
    ) -> dict:
        by_category = {}
        for cat in set(categories):
            by_category[cat] = {"preds": [], "labels": []}

        for i, (images, labels) in enumerate(loader):
            images = images.to(self.device)
            logits = self.model(images)
            preds = logits.argmax(dim=-1)

            batch_cats = categories[i * loader.batch_size : (i + 1) * loader.batch_size]
            for j, cat in enumerate(batch_cats):
                if i * loader.batch_size + j < len(categories):
                    by_category[cat]["preds"].append(preds[j].item())
                    by_category[cat]["labels"].append(labels[j].item())

        results = {}
        for cat, data in by_category.items():
            if data["labels"]:
                results[cat] = {
                    "accuracy": float(
                        accuracy_score(data["labels"], data["preds"])
                    ) * 100,
                    "count": len(data["labels"]),
                }
        return results

    def per_model_accuracy(self, loader: DataLoader, model_labels: List[str]) -> dict:
        return self.evaluate_by_category(loader, model_labels)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--variant", type=str, default="base")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent.parent
    metadata_paths = list(root.glob("dataset/metadata/*.csv"))

    from .dataset import create_dataloaders
    _, val_loader = create_dataloaders(
        root_dir=str(root),
        metadata_paths=[str(p) for p in metadata_paths],
        batch_size=32,
        num_workers=2,
        size=384,
        val_split=0.15,
        undersample=False,
    )

    evaluator = Evaluator(args.checkpoint, args.variant)
    results = evaluator.evaluate(val_loader)

    print(f"\n{'='*60}")
    print("EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Accuracy:    {results['accuracy']:.2f}%")
    print(f"Precision:   {results['precision']:.2f}%")
    print(f"Recall:      {results['recall']:.2f}%")
    print(f"F1-Score:    {results['f1']:.2f}%")
    print(f"Specificity: {results['specificity']:.2f}%")
    print(f"AUC-ROC:     {results['auc_roc']:.2f}%")
    print(f"\nConfusion Matrix:")
    cm = results["confusion_matrix"]
    print(f"              Predicted Real  Predicted AI")
    print(f"Actual Real   {cm['tn']:>6d}         {cm['fp']:>6d}")
    print(f"Actual AI     {cm['fn']:>6d}         {cm['tp']:>6d}")
    print(f"\nTotal samples: {results['total_samples']}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
