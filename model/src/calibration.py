"""Post-hoc confidence calibration: temperature scaling (Guo et al., 2017)
plus ECE/Brier computation, matching the metrics reported in the paper.

Usage:
    from src.calibration import TemperatureScaler, expected_calibration_error

    scaler = TemperatureScaler()
    scaler.fit(val_logits, val_labels)          # tensors, on any device
    test_probs = scaler.calibrate(test_logits)  # softmax at fitted T
    ece = expected_calibration_error(y_true, y_score)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemperatureScaler(nn.Module):
    """Single-parameter temperature scaling fitted on validation logits.

    Does not change predictions (argmax is invariant to T); only makes the
    softmax confidences match empirical accuracy.
    """
    def __init__(self):
        super().__init__()
        self.log_t = nn.Parameter(torch.zeros(1))

    @property
    def temperature(self) -> float:
        return float(self.log_t.detach().exp())

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.log_t.exp()

    def fit(self, logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 100) -> float:
        logits = logits.detach().float()
        labels = labels.detach().long()
        optimizer = torch.optim.LBFGS([self.log_t], lr=0.05, max_iter=max_iter)

        def _closure():
            optimizer.zero_grad()
            loss = F.cross_entropy(self.forward(logits), labels)
            loss.backward()
            return loss

        optimizer.step(_closure)
        return self.temperature

    @torch.no_grad()
    def calibrate(self, logits: torch.Tensor) -> torch.Tensor:
        """Return calibrated class probabilities."""
        return F.softmax(self.forward(logits.float()), dim=-1)


def expected_calibration_error(y_true, y_score, n_bins: int = 10) -> float:
    """ECE over equal-width confidence bins, with y_score the predicted
    probability of the positive (AI) class -- identical binning to the
    pilot notebooks so pre/post numbers are comparable."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    # confidence of the predicted class for a binary problem
    y_pred = (y_score >= 0.5).astype(int)
    conf = np.where(y_pred == 1, y_score, 1 - y_score)
    correct = (y_pred == y_true).astype(float)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        ece += (mask.mean()) * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def brier_score(y_true, y_score) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    return float(np.mean((y_score - y_true) ** 2))
