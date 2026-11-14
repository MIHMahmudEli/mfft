"""Statistical utilities for the paper: bootstrap confidence intervals
and McNemar's test for paired classifier comparison.

Usage:
    from src.stats import bootstrap_ci, mcnemar_test

    acc_ci = bootstrap_ci(y_true, y_pred, metric="accuracy")
    auc_ci = bootstrap_ci(y_true, y_score, metric="auc")
    p = mcnemar_test(y_true, preds_mfft, preds_baseline)
"""
import numpy as np
from typing import Callable, Dict, Union
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
)

_METRICS: Dict[str, Callable] = {
    "accuracy": accuracy_score,
    "precision": precision_score,
    "recall": recall_score,
    "f1": f1_score,
    "auc": roc_auc_score,  # expects scores, not hard predictions
}


def bootstrap_ci(
    y_true,
    y_pred_or_score,
    metric: Union[str, Callable] = "accuracy",
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict:
    """Percentile bootstrap CI for a classification metric.

    Pass hard predictions for accuracy/precision/recall/f1 and
    probability scores for auc.
    """
    y_true = np.asarray(y_true)
    y_hat = np.asarray(y_pred_or_score)
    fn = _METRICS[metric] if isinstance(metric, str) else metric

    rng = np.random.default_rng(seed)
    n = len(y_true)
    point = fn(y_true, y_hat)

    values = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        # resampling can produce a single-class sample where auc is undefined
        if len(np.unique(y_true[idx])) < 2:
            continue
        values.append(fn(y_true[idx], y_hat[idx]))
    values = np.sort(values)

    alpha = (1 - confidence) / 2
    return {
        "point": float(point),
        "lower": float(np.quantile(values, alpha)),
        "upper": float(np.quantile(values, 1 - alpha)),
        "n_resamples": len(values),
        "confidence": confidence,
    }


def mcnemar_test(y_true, pred_a, pred_b) -> dict:
    """McNemar's test (exact binomial for small discordant counts,
    continuity-corrected chi-square otherwise) comparing two classifiers
    evaluated on the same test set.

    Returns the discordant counts and two-sided p-value. A small p-value
    means the two classifiers' error patterns differ significantly.
    """
    from scipy import stats as sps

    y_true = np.asarray(y_true)
    a_correct = np.asarray(pred_a) == y_true
    b_correct = np.asarray(pred_b) == y_true

    n01 = int(np.sum(a_correct & ~b_correct))  # A right, B wrong
    n10 = int(np.sum(~a_correct & b_correct))  # A wrong, B right
    n_discordant = n01 + n10

    if n_discordant == 0:
        return {"n01": n01, "n10": n10, "statistic": 0.0, "p_value": 1.0, "method": "degenerate"}

    if n_discordant < 25:
        p = float(sps.binomtest(min(n01, n10), n_discordant, 0.5).pvalue)
        return {"n01": n01, "n10": n10, "statistic": float(min(n01, n10)),
                "p_value": p, "method": "exact-binomial"}

    stat = (abs(n01 - n10) - 1) ** 2 / n_discordant
    p = float(sps.chi2.sf(stat, df=1))
    return {"n01": n01, "n10": n10, "statistic": float(stat),
            "p_value": p, "method": "chi2-corrected"}
