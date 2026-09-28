"""Model evaluation and metrics calculation for Streamly churn risk prediction.

Calculates ranking, calibration, and classification metrics tailored to retention workflows.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_precision_at_recall(
    y_true: np.ndarray | pd.Series,
    y_probs: np.ndarray | pd.Series,
    target_recall: float = 0.60,
) -> float:
    """Compute the maximum precision achieved when recall is at or above target_recall.

    Useful for retention operations: "If we commit to catching at least 60% of all churners,
    what proportion of flagged accounts will actually be true churners?"

    Args:
        y_true: Ground truth binary labels.
        y_probs: Predicted churn probabilities in [0, 1].
        target_recall: Target recall constraint (e.g. 0.60 for 60%).

    Returns:
        float: Precision at the constrained operating point.
    """
    precisions, recalls, _ = precision_recall_curve(y_true, y_probs)
    # Filter points where recall meets or exceeds the target
    valid_precisions = precisions[recalls >= target_recall]
    if len(valid_precisions) == 0:
        return 0.0
    return float(np.max(valid_precisions))


def evaluate_predictions(
    y_true: np.ndarray | pd.Series,
    y_probs: np.ndarray | pd.Series,
    threshold: float = 0.50,
    target_recall: float = 0.60,
) -> dict[str, Any]:
    """Calculate comprehensive evaluation metrics for churn classification.

    Args:
        y_true: Ground truth binary labels (0 or 1).
        y_probs: Predicted churn probabilities in [0, 1].
        threshold: Decision threshold for discrete classification (default: 0.50).
        target_recall: Constraint for precision_at_recall metric (default: 0.60).

    Returns:
        dict[str, Any] containing scalar metrics and confusion matrix breakdown.
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_probs_arr = np.asarray(y_probs, dtype=float)
    y_pred = (y_probs_arr >= threshold).astype(int)

    roc_auc = float(roc_auc_score(y_true_arr, y_probs_arr))
    pr_auc = float(average_precision_score(y_true_arr, y_probs_arr))
    precision_at_target_recall = compute_precision_at_recall(
        y_true_arr, y_probs_arr, target_recall=target_recall
    )

    prec = float(precision_score(y_true_arr, y_pred, zero_division=0))
    rec = float(recall_score(y_true_arr, y_pred, zero_division=0))
    f1 = float(f1_score(y_true_arr, y_pred, zero_division=0))
    acc = float(accuracy_score(y_true_arr, y_pred))
    brier = float(brier_score_loss(y_true_arr, y_probs_arr))

    tn, fp, fn, tp = confusion_matrix(y_true_arr, y_pred).ravel()

    return {
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        f"precision_at_recall_{int(target_recall * 100)}": round(precision_at_target_recall, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "accuracy": round(acc, 4),
        "brier_score": round(brier, 4),
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
    }
