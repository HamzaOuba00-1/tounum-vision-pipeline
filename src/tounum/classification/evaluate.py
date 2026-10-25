"""Metrics, threshold selection and error analysis for the photo classifier."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(
    y_true: np.ndarray, probabilities: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    """Threshold-dependent and ranking metrics (F1 macro is the selection metric)."""
    y_true = np.asarray(y_true).astype(int)
    probabilities = np.asarray(probabilities, dtype=float)
    predictions = (probabilities >= threshold).astype(int)
    metrics = {
        "accuracy": accuracy_score(y_true, predictions),
        "balanced_accuracy": balanced_accuracy_score(y_true, predictions),
        "precision_photo": precision_score(y_true, predictions, pos_label=1, zero_division=0),
        "recall_photo": recall_score(y_true, predictions, pos_label=1, zero_division=0),
        "f1_macro": f1_score(y_true, predictions, average="macro", zero_division=0),
    }
    if len(np.unique(y_true)) == 2:  # ranking metrics need both classes
        metrics["roc_auc"] = roc_auc_score(y_true, probabilities)
        metrics["pr_auc"] = average_precision_score(y_true, probabilities)
    else:
        metrics["roc_auc"] = float("nan")
        metrics["pr_auc"] = float("nan")
    return {name: float(value) for name, value in metrics.items()}


def select_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    start: float = 0.10,
    stop: float = 0.90,
    num: int = 81,
) -> tuple[float, float, pd.DataFrame]:
    """Decision threshold maximising F1 macro **on the validation set**.

    Returns ``(best_threshold, best_f1, curve)``. Ties keep the lowest
    threshold (same as ``np.argmax`` in the notebook).
    """
    thresholds = np.linspace(start, stop, int(num))
    scores = [
        f1_score(y_true, (probabilities >= t).astype(int), average="macro", zero_division=0)
        for t in thresholds
    ]
    best = int(np.argmax(scores))
    curve = pd.DataFrame({"threshold": thresholds, "f1_macro": scores})
    return float(round(thresholds[best], 10)), float(scores[best]), curve


def confusion_frame(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> pd.DataFrame:
    predictions = (np.asarray(probabilities) >= threshold).astype(int)
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    return pd.DataFrame(matrix, index=["true_other", "true_photo"], columns=["pred_other", "pred_photo"])


def source_class_analysis(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    threshold: float,
    positive_class: str = "Photo",
    source_classes: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Binary error rate per original class.

    For the positive class the error is the false-negative rate; for every
    other class it is the false-positive rate (e.g. paintings taken for photos).
    """
    data = frame[["source_class", "label"]].copy()
    data["probability_photo"] = np.asarray(probabilities, dtype=float)
    data["prediction"] = (data["probability_photo"] >= threshold).astype(int)
    rows = []
    for source_class, group in data.groupby("source_class"):
        if source_class == positive_class:
            error_rate, error_type = (group["prediction"] == 0).mean(), "false_negative"
        else:
            error_rate, error_type = (group["prediction"] == 1).mean(), "false_positive"
        rows.append(
            {
                "source_class": source_class,
                "images": len(group),
                "mean_probability_photo": group["probability_photo"].mean(),
                "binary_error_rate": error_rate,
                "error_type": error_type,
            }
        )
    result = pd.DataFrame(rows)
    if source_classes is not None:
        order = {name: i for i, name in enumerate(source_classes)}
        result = result.sort_values("source_class", key=lambda s: s.map(order))
    return result.reset_index(drop=True)


def plateau_analysis(val_loss: Sequence[float], min_delta: float = 0.001, patience: int = 3) -> dict:
    """Best epoch and start of the validation-loss plateau (notebook analysis)."""
    values = np.asarray(val_loss, dtype=float)
    best_index = int(np.argmin(values))
    plateau_epoch = None
    for index in range(best_index + 1, len(values) - patience + 1):
        if np.all(values[index : index + patience] >= values[index - 1] - min_delta):
            plateau_epoch = index + 1
            break
    return {
        "best_epoch": best_index + 1,
        "best_val_loss": float(values[best_index]),
        "plateau_start_epoch": plateau_epoch,
    }
