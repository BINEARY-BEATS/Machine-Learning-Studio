"""Task-appropriate evaluation metrics."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)

from ml_studio.core.training.task import TaskType


def compute_metrics(
    task: TaskType,
    y_true=None,
    y_pred=None,
    y_proba=None,
    X=None,
    model=None,
) -> dict[str, Any]:
    if task == TaskType.REGRESSION:
        return _regression_metrics(y_true, y_pred)
    if task == TaskType.CLASSIFICATION:
        return _classification_metrics(y_true, y_pred, y_proba)
    if task == TaskType.CLUSTERING:
        return _clustering_metrics(X, y_pred, model)
    if task == TaskType.ANOMALY_DETECTION:
        return _anomaly_metrics(y_pred)
    if task == TaskType.TIME_SERIES:
        return _regression_metrics(y_true, y_pred)
    return {}


def _regression_metrics(y_true, y_pred) -> dict[str, Any]:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
    }


def _classification_metrics(y_true, y_pred, y_proba=None) -> dict[str, Any]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    if y_true is not None and len(y_true) > 0:
        if hasattr(y_true, "value_counts"):
            majority = y_true.value_counts().iloc[0]
        else:
            import pandas as pd
            majority = pd.Series(y_true).value_counts().iloc[0]
        metrics["baseline_accuracy"] = float(majority / len(y_true))
    else:
        metrics["baseline_accuracy"] = 0.0
    if y_proba is not None and len(np.unique(y_true)) == 2:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba[:, 1]))
        except Exception:
            pass
    return metrics


def _clustering_metrics(X, labels, model) -> dict[str, Any]:
    unique = np.unique(labels)
    unique = unique[unique >= 0] if hasattr(labels, "__iter__") else unique
    result: dict[str, Any] = {
        "n_clusters": int(len(unique)),
        "cluster_sizes": {int(k): int(np.sum(labels == k)) for k in unique},
    }
    if X is not None and len(unique) >= 2:
        try:
            result["silhouette"] = float(silhouette_score(X, labels))
        except Exception:
            pass
    if model is not None and hasattr(model, "inertia_"):
        result["inertia"] = float(model.inertia_)
    return result


def _anomaly_metrics(labels) -> dict[str, Any]:
    anomalies = int(np.sum(labels == -1)) if labels is not None else 0
    total = len(labels) if labels is not None else 0
    return {
        "anomaly_count": anomalies,
        "anomaly_ratio": anomalies / total if total else 0,
    }
