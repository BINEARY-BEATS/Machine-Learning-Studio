"""Task-appropriate evaluation metrics."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    davies_bouldin_score,
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
        return _anomaly_metrics(y_pred, X=X, model=model)
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
    labels = np.asarray(labels)
    non_noise = labels[labels >= 0]
    unique = np.unique(non_noise)
    result: dict[str, Any] = {
        "n_clusters": int(len(unique)),
        "cluster_sizes": {int(k): int(np.sum(labels == k)) for k in unique},
        "noise_ratio": float(np.mean(labels == -1)) if len(labels) else 0.0,
    }
    valid = labels >= 0
    n_valid_clusters = len(np.unique(labels[valid])) if valid.any() else 0
    if X is not None and n_valid_clusters >= 2 and int(valid.sum()) >= 2:
        X_arr = X.iloc[valid] if isinstance(X, pd.DataFrame) else np.asarray(X)[valid]
        y_arr = labels[valid]
        try:
            result["silhouette"] = float(silhouette_score(X_arr, y_arr))
        except Exception:
            pass
        try:
            result["davies_bouldin"] = float(davies_bouldin_score(X_arr, y_arr))
        except Exception:
            pass
    if model is not None and hasattr(model, "inertia_"):
        result["inertia"] = float(model.inertia_)
    result["cluster_profile"] = _cluster_profile(X, labels, unique)
    return result


def _cluster_profile(X, labels, unique) -> list[dict[str, Any]]:
    profile: list[dict[str, Any]] = []
    labels = np.asarray(labels)
    for k in unique:
        mask = labels == k
        entry: dict[str, Any] = {"cluster": int(k), "size": int(mask.sum()), "means": {}}
        if X is None or not mask.any():
            profile.append(entry)
            continue
        if isinstance(X, pd.DataFrame):
            sub = X.iloc[mask]
            for col in sub.columns:
                if pd.api.types.is_numeric_dtype(sub[col]):
                    entry["means"][str(col)] = float(sub[col].mean())
        else:
            arr = np.asarray(X)[mask]
            for j in range(arr.shape[1]):
                entry["means"][f"f{j}"] = float(np.mean(arr[:, j]))
        profile.append(entry)
    return profile


def _anomaly_metrics(labels, X=None, model=None) -> dict[str, Any]:
    labels = np.asarray(labels) if labels is not None else np.array([])
    anomalies = int(np.sum(labels == -1)) if len(labels) else 0
    total = len(labels)
    result = {
        "anomaly_count": anomalies,
        "anomaly_ratio": anomalies / total if total else 0.0,
    }
    scores = _anomaly_scores(model, X)
    if scores is not None and len(scores):
        result["score_mean"] = float(np.mean(scores))
        result["score_std"] = float(np.std(scores))
        result["score_min"] = float(np.min(scores))
        result["score_max"] = float(np.max(scores))
    return result


def _anomaly_scores(model, X) -> np.ndarray | None:
    if model is None or X is None:
        return None
    try:
        if hasattr(model, "decision_function"):
            return np.asarray(model.decision_function(X), dtype=float)
        if hasattr(model, "score_samples"):
            return np.asarray(model.score_samples(X), dtype=float)
    except Exception:
        return None
    return None
