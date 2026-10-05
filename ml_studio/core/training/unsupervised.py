"""Helpers for clustering / anomaly estimators without a uniform predict API."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances


def _as_array(X) -> np.ndarray:
    if isinstance(X, pd.DataFrame):
        return X.to_numpy(dtype=float, copy=False)
    return np.asarray(X, dtype=float)


def fit_predict_labels(model: Any, X) -> np.ndarray:
    """Fit model and return labels (fit_predict / labels_ / predict)."""
    if hasattr(model, "fit_predict"):
        try:
            return np.asarray(model.fit_predict(X))
        except AttributeError:
            # e.g. LocalOutlierFactor(novelty=True) exposes fit_predict but rejects it
            pass
    model.fit(X)
    if hasattr(model, "labels_"):
        return np.asarray(model.labels_)
    if hasattr(model, "predict"):
        return np.asarray(model.predict(X))
    raise AttributeError(
        f"{type(model).__name__} has no fit_predict, labels_, or predict"
    )


def predict_new(model: Any, X_new) -> np.ndarray:
    """Predict labels for new rows; DBSCAN uses nearest-core within eps else -1."""
    if _is_dbscan(model):
        return _dbscan_assign(model, X_new)
    if hasattr(model, "predict"):
        return np.asarray(model.predict(X_new))
    raise AttributeError(f"{type(model).__name__} cannot predict on new data")


def _is_dbscan(model: Any) -> bool:
    name = type(model).__name__
    if name == "DBSCAN":
        return True
    return (
        hasattr(model, "components_")
        and hasattr(model, "core_sample_indices_")
        and hasattr(model, "labels_")
        and hasattr(model, "eps")
        and not hasattr(type(model), "predict")
    )


def _dbscan_assign(model: Any, X_new) -> np.ndarray:
    X_arr = _as_array(X_new)
    core = getattr(model, "components_", None)
    if core is None or len(core) == 0:
        return np.full(shape=(len(X_arr),), fill_value=-1, dtype=int)
    core_labels = np.asarray(model.labels_)[np.asarray(model.core_sample_indices_)]
    dists = pairwise_distances(X_arr, core)
    nearest = dists.argmin(axis=1)
    min_d = dists[np.arange(len(X_arr)), nearest]
    assigned = np.where(min_d <= float(model.eps), core_labels[nearest], -1)
    return assigned.astype(int)
