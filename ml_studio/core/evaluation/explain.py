"""Model explainability."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance, partial_dependence


def compute_permutation_importance(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    n_repeats: int = 10,
) -> dict[str, float]:
    result = permutation_importance(model, X, y, n_repeats=n_repeats, random_state=42, n_jobs=-1)
    return {col: float(imp) for col, imp in zip(X.columns, result.importances_mean)}


def compute_shap_values(model, X: pd.DataFrame, max_samples: int = 500) -> dict[str, Any] | None:
    try:
        import shap
    except ImportError:
        return None
    sample = X.head(max_samples)
    try:
        explainer = shap.Explainer(model, sample)
        values = explainer(sample)
        return {
            "values": values.values.tolist(),
            "base_values": values.base_values.tolist() if hasattr(values, "base_values") else [],
            "feature_names": list(X.columns),
        }
    except Exception:
        try:
            explainer = shap.KernelExplainer(model.predict, sample)
            sv = explainer.shap_values(sample)
            return {"values": np.array(sv).tolist(), "feature_names": list(X.columns)}
        except Exception:
            return None


def compute_partial_dependence(
    model,
    X: pd.DataFrame,
    feature: str,
    grid_resolution: int = 50,
) -> dict[str, list]:
    col_idx = list(X.columns).index(feature)
    pd_result = partial_dependence(model, X, features=[col_idx], grid_resolution=grid_resolution)
    return {
        "feature": feature,
        "values": pd_result["grid_values"][0].tolist(),
        "average": pd_result["average"][0].tolist(),
    }


def explain_single_prediction(
    model,
    X: pd.DataFrame,
    row_index: int,
    feature_names: list[str] | None = None,
) -> dict[str, Any]:
    row = X.iloc[[row_index]]
    prediction = model.predict(row)[0]
    result: dict[str, Any] = {"prediction": prediction, "features": row.iloc[0].to_dict()}
    if hasattr(model, "predict_proba"):
        try:
            result["probabilities"] = model.predict_proba(row)[0].tolist()
        except Exception:
            pass
    shap = compute_shap_values(model, row)
    if shap:
        result["shap"] = shap
    return result
