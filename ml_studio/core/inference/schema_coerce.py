"""Build feature schema and coerce/decode helpers for inference."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def build_feature_schema(X: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Per-column schema from RAW X_train: kind, dtype, nullable, categories, min/max/ref."""
    schema: dict[str, dict[str, Any]] = {}
    for col in X.columns:
        s = X[col]
        entry: dict[str, Any] = {
            "kind": "categorical",
            "dtype": str(s.dtype),
            "nullable": bool(s.isna().any()),
            "categories": None,
            "min": None,
            "max": None,
            "ref": None,
        }
        if pd.api.types.is_bool_dtype(s):
            entry["kind"] = "boolean"
            mode = s.mode(dropna=True)
            entry["ref"] = bool(mode.iloc[0]) if len(mode) else False
        elif pd.api.types.is_datetime64_any_dtype(s):
            entry["kind"] = "datetime"
            valid = s.dropna()
            entry["ref"] = valid.iloc[0].isoformat() if len(valid) else None
        elif pd.api.types.is_numeric_dtype(s):
            entry["kind"] = "numeric"
            valid = s.dropna()
            if len(valid):
                entry["min"] = float(valid.min())
                entry["max"] = float(valid.max())
                entry["ref"] = float(valid.median())
        else:
            entry["kind"] = "categorical"
            counts = s.dropna().astype(str).value_counts()
            cats = counts.index.tolist()[:50]
            entry["categories"] = cats
            entry["ref"] = cats[0] if cats else None
        schema[col] = entry
    return schema


def coerce_value(kind: str, field: str, value: Any) -> Any:
    """Cast one value; empty → NaN; uncastable raises ValueError({field: msg})."""
    if value is None or (isinstance(value, str) and value.strip() == ""):
        return np.nan
    try:
        if kind == "numeric":
            return float(value)
        if kind == "boolean":
            return _coerce_bool(value)
        if kind == "datetime":
            ts = pd.to_datetime(value, errors="raise")
            return ts
        return str(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError({field: f"must be a valid {kind} value"}) from None


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().lower()
    if text in ("1", "true", "yes", "y"):
        return True
    if text in ("0", "false", "no", "n"):
        return False
    raise ValueError("invalid boolean")


def decode_predictions(preds: Any, target_classes: list | None) -> Any:
    """Map integer class indices back to original labels."""
    if not target_classes:
        return preds
    arr = np.asarray(preds)
    classes = list(target_classes)

    def _one(i: Any) -> Any:
        try:
            idx = int(i)
        except (TypeError, ValueError):
            return i
        if 0 <= idx < len(classes):
            return classes[idx]
        return i

    if arr.ndim == 0:
        return _one(arr.item())
    flat = [_one(v) for v in arr.ravel()]
    if len(flat) == 1 and arr.ndim <= 1:
        return flat[0]
    return np.asarray(flat, dtype=object).reshape(arr.shape)


def needs_auto_encode(X: pd.DataFrame) -> bool:
    """True when non-numeric columns or any NaN remain."""
    if X.isna().any().any():
        return True
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            return True
    return False
