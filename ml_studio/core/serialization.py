"""JSON-safe conversion for project / pipeline persistence."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def to_jsonable(obj: Any) -> Any:
    """Convert numpy/pandas/set values into JSON-serializable Python types."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        val = float(obj)
        if np.isnan(val) or np.isinf(val):
            return None
        return val
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return [to_jsonable(x) for x in obj.tolist()]
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, pd.Series):
        return [to_jsonable(x) for x in obj.tolist()]
    if isinstance(obj, set):
        return [to_jsonable(x) for x in sorted(obj, key=lambda v: str(v))]
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(x) for x in obj]
    if hasattr(obj, "item") and callable(obj.item):
        try:
            return to_jsonable(obj.item())
        except Exception:
            pass
    return str(obj)
