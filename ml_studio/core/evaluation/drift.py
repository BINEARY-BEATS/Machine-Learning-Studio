"""Population Stability Index (PSI) and Kolmogorov–Smirnov drift checks."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index between two 1-d numeric samples."""
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    expected = expected[np.isfinite(expected)]
    actual = actual[np.isfinite(actual)]
    if len(expected) < 5 or len(actual) < 5:
        return float("nan")
    qs = np.linspace(0, 100, bins + 1)
    breaks = np.unique(np.percentile(expected, qs))
    if len(breaks) < 3:
        return 0.0
    e_counts = np.histogram(expected, bins=breaks)[0].astype(float)
    a_counts = np.histogram(actual, bins=breaks)[0].astype(float)
    e_pct = (e_counts + 1e-6) / (e_counts.sum() + 1e-6 * len(e_counts))
    a_pct = (a_counts + 1e-6) / (a_counts.sum() + 1e-6 * len(a_counts))
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


def ks_statistic(expected: np.ndarray, actual: np.ndarray) -> float:
    """Two-sample KS statistic (no p-value dependency)."""
    expected = np.sort(np.asarray(expected, dtype=float))
    actual = np.sort(np.asarray(actual, dtype=float))
    expected = expected[np.isfinite(expected)]
    actual = actual[np.isfinite(actual)]
    if len(expected) < 5 or len(actual) < 5:
        return float("nan")
    data_all = np.concatenate([expected, actual])
    cdf1 = np.searchsorted(expected, data_all, side="right") / len(expected)
    cdf2 = np.searchsorted(actual, data_all, side="right") / len(actual)
    return float(np.max(np.abs(cdf1 - cdf2)))


def drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    columns: list[str] | None = None,
) -> dict[str, Any]:
    """
    Compare current batch to reference (train) distribution.

    Returns per-column PSI / KS and an overall status.
    """
    cols = columns or [
        c
        for c in reference.columns
        if c in current.columns and pd.api.types.is_numeric_dtype(reference[c])
    ]
    rows = []
    for col in cols:
        ref = reference[col].dropna().to_numpy()
        cur = current[col].dropna().to_numpy()
        p = psi(ref, cur)
        k = ks_statistic(ref, cur)
        status = "stable"
        if np.isfinite(p):
            if p >= 0.25:
                status = "drift"
            elif p >= 0.1:
                status = "shift"
        rows.append(
            {
                "column": col,
                "psi": None if not np.isfinite(p) else round(p, 4),
                "ks": None if not np.isfinite(k) else round(k, 4),
                "status": status,
            }
        )
    drifted = sum(1 for r in rows if r["status"] == "drift")
    shifted = sum(1 for r in rows if r["status"] == "shift")
    overall = "stable"
    if drifted:
        overall = "drift"
    elif shifted:
        overall = "shift"
    return {
        "overall": overall,
        "n_columns": len(rows),
        "n_drift": drifted,
        "n_shift": shifted,
        "columns": rows,
    }
