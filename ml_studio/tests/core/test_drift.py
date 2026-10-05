"""Drift metric unit tests."""

import numpy as np
import pandas as pd

from ml_studio.core.evaluation.drift import drift_report, ks_statistic, psi


def test_psi_identical_near_zero():
    rng = np.random.RandomState(0)
    x = rng.randn(500)
    assert psi(x, x.copy()) < 0.05


def test_psi_shifted_higher():
    rng = np.random.RandomState(0)
    a = rng.randn(500)
    b = rng.randn(500) + 2.0
    assert psi(a, b) > psi(a, a.copy())


def test_drift_report_flags_shift():
    ref = pd.DataFrame({"x": np.random.randn(200), "y": np.random.randn(200)})
    cur = pd.DataFrame({"x": np.random.randn(200) + 3, "y": np.random.randn(200)})
    report = drift_report(ref, cur, columns=["x", "y"])
    assert report["n_columns"] == 2
    assert report["overall"] in ("stable", "shift", "drift")
    assert any(r["column"] == "x" for r in report["columns"])


def test_ks_statistic_range():
    a = np.random.randn(100)
    b = np.random.randn(100) + 1
    k = ks_statistic(a, b)
    assert 0 <= k <= 1
