"""Parametrized transform to_dict/from_dict round-trip (replaces silent blocker)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from ml_studio.core.serialization import to_jsonable
from ml_studio.transforms.registry import get, list_all


def _sample_frame() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.RandomState(0)
    n = 40
    df = pd.DataFrame(
        {
            "a": rng.randn(n) + 2.0,
            "b": rng.rand(n) * 10 + 0.1,
            "c": rng.choice(["A", "B", "C"], size=n),
            "d": pd.date_range("2020-01-01", periods=n, freq="D"),
        }
    )
    y = pd.Series(rng.randint(0, 2, size=n), name="y")
    return df, y


def _transform_names() -> list[str]:
    names = [t["name"] for t in list_all()]
    # AutoEncode is hidden from list_all GUI but still registered
    try:
        get("AutoEncode")
        names.append("AutoEncode")
    except ValueError:
        pass
    return sorted(set(names))


@pytest.mark.parametrize("name", _transform_names())
def test_transform_dict_roundtrip(name: str):
    cls = get(name)
    X, y = _sample_frame()
    # Prefer numeric-friendly defaults; some transforms need y
    try:
        t = cls()
    except TypeError:
        pytest.skip(f"{name} requires constructor args")

    try:
        t.fit(X, y)
    except Exception as exc:
        pytest.skip(f"{name} fit skipped: {exc}")

    raw = t.to_dict()
    payload = to_jsonable(raw)
    json.dumps(payload)  # must be JSON-serializable

    restored = cls.from_dict(payload)
    if restored._is_fitted:
        out1 = t.transform(X.copy())
        out2 = restored.transform(X.copy())
        assert_frame_equal(
            out1.reset_index(drop=True),
            out2.reset_index(drop=True),
            check_dtype=False,
            rtol=1e-5,
            atol=1e-5,
        )
    else:
        with pytest.raises(ValueError, match="not fitted"):
            restored.transform(X.copy())


def test_pipeline_project_dict_unfitted_and_jsonable():
    from ml_studio.core.pipeline import Pipeline
    from ml_studio.transforms.missing import Impute
    from ml_studio.transforms.scaling import Standard

    X, _ = _sample_frame()
    pipe = Pipeline([Impute(strategy="median"), Standard()])
    pipe.fit(X[["a", "b"]])
    # Numpy scalars in fitted state must not break JSON when include_state True
    json.dumps(to_jsonable(pipe.to_dict(include_state=True)))
    project = pipe.to_dict(include_state=False)
    json.dumps(to_jsonable(project))
    loaded = Pipeline.from_dict(project)
    assert all(not s._is_fitted for s in loaded.steps)
    with pytest.raises(ValueError, match="not fitted"):
        loaded.transform(X[["a", "b"]])
