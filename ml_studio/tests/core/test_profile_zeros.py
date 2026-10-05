"""Regression: profile zeros and top_values display."""

from __future__ import annotations

import pandas as pd

from ml_studio.core.dataset import Dataset
from ml_studio.core.profiling import profile_dataset


def test_column_profile_keeps_zero_stats_and_top_values():
    df = pd.DataFrame({"z": [0, 0, 1, 0, 2], "c": ["a", "a", "b", "a", "c"]})
    ds = Dataset(name="t", source="mem")
    ds.set_dataframe(df, reason="test")
    profile = profile_dataset(ds)
    z = next(c for c in profile.columns if c.name == "z")
    assert z.min == 0.0
    assert z.mean is not None
    assert z.top_values[:1] == [0] or z.top_values[0] == 0
    c = next(col for col in profile.columns if col.name == "c")
    assert c.top_values[0] == "a"
    assert len(c.top_values) <= 3
