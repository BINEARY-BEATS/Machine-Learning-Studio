"""Tests for data preparation and table filter fixes."""

import numpy as np
import pandas as pd
import pytest

from ml_studio.core.training.data_prep import prepare_for_training, suggest_training_columns
from ml_studio.core.training.task import TaskType
from ml_studio.gui.widgets.data_table import DataFrameTableModel


def test_filter_with_backslash_does_not_crash():
    df = pd.DataFrame({"a": ["foo\\bar", "baz"], "b": [1, 2]})
    model = DataFrameTableModel(df)
    model.apply_filter("\\")  # previously caused ArrowInvalid regex error
    assert model.rowCount() >= 0


def test_prepare_mixed_types():
    rng = np.random.RandomState(42)
    n = 20
    df = pd.DataFrame({
        "age": rng.randint(20, 60, n).astype(float),
        "sex": rng.choice(["M", "F"], n),
        "fare": rng.uniform(5, 50, n),
        "survived": rng.choice([0, 1], n),
    })
    df.loc[0, "age"] = np.nan
    prepared, target, features, _ = prepare_for_training(
        df, TaskType.CLASSIFICATION, target_column="survived"
    )
    assert len(prepared) >= 10
    assert target == "survived"
    assert features


def test_suggest_columns():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    target, features, task = suggest_training_columns(df)
    assert target == "b"
    assert task == TaskType.REGRESSION
