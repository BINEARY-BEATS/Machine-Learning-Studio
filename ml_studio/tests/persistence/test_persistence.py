"""Persistence tests."""

import numpy as np
import pandas as pd
import pytest

from ml_studio.core.persistence.serializer import InferencePipeline
from ml_studio.core.training.task import TaskType
from ml_studio.core.training.registry import get_model


def test_inference_pipeline_save_load(tmp_path):
    model = get_model("linear_regression")
    X = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    y = pd.Series([1.0, 2.0])
    model.fit(X, y)
    pipe = InferencePipeline(
        estimator=model,
        preprocessing=None,
        feature_columns=["a", "b"],
        target_column="y",
        task=TaskType.REGRESSION,
    )
    pipe.save(tmp_path)
    loaded = InferencePipeline.load(tmp_path)
    np.testing.assert_array_almost_equal(
        pipe.predict(X), loaded.predict(X)
    )
