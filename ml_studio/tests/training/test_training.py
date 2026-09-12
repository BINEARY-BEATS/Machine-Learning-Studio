"""Training tests."""

import numpy as np
import pandas as pd

from ml_studio.core.training.cv import recommend_cv_strategy
from ml_studio.core.training.task import TaskType
from ml_studio.core.training.registry import MODEL_REGISTRY, get_models_for_task


def test_model_registry_has_classification():
    models = get_models_for_task(TaskType.CLASSIFICATION)
    assert len(models) >= 5


def test_model_registry_has_regression():
    models = get_models_for_task(TaskType.REGRESSION)
    assert len(models) >= 5


def test_recommend_cv_classification():
    cv = recommend_cv_strategy(TaskType.CLASSIFICATION, 100, n_classes=2)
    assert cv is not None


def test_recommend_cv_timeseries():
    cv = recommend_cv_strategy(TaskType.TIME_SERIES, 100, is_time_series=True)
    assert "TimeSeriesSplit" in type(cv).__name__
