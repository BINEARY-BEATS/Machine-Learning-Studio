"""Evaluation tests."""

import numpy as np

from ml_studio.core.evaluation.metrics import compute_metrics
from ml_studio.core.training.task import TaskType


def test_regression_metrics():
    y = np.array([1.0, 2.0, 3.0])
    p = np.array([1.1, 2.1, 2.9])
    m = compute_metrics(TaskType.REGRESSION, y, p)
    assert "r2" in m
    assert m["r2"] > 0.9


def test_classification_metrics():
    y = np.array([0, 1, 0, 1])
    p = np.array([0, 1, 0, 0])
    m = compute_metrics(TaskType.CLASSIFICATION, y, p)
    assert "accuracy" in m
