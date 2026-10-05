"""B2 acceptance: categorical round-trip via TrainingWorker → registry → predict."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml_studio.core.inference.predictor import Predictor
from ml_studio.core.persistence.model_registry import ModelRegistry
from ml_studio.core.training.task import TaskType
from ml_studio.core.training.trainer import TrainingConfig
from ml_studio.gui.workers.training_worker import TrainingWorker


def _churn_df(n: int = 80, seed: int = 0) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    cities = rng.choice(["Lahore", "Karachi", "Islamabad"], n)
    ages = rng.randint(18, 70, n).astype(float)
    # Label correlated with city for a learnable signal
    y = np.where(cities == "Lahore", "yes", "no")
    flip = rng.rand(n) < 0.15
    y = np.where(flip, rng.choice(["yes", "no"], n), y)
    return pd.DataFrame({"city": cities, "age": ages, "churn": y})


def test_roundtrip_train_worker_predict_decoded_labels(tmp_path: Path):
    df = _churn_df(100, seed=1)
    config = TrainingConfig(
        task=TaskType.CLASSIFICATION,
        model_id="logistic_regression",
        target_column="churn",
        feature_columns=["city", "age"],
        test_size=0.25,
        random_state=1,
        cv_splits=3,
    )
    worker = TrainingWorker(df, config, preprocessing=None, prepare_data=True)
    result = worker.do_work()

    assert result.target_classes is not None
    assert set(result.target_classes) == {"yes", "no"}
    assert result.feature_schema
    assert result.feature_schema["city"]["kind"] == "categorical"
    assert result.feature_schema["age"]["kind"] == "numeric"

    registry = ModelRegistry(tmp_path / "models")
    mv = registry.register(result, name="churn-clf")
    pipe = registry.load_pipeline(mv.model_id)
    assert pipe.target_classes is not None
    assert pipe.feature_schema

    predictor = Predictor(pipe)
    out = predictor.predict_single({"city": "Lahore", "age": "31"})
    assert out.prediction in ("yes", "no")
    assert isinstance(out.prediction, str)
    assert isinstance(out.probabilities, dict)
    assert set(out.probabilities) == set(pipe.target_classes)
    assert abs(sum(out.probabilities.values()) - 1.0) < 1e-5


def test_unseen_category_and_blank_numeric_do_not_raise(tmp_path: Path):
    df = _churn_df(80, seed=2)
    config = TrainingConfig(
        task=TaskType.CLASSIFICATION,
        model_id="logistic_regression",
        target_column="churn",
        feature_columns=["city", "age"],
        test_size=0.25,
        random_state=2,
        cv_splits=3,
    )
    result = TrainingWorker(df, config, prepare_data=True).do_work()
    registry = ModelRegistry(tmp_path / "models")
    mv = registry.register(result, name="t")
    predictor = Predictor(registry.load_pipeline(mv.model_id))

    out = predictor.predict_single({"city": "Multan", "age": ""})
    assert out.prediction in ("yes", "no")


def test_non_numeric_text_in_numeric_field_raises(tmp_path: Path):
    df = _churn_df(80, seed=3)
    config = TrainingConfig(
        task=TaskType.CLASSIFICATION,
        model_id="logistic_regression",
        target_column="churn",
        feature_columns=["city", "age"],
        test_size=0.25,
        random_state=3,
        cv_splits=3,
    )
    result = TrainingWorker(df, config, prepare_data=True).do_work()
    registry = ModelRegistry(tmp_path / "models")
    mv = registry.register(result, name="t")
    predictor = Predictor(registry.load_pipeline(mv.model_id))

    with pytest.raises(ValueError) as exc:
        predictor.predict_single({"city": "Lahore", "age": "thirty"})
    err = exc.value.args[0]
    assert isinstance(err, dict)
    assert "age" in err


def test_coerce_row_unknown_category_allowed(tmp_path: Path):
    df = _churn_df(60, seed=4)
    config = TrainingConfig(
        task=TaskType.CLASSIFICATION,
        model_id="logistic_regression",
        target_column="churn",
        feature_columns=["city", "age"],
        test_size=0.25,
        random_state=4,
        cv_splits=3,
    )
    result = TrainingWorker(df, config, prepare_data=True).do_work()
    registry = ModelRegistry(tmp_path / "models")
    mv = registry.register(result, name="t")
    pipe = registry.load_pipeline(mv.model_id)
    row = pipe.coerce_row({"city": "Peshawar", "age": "40"})
    assert row.loc[0, "city"] == "Peshawar"
    assert float(row.loc[0, "age"]) == 40.0
