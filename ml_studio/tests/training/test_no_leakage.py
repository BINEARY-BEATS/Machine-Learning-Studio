"""B1 acceptance: no prep leakage into test/CV folds; row alignment; GUI clone."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import KFold, train_test_split

from ml_studio.core.pipeline import Pipeline
from ml_studio.core.training.cv_runner import (
    apply_preprocessing,
    cross_val_score_leakfree,
    fit_preprocessing,
    split_data,
)
from ml_studio.core.training.task import TaskType
from ml_studio.core.training.trainer import Trainer, TrainingConfig
from ml_studio.transforms.base import BaseTransform
from ml_studio.transforms.missing import DropRows, Impute
from ml_studio.transforms.registry import _REGISTRY, _auto_discover


class FitRecorder(BaseTransform):
    """Records every index set seen during _fit for leakage assertions."""

    recorded: list[set] = []

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @classmethod
    def get_schema(cls) -> dict:
        return {"type": "object", "properties": {}}

    def _fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> None:
        FitRecorder.recorded.append(set(X.index.tolist()))
        self.fitted_columns_ = list(X.columns)

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X

    def to_dict(self) -> dict:
        return dict(self.params)

    @classmethod
    def from_dict(cls, d: dict) -> "FitRecorder":
        return cls(**d)


@pytest.fixture
def register_fit_recorder():
    _auto_discover()
    _REGISTRY["FitRecorder"] = FitRecorder
    FitRecorder.recorded = []
    yield FitRecorder
    FitRecorder.recorded = []
    _REGISTRY.pop("FitRecorder", None)


def _binary_df(n: int = 60, seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    x = rng.randn(n)
    y = (x > 0).astype(int)
    return pd.DataFrame({"x": x, "y": y})


def test_fit_recorder_never_sees_holdout_test(register_fit_recorder):
    """Holdout test indices must never appear in any preprocessing fit set."""
    df = _binary_df(60, seed=7)
    config = TrainingConfig(
        task=TaskType.CLASSIFICATION,
        model_id="logistic_regression",
        target_column="y",
        feature_columns=["x"],
        test_size=0.25,
        random_state=7,
        cv_splits=3,
    )
    X_raw = df[["x"]]
    y_raw = df["y"]
    _, X_test, _, _ = train_test_split(
        X_raw,
        y_raw,
        test_size=0.25,
        random_state=7,
        stratify=y_raw,
    )
    test_idx = set(X_test.index.tolist())

    prep = Pipeline(steps=[FitRecorder()])
    Trainer().train(df, config, preprocessing=prep)

    assert FitRecorder.recorded, "FitRecorder was never fitted"
    for seen in FitRecorder.recorded:
        leaked = seen & test_idx
        assert not leaked, f"test indices leaked into fit: {leaked}"


def test_cv_fold_fits_exclude_validation_indices(register_fit_recorder):
    """Each CV fold's prep fit must exclude that fold's validation indices."""
    df = _binary_df(48, seed=9)
    X, y = df[["x"]], df["y"]
    prep = Pipeline(steps=[FitRecorder()])
    cv = KFold(n_splits=3, shuffle=True, random_state=42)
    FitRecorder.recorded = []

    def factory():
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(max_iter=200)

    cross_val_score_leakfree(factory, prep, X, y, cv, "accuracy")
    assert len(FitRecorder.recorded) == 3
    for (tr, va), seen in zip(cv.split(X, y), FitRecorder.recorded):
        val_labels = set(X.index[va].tolist())
        train_labels = set(X.index[tr].tolist())
        assert seen.isdisjoint(val_labels)
        assert seen == train_labels

def test_impute_mean_ignores_test_outlier():
    """Extreme outlier only in the test split must not affect the fitted mean."""
    rng = np.random.RandomState(0)
    n = 100
    x = rng.randn(n)
    y = (x > 0).astype(int)
    df = pd.DataFrame({"x": x, "y": y})

    X_raw, y_raw = df[["x"]], df["y"]
    X_train, X_test, _, _ = train_test_split(
        X_raw, y_raw, test_size=0.2, random_state=42, stratify=y_raw
    )
    # Put an extreme outlier only in the first test row
    outlier_pos = X_test.index[0]
    df.loc[outlier_pos, "x"] = 1_000_000.0
    train_only_mean = float(df.loc[X_train.index, "x"].mean())

    prep = Pipeline(steps=[Impute(strategy="mean", columns=["x"])])
    config = TrainingConfig(
        task=TaskType.CLASSIFICATION,
        model_id="logistic_regression",
        target_column="y",
        feature_columns=["x"],
        test_size=0.2,
        random_state=42,
        cv_splits=3,
    )
    result = Trainer().train(df, config, preprocessing=prep)
    fitted = result.preprocessing.steps[0]
    assert abs(float(fitted.imputers_["x"]) - train_only_mean) < 1e-9


def test_drop_rows_keeps_x_y_aligned():
    """DropRows may shrink X; y must stay index-aligned and training must succeed."""
    df = pd.DataFrame(
        {
            "a": [1.0, np.nan, 3.0, np.nan, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                  11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0],
            "b": [1.0] * 20,
            "y": [0, 1] * 10,
        }
    )
    prep = Pipeline(steps=[DropRows(threshold=0.0, columns=["a"])])
    config = TrainingConfig(
        task=TaskType.CLASSIFICATION,
        model_id="logistic_regression",
        target_column="y",
        feature_columns=["a", "b"],
        test_size=0.25,
        random_state=0,
        cv_splits=3,
    )
    X = df[["a", "b"]]
    y = df["y"]
    X_train, _, y_train, _ = split_data(X, y, config)
    fitted, X_train_t, y_train_aligned = fit_preprocessing(prep, X_train, y_train)
    assert len(X_train_t) == len(y_train_aligned)
    assert list(X_train_t.index) == list(y_train_aligned.index)
    assert len(X_train_t) < len(X_train)

    result = Trainer().train(df, config, preprocessing=prep)
    assert result.train_size > 0
    assert result.preprocessing is not None
    assert result.preprocessing.steps[0]._is_fitted


def test_original_pipeline_stays_unfitted_after_training():
    """Training must clone — Prepare-page step instances stay unfitted."""
    df = _binary_df(40, seed=3)
    prep = Pipeline(steps=[Impute(strategy="mean", columns=["x"])])
    assert prep.steps[0]._is_fitted is False

    config = TrainingConfig(
        task=TaskType.CLASSIFICATION,
        model_id="logistic_regression",
        target_column="y",
        feature_columns=["x"],
        test_size=0.25,
        random_state=3,
        cv_splits=3,
    )
    result = Trainer().train(df, config, preprocessing=prep)
    assert prep.steps[0]._is_fitted is False
    assert result.preprocessing is not None
    assert result.preprocessing.steps[0]._is_fitted is True
    assert result.preprocessing.steps[0] is not prep.steps[0]


def test_clone_unfitted_rebuilds_fresh_instances():
    src = Pipeline(steps=[Impute(strategy="median", columns=["a"])])
    src.steps[0].fit(pd.DataFrame({"a": [1.0, 2.0, np.nan]}))
    assert src.steps[0]._is_fitted

    cloned = src.clone_unfitted()
    assert cloned.steps[0] is not src.steps[0]
    assert cloned.steps[0]._is_fitted is False
    assert cloned.steps[0].params["strategy"] == "median"
    assert getattr(cloned.steps[0], "enabled", True) is True


def test_clone_unfitted_copies_enabled_flag():
    step = Impute(strategy="mean")
    step.enabled = False
    pipe = Pipeline(steps=[step])
    cloned = pipe.clone_unfitted()
    assert cloned.steps[0].enabled is False


def test_split_data_time_series_is_chronological():
    n = 50
    df = pd.DataFrame({"x": np.arange(n, dtype=float), "y": np.arange(n, dtype=float)})
    config = TrainingConfig(
        task=TaskType.REGRESSION,
        model_id="linear_regression",
        target_column="y",
        feature_columns=["x"],
        test_size=0.2,
        random_state=0,
        is_time_series=True,
    )
    X_tr, X_te, y_tr, y_te = split_data(df[["x"]], df["y"], config)
    assert X_tr.index.max() < X_te.index.min()
    assert y_tr.index.max() < y_te.index.min()


def test_cross_val_score_leakfree_refits_prep_per_fold(register_fit_recorder):
    df = _binary_df(48, seed=11)
    X, y = df[["x"]], df["y"]
    prep = Pipeline(steps=[FitRecorder()])
    cv = KFold(n_splits=3, shuffle=True, random_state=0)

    def factory():
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(max_iter=200)

    scores = cross_val_score_leakfree(factory, prep, X, y, cv, "accuracy")
    assert len(scores) == 3
    assert all(np.isfinite(scores))
    # Exactly one fit per fold (fold-train only)
    assert len(FitRecorder.recorded) == 3
    for (_, va), seen in zip(cv.split(X, y), FitRecorder.recorded):
        assert seen.isdisjoint(set(X.index[va].tolist()))


def test_apply_preprocessing_aligns_y_after_drop():
    X = pd.DataFrame({"a": [1.0, np.nan, 3.0, 4.0], "b": [1, 1, 1, 1]})
    y = pd.Series([0, 1, 0, 1], name="y")
    fitted, X_t, y_a = fit_preprocessing(
        Pipeline(steps=[DropRows(threshold=0.0, columns=["a"])]), X, y
    )
    X2, y2 = apply_preprocessing(fitted, X, y)
    assert len(X_t) == len(y_a)
    assert len(X2) == len(y2)
    assert list(X2.index) == list(y2.index)


def test_app_controller_passes_cloned_unfitted_pipeline():
    """build_training_worker must not share step instances with Prepare page."""
    from unittest.mock import MagicMock

    from ml_studio.gui.app_controller import AppController
    from ml_studio.core.training.task import TaskType

    container = MagicMock()
    container.model_registry_dir = "."
    controller = AppController(container)
    step = Impute(strategy="mean", columns=["feature1"])
    prepare_pipe = Pipeline(steps=[step])

    pages = {"train": MagicMock(), "prepare": MagicMock()}
    pages["train"].build_config.return_value = TrainingConfig(
        task=TaskType.CLASSIFICATION,
        target_column="target",
        feature_columns=["feature1"],
        model_id="logistic_regression",
        hyperparameters={},
        test_size=0.2,
    )
    pages["prepare"].pipeline = prepare_pipe
    controller.current_dataset = MagicMock()
    controller.current_dataset.dataframe = pd.DataFrame(
        {"feature1": list(range(20)), "target": [0, 1] * 10}
    )

    worker = controller.build_training_worker(pages)
    assert worker is not None
    assert worker.preprocessing is not None
    assert worker.preprocessing.steps[0] is not step
    assert worker.preprocessing.steps[0]._is_fitted is False
    assert isinstance(worker.preprocessing.steps[0], Impute)
