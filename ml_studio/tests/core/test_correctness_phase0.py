"""Phase 0 correctness: no leakage, categorical round-trip, frozen InferencePipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml_studio.core.inference.predictor import Predictor
from ml_studio.core.persistence.model_registry import ModelRegistry
from ml_studio.core.persistence.serializer import InferencePipeline
from ml_studio.core.pipeline import Pipeline
from ml_studio.core.training.data_prep import EncodingBundle, prepare_for_training
from ml_studio.core.training.task import TaskType
from ml_studio.core.training.trainer import Trainer, TrainingConfig
from ml_studio.transforms.registry import get as get_transform


def _categorical_df(n: int = 80, seed: int = 0) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    colors = rng.choice(["red", "blue", "green"], n)
    sizes = rng.choice(["S", "M", "L"], n)
    # Target correlated with color for a learnable signal
    y = np.where(colors == "red", "yes", np.where(colors == "blue", "no", "maybe"))
    # Flip a few for noise
    flip = rng.rand(n) < 0.1
    y = np.where(flip, rng.choice(["yes", "no", "maybe"], n), y)
    return pd.DataFrame(
        {
            "color": colors,
            "size": sizes,
            "x_num": rng.randn(n),
            "label": y,
        }
    )


def test_prepare_for_training_does_not_encode():
    """Selection helper must not LabelEncode (encoding is train-only in Trainer)."""
    df = _categorical_df(30)
    prepared, target, features, meta = prepare_for_training(
        df, TaskType.CLASSIFICATION, target_column="label"
    )
    assert target == "label"
    assert "color" in features
    assert not pd.api.types.is_numeric_dtype(prepared["color"])
    assert meta.get("label_encoders") == {}


def test_no_prep_fit_on_full_before_split():
    """StandardScaler means must differ when fit on full vs train-only (leakage check)."""
    rng = np.random.RandomState(42)
    n = 100
    # Test set systematically shifted so full-fit mean != train-fit mean
    x = np.concatenate([rng.randn(80), rng.randn(20) + 5.0])
    y = (x > 0).astype(int)
    df = pd.DataFrame({"x": x, "y": y})

    Standard = get_transform("Standard")
    prep = Pipeline(steps=[Standard(columns=["x"])])

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
    assert result.preprocessing is not None

    # Reconstruct train indices the same way Trainer splits
    from sklearn.model_selection import train_test_split

    X_raw = df[["x"]]
    y_raw = df["y"]
    X_tr, X_te, _, _ = train_test_split(
        X_raw, y_raw, test_size=0.2, random_state=42, stratify=y_raw
    )
    train_only_mean = float(X_tr["x"].mean())
    full_mean = float(X_raw["x"].mean())
    assert abs(train_only_mean - full_mean) > 0.1

    step = result.preprocessing.steps[0]
    fitted_mean = step.means_.get("x") if hasattr(step, "means_") else None
    assert fitted_mean is not None
    assert abs(fitted_mean - train_only_mean) < abs(fitted_mean - full_mean) + 1e-9


def test_categorical_round_trip_predict_and_decode(tmp_path: Path):
    df = _categorical_df(100)
    config = TrainingConfig(
        task=TaskType.CLASSIFICATION,
        model_id="logistic_regression",
        target_column="label",
        feature_columns=["color", "size", "x_num"],
        test_size=0.25,
        random_state=0,
        cv_splits=3,
    )
    result = Trainer().train(df, config, preprocessing=None)
    assert result.encoding is not None
    assert result.encoding.target_encoder is not None
    assert "color" in result.encoding.feature_encoders

    registry = ModelRegistry(tmp_path / "models")
    mv = registry.register(result, name="cat-clf")
    pipe = registry.load_pipeline(mv.model_id)
    assert pipe.encoding is not None
    assert pipe.encoding.target_encoder is not None

    predictor = Predictor(pipe)
    # A clearly "red-ish" row should predict a known string label
    pred = predictor.predict_single({"color": "red", "size": "M", "x_num": 0.1})
    assert pred.prediction in ("yes", "no", "maybe")
    assert isinstance(pred.prediction, str)

    # Save/load identity
    out = tmp_path / "art"
    pipe.save(out)
    loaded = InferencePipeline.load(out)
    pred2 = Predictor(loaded).predict_single({"color": "red", "size": "M", "x_num": 0.1})
    assert pred2.prediction == pred.prediction


def test_encoding_bundle_fit_transform_inverse():
    X = pd.DataFrame({"a": ["x", "y", "x"], "b": [1.0, 2.0, 3.0]})
    y = pd.Series(["cat", "dog", "cat"], name="t")
    enc = EncodingBundle().fit(X, y, TaskType.CLASSIFICATION, "t")
    Xt = enc.transform_features(X)
    assert pd.api.types.is_numeric_dtype(Xt["a"])
    yt = enc.transform_target(y)
    assert list(enc.inverse_target(yt)) == ["cat", "dog", "cat"]


def test_prep_steps_not_fit_with_test_rows():
    """Spy: pipeline.fit must receive only train-sized data."""
    rng = np.random.RandomState(1)
    n = 50
    df = pd.DataFrame({"x": rng.randn(n), "y": rng.randint(0, 2, n)})
    Standard = get_transform("Standard")
    prep = Pipeline(steps=[Standard(columns=["x"])])

    fit_sizes: list[int] = []
    orig_fit = prep.fit

    def spy_fit(X, y=None):
        fit_sizes.append(len(X))
        return orig_fit(X, y)

    prep.fit = spy_fit  # type: ignore[method-assign]

    config = TrainingConfig(
        task=TaskType.CLASSIFICATION,
        model_id="logistic_regression",
        target_column="y",
        feature_columns=["x"],
        test_size=0.2,
        random_state=1,
        cv_splits=3,
    )
    Trainer().train(df, config, preprocessing=prep)
    assert fit_sizes, "preprocessing.fit was never called"
    assert fit_sizes[0] == int(n * 0.8) or fit_sizes[0] == n - int(n * 0.2)
    assert fit_sizes[0] < n
