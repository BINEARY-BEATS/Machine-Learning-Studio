"""Core tests."""

import numpy as np
import pandas as pd
import pytest

from ml_studio.core.schema import detect_task_type, infer_schema
from ml_studio.core.dataset import Dataset
from ml_studio.core.profiling import optimize_dtypes, profile_dataset
from ml_studio.core.pipeline import PreprocessingPipeline
from ml_studio.core.transforms.base import get_transform, list_transforms
from ml_studio.core.transforms import registry  # noqa: F401
from ml_studio.core.training.task import TaskType
from ml_studio.core.training.registry import get_models_for_task, get_model
from ml_studio.core.training.trainer import Trainer, TrainingConfig
from ml_studio.core.evaluation.metrics import compute_metrics
from ml_studio.core.persistence.serializer import InferencePipeline
from ml_studio.core.persistence.model_registry import ModelRegistry
from ml_studio.core.inference.predictor import Predictor


@pytest.fixture
def sample_df():
    rng = np.random.RandomState(42)
    return pd.DataFrame({
        "a": rng.randn(200),
        "b": rng.randn(200),
        "c": rng.choice(["X", "Y", "Z"], 200),
        "target": rng.randn(200),
    })


def test_infer_schema(sample_df):
    schema = infer_schema(sample_df)
    assert len(schema.columns) == 4


def test_detect_task_type_regression(sample_df):
    assert detect_task_type(sample_df["target"]) == "REGRESSION"


def test_dataset_versioning(sample_df):
    ds = Dataset(name="test")
    ds.set_dataframe(sample_df, reason="load")
    assert ds.version == 2
    assert ds.row_count == 200


def test_optimize_dtypes(sample_df):
    optimized, report = optimize_dtypes(sample_df)
    assert "memory_before" in report


def test_profile_dataset(sample_df):
    ds = Dataset(name="test")
    ds.set_dataframe(sample_df)
    profile = profile_dataset(ds)
    assert profile.row_count == 200


def test_transforms_registered():
    assert "scale_standard" in list_transforms()


def test_pipeline_fit_transform(sample_df):
    pipe = PreprocessingPipeline()
    pipe.add(get_transform("missing_median"))
    pipe.add(get_transform("scale_standard"))
    X = sample_df.drop(columns=["target"])
    out = pipe.fit_transform(X)
    assert out.shape[0] == 200


def test_regression_training(sample_df):
    config = TrainingConfig(
        task=TaskType.REGRESSION,
        model_id="ridge",
        target_column="target",
        feature_columns=["a", "b"],
    )
    trainer = Trainer()
    result = trainer.train(sample_df, config)
    assert "r2" in result.metrics


def test_classification_training():
    rng = np.random.RandomState(42)
    df = pd.DataFrame({
        "a": rng.randn(200),
        "b": rng.randn(200),
        "target": rng.choice([0, 1], 200),
    })
    config = TrainingConfig(
        task=TaskType.CLASSIFICATION,
        model_id="logistic_regression",
        target_column="target",
        feature_columns=["a", "b"],
    )
    result = Trainer().train(df, config)
    assert "accuracy" in result.metrics


def test_clustering_training(sample_df):
    config = TrainingConfig(
        task=TaskType.CLUSTERING,
        model_id="kmeans",
        target_column="target",
        feature_columns=["a", "b"],
    )
    result = Trainer().train(sample_df, config)
    assert "n_clusters" in result.metrics


def test_model_registry_roundtrip(sample_df, tmp_path):
    config = TrainingConfig(
        task=TaskType.REGRESSION,
        model_id="linear_regression",
        target_column="target",
        feature_columns=["a", "b"],
    )
    result = Trainer().train(sample_df, config)
    registry = ModelRegistry(tmp_path)
    mv = registry.register(result, name="test_model")
    pipeline = registry.load_pipeline(mv.model_id)
    preds_before = pipeline.predict(sample_df[["a", "b"]])
    assert len(preds_before) == 200


def test_integration_train_save_load_predict(sample_df, tmp_path):
    config = TrainingConfig(
        task=TaskType.REGRESSION,
        model_id="ridge",
        target_column="target",
        feature_columns=["a", "b"],
    )
    result = Trainer().train(sample_df, config)
    registry = ModelRegistry(tmp_path)
    mv = registry.register(result, name="integration")
    pipeline = registry.load_pipeline(mv.model_id)
    predictor = Predictor(pipeline)
    pred = predictor.predict_single({"a": 0.5, "b": -0.3})
    assert pred.prediction is not None

    reloaded = ModelRegistry(tmp_path).load_pipeline(mv.model_id)
    p1 = pipeline.predict(sample_df[["a", "b"]])
    p2 = reloaded.predict(sample_df[["a", "b"]])
    np.testing.assert_array_almost_equal(p1, p2)
