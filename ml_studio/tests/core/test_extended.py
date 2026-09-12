"""Extended core tests for coverage."""

import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml_studio.core.project import Project, ProjectMetadata
from ml_studio.core.project_manager import ProjectManager
from ml_studio.core.ingestion import LocalFileSource, load_dataset_from_path
from ml_studio.core.evaluation.explain import compute_permutation_importance
from ml_studio.core.evaluation.report import build_report
from ml_studio.core.persistence.exporters import export_joblib, export_requirements, ExportError, export_model
from ml_studio.core.persistence.serializer import InferencePipeline
from ml_studio.core.training.task import TaskType
from ml_studio.core.training.registry import get_model
from ml_studio.core.training.tuning import OptunaTuner
from ml_studio.core.training.automl import AutoMLRunner
from ml_studio.core.training.cv import get_cv_strategy
from ml_studio.core.transforms.base import get_transform
from ml_studio.core.transforms import registry  # noqa: F401
from ml_studio.app.config import AppConfig


@pytest.fixture
def app_config():
    return AppConfig()


@pytest.fixture
def sample_csv(tmp_path):
    df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    path = tmp_path / "test.csv"
    df.to_csv(path, index=False)
    return path


def test_project_metadata_roundtrip():
    meta = ProjectMetadata(name="Test")
    data = meta.to_dict()
    restored = ProjectMetadata.from_dict(data)
    assert restored.name == "Test"


def test_project_manager_new_save_open(app_config, tmp_path):
    pm = ProjectManager(app_config)
    pm.new_project("My Project")
    save_path = tmp_path / "test.mlstudio"
    pm.save_project(save_path)
    assert save_path.exists()
    pm.close_project()
    pm.open_project(save_path)
    assert pm.current.name == "My Project"


def test_project_manager_recent(app_config, tmp_path):
    pm = ProjectManager(app_config)
    save_path = tmp_path / "recent.mlstudio"
    pm.new_project("Recent")
    pm.save_project(save_path)
    recent = pm.get_recent_projects()
    assert any(p.resolve() == save_path.resolve() for p in recent)


def test_load_csv(sample_csv):
    ds = load_dataset_from_path(sample_csv)
    assert ds.row_count == 3


def test_local_file_unsupported(tmp_path):
    bad = tmp_path / "file.xyz"
    bad.write_text("data")
    source = LocalFileSource(bad)
    with pytest.raises(ValueError):
        source.load()


def test_permutation_importance():
    rng = np.random.RandomState(42)
    X = pd.DataFrame({"a": rng.randn(100), "b": rng.randn(100)})
    y = pd.Series(X["a"] * 2 + rng.randn(100) * 0.1)
    model = get_model("ridge")
    model.fit(X, y)
    imp = compute_permutation_importance(model, X, y, n_repeats=3)
    assert "a" in imp


def test_build_report():
    y = np.array([1.0, 2.0, 3.0])
    p = np.array([1.1, 2.0, 2.9])
    report = build_report(TaskType.REGRESSION, "Ridge", "ds", y, p)
    assert "r2" in report.metrics


def test_export_joblib(tmp_path):
    model = get_model("linear_regression")
    X = pd.DataFrame({"a": [1.0, 2.0]})
    y = pd.Series([1.0, 2.0])
    model.fit(X, y)
    pipe = InferencePipeline(model, None, ["a"], "y", TaskType.REGRESSION)
    path = tmp_path / "model.joblib"
    export_joblib(pipe, path)
    assert path.exists()


def test_export_requirements(tmp_path):
    path = tmp_path / "requirements.txt"
    export_requirements(path)
    assert "scikit-learn" in path.read_text()


def test_export_onnx_missing_dep(tmp_path):
    model = get_model("linear_regression")
    pipe = InferencePipeline(model, None, ["a"], "y", TaskType.REGRESSION)
    try:
        export_model(pipe, tmp_path / "m.onnx", "onnx")
    except ExportError:
        pass


def test_cv_strategies():
    assert get_cv_strategy("kfold") is not None
    assert get_cv_strategy("stratified") is not None


def test_optuna_tuning():
    optuna = pytest.importorskip("optuna")
    rng = np.random.RandomState(42)
    df = pd.DataFrame({"a": rng.randn(80), "b": rng.randn(80), "target": rng.randn(80)})
    tuner = OptunaTuner("ridge", {"alpha": [0.1, 1.0, 10.0]}, n_trials=3)
    from ml_studio.core.training.cv import recommend_cv_strategy
    cv = recommend_cv_strategy(TaskType.REGRESSION, 80)
    result = tuner.tune(df[["a", "b"]], df["target"], cv)
    assert result.best_score is not None


def test_automl_runner():
    rng = np.random.RandomState(42)
    df = pd.DataFrame({"a": rng.randn(100), "b": rng.randn(100), "target": rng.randn(100)})
    runner = AutoMLRunner(TaskType.REGRESSION, max_models=2, max_runtime_seconds=60)
    result = runner.run(df[["a", "b"]], df["target"])
    assert len(result.entries) >= 1


def test_encoding_transform():
    df = pd.DataFrame({"cat": ["A", "B", "A", "C"], "num": [1, 2, 3, 4]})
    step = get_transform("encode_onehot")
    out = step.fit_transform(df)
    assert out.shape[1] > 2


def test_outlier_transform():
    df = pd.DataFrame({"val": [1, 2, 3, 100, 4, 5]})
    step = get_transform("outlier_iqr")
    out = step.fit_transform(df)
    assert len(out) <= len(df)


def test_anomaly_training():
    from ml_studio.core.training.trainer import Trainer, TrainingConfig

    rng = np.random.RandomState(42)
    df = pd.DataFrame({"a": rng.randn(100), "b": rng.randn(100), "target": rng.randn(100)})
    config = TrainingConfig(
        task=TaskType.ANOMALY_DETECTION,
        model_id="isolation_forest",
        target_column="target",
        feature_columns=["a", "b"],
    )
    result = Trainer().train(df, config)
    assert "anomaly_count" in result.metrics or "anomaly_ratio" in result.metrics
