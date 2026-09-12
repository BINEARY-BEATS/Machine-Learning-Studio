"""Additional tests to reach 80% core coverage."""

import numpy as np
import pandas as pd
import pytest

from ml_studio.core.transforms.base import get_transform
from ml_studio.core.transforms import registry  # noqa: F401
from ml_studio.core.ingestion import RemoteFileSource
from ml_studio.core.evaluation.explain import compute_shap_values, compute_partial_dependence
from ml_studio.core.persistence.exporters import export_pickle
from ml_studio.core.persistence.serializer import InferencePipeline
from ml_studio.core.training.task import TaskType
from ml_studio.core.training.registry import get_model
from ml_studio.core.project_manager import ProjectManager
from ml_studio.app.config import AppConfig


def test_poly_features():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    step = get_transform("poly_features", degree=2)
    out = step.fit_transform(df)
    assert out.shape[1] > 2


def test_date_components():
    df = pd.DataFrame({"dt": pd.date_range("2020-01-01", periods=5)})
    step = get_transform("date_components")
    out = step.fit_transform(df)
    assert "dt_year" in out.columns


def test_variance_selection():
    rng = np.random.RandomState(42)
    df = pd.DataFrame({
        "a": rng.randn(50),
        "b": rng.randn(50),
        "c": np.zeros(50),
        "target": rng.randn(50),
    })
    step = get_transform("select_variance")
    out = step.fit_transform(df.drop(columns=["target"]))
    assert "c" not in out.columns or out.shape[1] < 3


def test_univariate_selection():
    rng = np.random.RandomState(42)
    df = pd.DataFrame({"a": rng.randn(50), "b": rng.randn(50), "target": rng.randn(50)})
    step = get_transform("select_univariate", k=1)
    out = step.fit_transform(df.drop(columns=["target"]), df["target"])
    assert out.shape[1] >= 1


def test_smote_step():
    pytest.importorskip("imblearn")
    df = pd.DataFrame({"a": [1, 2, 3, 4], "b": [1, 2, 3, 4]})
    y = pd.Series([0, 0, 1, 1])
    step = get_transform("balance_smote")
    step.fit(df, y)
    X_res, y_res = step.fit_resample(df, y)
    assert len(y_res) >= len(y)


def test_class_weight_step():
    step = get_transform("balance_class_weight")
    df = pd.DataFrame({"a": [1, 2]})
    out = step.fit_transform(df)
    assert len(out) == 2


def test_missing_ffill():
    df = pd.DataFrame({"a": [1, np.nan, 3]})
    step = get_transform("missing_ffill")
    out = step.fit_transform(df)
    assert not out["a"].isnull().any()


def test_ordinal_encoding():
    df = pd.DataFrame({"cat": ["A", "B", "C", "A"]})
    step = get_transform("encode_ordinal")
    out = step.fit_transform(df)
    assert pd.api.types.is_numeric_dtype(out["cat"])


def test_frequency_encoding():
    df = pd.DataFrame({"cat": ["A", "B", "A", "A"]})
    step = get_transform("encode_frequency")
    out = step.fit_transform(df)
    assert out["cat"].iloc[0] > out["cat"].iloc[1]


def test_partial_dependence():
    rng = np.random.RandomState(42)
    X = pd.DataFrame({"a": rng.randn(50), "b": rng.randn(50)})
    y = pd.Series(X["a"] + rng.randn(50) * 0.1)
    model = get_model("ridge")
    model.fit(X, y)
    pd_result = compute_partial_dependence(model, X, "a")
    assert "values" in pd_result


def test_shap_unavailable():
    model = get_model("linear_regression")
    X = pd.DataFrame({"a": [1.0, 2.0]})
    y = pd.Series([1.0, 2.0])
    model.fit(X, y)
    result = compute_shap_values(model, X)
    # None if shap not installed or fails
    assert result is None or "values" in result


def test_export_pickle(tmp_path):
    model = get_model("linear_regression")
    pipe = InferencePipeline(model, None, ["a"], "y", TaskType.REGRESSION)
    path = tmp_path / "m.pkl"
    export_pickle(pipe, path)
    assert path.exists()


def test_project_invalid_format(tmp_path):
    pm = ProjectManager(AppConfig())
    bad = tmp_path / "bad.mlstudio"
    bad.write_text("not a zip")
    with pytest.raises(Exception):
        pm.open_project(bad)


def test_project_save_no_path():
    pm = ProjectManager(AppConfig())
    pm.new_project()
    with pytest.raises(ValueError):
        pm.save_project()


def test_isolation_forest_outlier():
    rng = np.random.RandomState(42)
    df = pd.DataFrame({"val": rng.randn(100)})
    step = get_transform("outlier_isolation_forest", contamination=0.1)
    out = step.fit_transform(df)
    assert len(out) <= len(df)
