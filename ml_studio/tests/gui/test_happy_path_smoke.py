"""End-to-end happy path: import → prepare → train → predict → save/open."""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest

from ml_studio.app.config import AppConfig
from ml_studio.app.container import AppContainer
from ml_studio.core.dataset import Dataset
from ml_studio.core.inference.predictor import Predictor
from ml_studio.core.persistence.serializer import InferencePipeline
from ml_studio.core.pipeline import Pipeline
from ml_studio.core.project_manager import ProjectManager
from ml_studio.core.training.task import TaskType
from ml_studio.core.training.trainer import Trainer
from ml_studio.gui.app_controller import AppController
from ml_studio.gui.pages.prepare_page import PreparePage
from ml_studio.gui.pages.train_page import TrainPage
from ml_studio.transforms.registry import get as get_transform
from ml_studio.transforms.registry import list_all


@pytest.fixture
def container():
    return AppContainer(AppConfig())


@pytest.fixture
def tiny_df():
    rng = list(range(40))
    return pd.DataFrame(
        {
            "x1": [i / 40 for i in rng],
            "x2": [1.0 + i / 20 for i in rng],
            "y": [0 if i < 20 else 1 for i in rng],
        }
    )


def test_custom_python_hidden_from_registry():
    names = {m["name"] for m in list_all()}
    assert "CustomPython" not in names


def test_api_prepare_and_sweep_raise(tmp_path, monkeypatch):
    from ml_studio.api import Project

    monkeypatch.chdir(tmp_path)
    p = Project.create("smoke_api")
    df_path = tmp_path / "d.csv"
    pd.DataFrame({"a": [1, 2], "t": [0, 1]}).to_csv(df_path, index=False)
    p.load_data(str(df_path))
    p.set_target("t")

    with pytest.raises(NotImplementedError, match="prepare"):
        p.prepare()
    with pytest.raises(NotImplementedError, match="sweep"):
        p.sweep(["logistic_regression"])


def test_happy_path_train_predict_and_session(qtbot, container, tiny_df, tmp_path):
    """Wire GUI pages + train + predict + .mlstudio round-trip."""
    dataset = Dataset(_dataframe=tiny_df, name="smoke", target_column="y")
    controller = AppController(container, registry_dir=tmp_path / "models")
    controller.new_project()
    controller.current_dataset = dataset

    prepare = PreparePage(container)
    train = TrainPage(container)
    qtbot.addWidget(prepare)
    qtbot.addWidget(train)

    pages = {
        "data": MagicMock(),
        "prepare": prepare,
        "train": train,
        "evaluate": MagicMock(),
        "models": MagicMock(),
        "predict": MagicMock(),
        "home": MagicMock(),
    }

    controller.on_dataset_loaded(dataset, pages)
    assert prepare.dataset is dataset
    assert train.get_task() in (TaskType.CLASSIFICATION, TaskType.REGRESSION)

    # Explicit classification + logistic for a deterministic smoke path
    train.set_dataset("smoke", list(tiny_df.columns), "y", TaskType.CLASSIFICATION)
    train._cv_spin.setValue(3)
    idx = train._model_combo.findText("Logistic Regression")
    if idx >= 0:
        train._model_combo.setCurrentIndex(idx)

    Impute = get_transform("Impute")
    prepare.pipeline.add(Impute(strategy="mean", columns=["x1", "x2"]))
    prepare._refresh_ui()

    config = train.build_config()
    assert config is not None
    assert config.target_column == "y"
    assert "y" not in config.feature_columns

    # Time-series label uses UserRole data, not raw display text
    ts_idx = train._task_combo.findData(TaskType.TIME_SERIES.value)
    assert ts_idx >= 0
    train._task_combo.setCurrentIndex(ts_idx)
    assert train.get_task() == TaskType.TIME_SERIES
    assert "TIME_SERIES" in train._task_combo.currentText()
    # Restore classification for actual training
    train.set_dataset("smoke", list(tiny_df.columns), "y", TaskType.CLASSIFICATION)
    train._cv_spin.setValue(3)
    if idx >= 0:
        train._model_combo.setCurrentIndex(idx)
    config = train.build_config()

    trainer = Trainer()
    result = trainer.train(
        tiny_df,
        config,
        preprocessing=Pipeline(steps=list(prepare.pipeline.steps)),
    )
    assert result.estimator is not None
    assert "accuracy" in result.metrics or "f1" in result.metrics

    pipe = InferencePipeline(
        estimator=result.estimator,
        preprocessing=result.preprocessing,
        feature_columns=result.feature_columns,
        target_column=result.target_column,
        task=result.task,
    )
    predictor = Predictor(pipe)
    pred = predictor.predict_single({"x1": 0.95, "x2": 1.85})
    assert pred.prediction in (0, 1, 0.0, 1.0)

    # Persist session and reopen
    pm = ProjectManager(AppConfig())
    pm._recent_path = tmp_path / "recent.json"
    project = pm.new_project("Smoke")
    project.dataset = dataset
    project.pipeline = prepare.pipeline
    project.schema_overrides = {"y": "target"}
    path = tmp_path / "smoke.mlstudio"
    pm.save_project(path)
    pm.close_project()
    opened = pm.open_project(path)
    assert opened.dataset is not None
    assert len(opened.dataset.dataframe) == 40
    assert len(opened.pipeline.steps) == 1
