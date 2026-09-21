import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
import pandas as pd
from ml_studio.gui.app_controller import AppController
from ml_studio.app.config import AppConfig
from ml_studio.app.container import AppContainer
from ml_studio.core.pipeline import Pipeline
from ml_studio.core.training.task import TaskType
from ml_studio.core.training.trainer import TrainingConfig

@pytest.fixture
def container():
    config = AppConfig()
    return AppContainer(config=config)

def test_app_controller_initial_state(container):
    controller = AppController(container)
    assert controller.container.project_manager.current is None
    assert controller.container is container

def test_app_controller_new_project_creates_project(container):
    controller = AppController(container)
    controller.new_project()
    assert controller.container.project_manager.current is not None
    assert controller.container.project_manager.current.metadata.name == "Untitled Project"

def test_app_controller_open_project_loads_data(container, tmp_path):
    controller = AppController(container)
    parent_mock = MagicMock()
    proj_path = tmp_path / "test.mlstudio"
    proj_path.touch()
    with patch("ml_studio.gui.app_controller.QFileDialog.getOpenFileName", return_value=(str(proj_path), "")):
        with patch.object(container.project_manager, "open_project", return_value=True):
            result = controller.open_project(parent_mock)
            assert result is True

def test_app_controller_open_project_cancelled(container):
    controller = AppController(container)
    parent_mock = MagicMock()
    with patch("ml_studio.gui.app_controller.QFileDialog.getOpenFileName", return_value=("", "")):
        result = controller.open_project(parent_mock)
        assert result is False

def test_app_controller_save_project(container):
    controller = AppController(container)
    parent_mock = MagicMock()
    controller.new_project()
    controller.container.project_manager.current.path = None
    controller.container.project_manager.save_project = MagicMock()
    with patch("ml_studio.gui.app_controller.QFileDialog.getSaveFileName", return_value=("test.yaml", "")):
        controller.save_project(parent_mock)
        controller.container.project_manager.save_project.assert_called_once()

def test_app_controller_pick_dataset_path(container):
    controller = AppController(container)
    with patch("ml_studio.gui.app_controller.QFileDialog.getOpenFileName", return_value=("data.csv", "")):
        res = controller.pick_dataset_path(None)
        assert res == Path("data.csv")

def test_app_controller_cleanup_worker(container):
    controller = AppController(container)
    thread = MagicMock()
    thread.isRunning.return_value = True
    controller._active_thread = thread
    controller.cleanup_worker()
    thread.quit.assert_called_once()
    thread.wait.assert_called_once()
    assert controller._active_thread is None

def test_app_controller_start_worker(container):
    controller = AppController(container)
    worker = MagicMock()
    controller.start_worker(worker, MagicMock(), MagicMock(), MagicMock(), MagicMock())
    assert controller._active_worker == worker

def test_app_controller_build_training_worker_with_pipeline(container):
    controller = AppController(container)
    pages = {"train": MagicMock(), "prepare": MagicMock()}
    
    mock_config = TrainingConfig(
        task=TaskType.CLASSIFICATION,
        target_column="target",
        feature_columns=["feature1"],
        model_id="Random Forest",
        hyperparameters={},
        test_size=0.2
    )
    pages["train"].build_config.return_value = mock_config
    pages["prepare"].pipeline = Pipeline(steps=[("scaler", MagicMock())])
    
    controller.current_dataset = MagicMock()
    controller.current_dataset.dataframe = pd.DataFrame({"feature1": list(range(20)), "target": [0,1]*10})
    
    worker = controller.build_training_worker(pages)
    assert worker is not None
    assert worker.preprocessing is not None
    assert len(worker.preprocessing.steps) == 1

def test_app_controller_on_dataset_loaded(container):
    controller = AppController(container)
    controller.new_project()
    dataset = MagicMock()
    dataset.name = "MyData"
    dataset.dataframe = pd.DataFrame({"col1": [1, 2], "col2": [0, 1]})
    pages = {"data": MagicMock(), "train": MagicMock(), "prepare": MagicMock(), "home": MagicMock()}
    
    with patch("ml_studio.gui.app_controller.suggest_training_columns", return_value=("col2", ["col1"], TaskType.CLASSIFICATION)):
        controller.on_dataset_loaded(dataset, pages)
    
    assert controller.current_dataset == dataset
    pages["data"].set_dataset.assert_called_once_with(dataset)
    pages["prepare"].set_dataset.assert_called_once_with(dataset)
    pages["train"].set_dataset.assert_called_once()
    pages["home"].refresh_stats.assert_called_once()

def test_app_controller_optimize_memory(container):
    controller = AppController(container)
    dataset = MagicMock()
    dataset.dataframe = pd.DataFrame({"a": [1, 2, 3]})
    controller.current_dataset = dataset
    pages = {"data": MagicMock(), "prepare": MagicMock()}

    worker = controller.build_optimize_worker()
    assert worker is not None
    with patch.object(dataset, "set_dataframe") as mock_set:
        msg = controller.apply_optimize_result(
            {"dataframe": pd.DataFrame({"a": [1]}), "report": {"reduction_pct": 50.0}},
            pages,
        )
        mock_set.assert_called_once()
        assert "50.0%" in msg
        pages["data"].set_dataset.assert_called()

def test_app_controller_validate_training(container):
    controller = AppController(container)
    pages = {"train": MagicMock()}
    pages["train"].build_config.return_value = None
    with patch("ml_studio.gui.app_controller.QMessageBox.warning") as mock_warn:
        assert not controller.validate_training(pages, None)
        mock_warn.assert_called_once()

def test_app_controller_on_training_complete(container):
    controller = AppController(container)
    pages = {"evaluate": MagicMock(), "predict": MagicMock(), "models": MagicMock(), "home": MagicMock()}
    
    result = MagicMock()
    result.task = TaskType.CLASSIFICATION
    result.metrics = {"accuracy": 0.95, "baseline_accuracy": 0.50}
    result.model_id = "logistic_regression"
    result.feature_columns = ["a"]
    controller.registry = MagicMock()
    controller.current_dataset = MagicMock()
    controller.current_dataset.name = "ds"
    
    with patch("ml_studio.core.inference.predictor.Predictor"):
        ok, msg = controller.on_training_complete(result, pages)
    assert ok is True
    assert "complete" in msg.lower()
    assert controller.current_result == result
    pages["evaluate"].add_run.assert_called_once()


def test_app_controller_quality_gate_fails(container):
    controller = AppController(container)
    pages = {"evaluate": MagicMock(), "predict": MagicMock(), "models": MagicMock(), "home": MagicMock()}
    result = MagicMock()
    result.task = TaskType.REGRESSION
    result.metrics = {"r2": -0.5}
    ok, msg = controller.on_training_complete(result, pages)
    assert ok is False
    assert "Quality gate" in msg
    pages["evaluate"].add_run.assert_not_called()
