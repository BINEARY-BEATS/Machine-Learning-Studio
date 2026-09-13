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
    assert worker.preprocessing is pages["prepare"].pipeline

def test_app_controller_build_training_worker_no_pipeline(container):
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
    pages["prepare"].pipeline = Pipeline(steps=[])
    
    controller.current_dataset = MagicMock()
    controller.current_dataset.dataframe = pd.DataFrame({"feature1": list(range(20)), "target": [0,1]*10})
    
    worker = controller.build_training_worker(pages)
    assert worker is not None
    assert worker.preprocessing is None

def test_app_controller_load_dataset(container):
    controller = AppController(container)
    
    with patch("ml_studio.gui.app_controller.DatasetLoadWorker") as mock_worker_cls:
        mock_worker = MagicMock()
        mock_worker_cls.return_value = mock_worker
        controller.load_dataset(Path("data.csv"), MagicMock(), MagicMock(), MagicMock(), MagicMock())
        mock_worker_cls.assert_called_once()
        mock_worker.run_in_thread.assert_called_once()
