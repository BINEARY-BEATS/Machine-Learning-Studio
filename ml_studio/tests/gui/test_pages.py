import pytest
from unittest.mock import MagicMock
import pandas as pd
from ml_studio.gui.pages.home import HomePage
from ml_studio.gui.pages.data_page import DataPage
from ml_studio.gui.pages.train_page import TrainPage
from ml_studio.gui.pages.evaluate_page import EvaluatePage
from ml_studio.gui.pages.predict_page import PredictPage
from ml_studio.gui.pages.models_page import ModelsPage
from ml_studio.core.training.task import TaskType

@pytest.fixture
def container():
    return MagicMock()

class MockProject:
    name = "Test Project"

def test_home_page_refresh_stats(qtbot, container):
    page = HomePage(container)
    qtbot.addWidget(page)
    
    container.project_manager.current = MockProject()
    
    mock_controller = MagicMock()
    mock_controller.current_dataset.row_count = 100
    page.refresh_stats(mock_controller)
    assert page._project_card._value.text() == "Test Project"

def test_data_page_set_dataset(qtbot, container):
    page = DataPage(container)
    qtbot.addWidget(page)
    mock_dataset = MagicMock()
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    mock_dataset.dataframe = df
    mock_dataset.row_count = 2
    mock_dataset.column_count = 2
    page.set_dataset(mock_dataset)
    assert page._dataset is mock_dataset

def test_train_page_set_dataset(qtbot, container):
    page = TrainPage(container)
    qtbot.addWidget(page)
    
    page.set_dataset("MyData", ["Feature1", "Feature2", "Target"], "Target", TaskType.CLASSIFICATION)
    assert page._target_combo.count() > 0
    assert page._feature_list.count() > 0
    assert page.get_task() == TaskType.CLASSIFICATION

def test_evaluate_page_set_results(qtbot, container):
    page = EvaluatePage(container)
    qtbot.addWidget(page)
    page.show_metrics({"accuracy": 0.95})
    assert page._metrics_table is not None

def test_predict_page_set_predictor(qtbot, container):
    page = PredictPage(container)
    qtbot.addWidget(page)
    mock_predictor = MagicMock()
    
    page.bind_predictor(mock_predictor, ["Feature1", "Feature2"], TaskType.CLASSIFICATION)
    assert page._predictor is mock_predictor

def test_models_page_refresh_list(qtbot, container):
    page = ModelsPage(container)
    qtbot.addWidget(page)
    mock_registry = MagicMock()
    mock_registry.list_models.return_value = [MagicMock()]
    page.refresh(mock_registry)
    assert page._table.rowCount() == 1
