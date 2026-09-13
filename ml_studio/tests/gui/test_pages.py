import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication
from ml_studio.app.config import AppConfig
from ml_studio.app.container import AppContainer
from ml_studio.gui.pages.home import HomePage
from ml_studio.gui.pages.data_page import DataPage
from ml_studio.gui.pages.settings_page import SettingsPage
from ml_studio.gui.pages.prepare_page import PreparePage
from ml_studio.gui.pages.train_page import TrainPage
from ml_studio.gui.pages.evaluate_page import EvaluatePage
from ml_studio.gui.pages.predict_page import PredictPage
from ml_studio.gui.pages.models_page import ModelsPage

@pytest.fixture
def container():
    config = AppConfig()
    return AppContainer(config=config)

def test_home_page_refresh_stats(qtbot, container, mocker):
    page = HomePage(container)
    qtbot.addWidget(page)
    
    mock_pm = mocker.MagicMock()
    mock_pm.current.name = "My Test Project"
    # Using patch directly to container.project_manager is easiest here if it's accessible
    container.project_manager = mock_pm
    
    page.refresh_stats(controller=None)
    assert page._project_card._value.text() == "My Test Project"

def test_data_page_set_dataset(qtbot, container, mocker):
    page = DataPage(container)
    qtbot.addWidget(page)
    
    mock_dataset = mocker.MagicMock()
    mock_dataset.dataframe.shape = (100, 5)
    mock_dataset.dataframe.memory_usage.return_value.sum.return_value = 1024
    
    page.set_dataset(mock_dataset)
    # The summary should be updated
    assert "100" in page._summary.text()
    assert "5" in page._summary.text()

def test_train_page_set_dataset(qtbot, container):
    page = TrainPage(container)
    qtbot.addWidget(page)
    
    page.set_dataset("Test Dataset", ["col1", "col2", "target"], "target", "classification")
    
    assert page._dataset_label.text() == "Dataset: Test Dataset"
    assert page._target_combo.currentText() == "target"
    assert page._task_combo.currentText() == "Classification"

def test_evaluate_page_set_results(qtbot, container, mocker):
    page = EvaluatePage(container)
    qtbot.addWidget(page)
    
    result = mocker.MagicMock()
    result.metrics = {"accuracy": 0.95, "f1": 0.92}
    result.model_info = {"type": "RandomForest", "params": {}}
    result.task = "classification"
    
    page.set_results(result)
    
    # Check that text contains the metrics
    assert "accuracy" in page._metrics.toPlainText()
    assert "0.95" in page._metrics.toPlainText()

def test_predict_page_set_predictor(qtbot, container, mocker):
    page = PredictPage(container)
    qtbot.addWidget(page)
    
    predictor = mocker.MagicMock()
    predictor.schema = {"col1": "float64", "col2": "int64"}
    
    page.set_predictor(predictor)
    
    assert page._predictor is predictor
    # Check schema was populated
    # The schema editor will have rows
    assert page._schema.rowCount() == 2

def test_models_page_refresh_list(qtbot, container, mocker):
    page = ModelsPage(container)
    qtbot.addWidget(page)
    
    mock_controller = mocker.MagicMock()
    mock_controller.registry.list_models.return_value = [
        mocker.MagicMock(id="model1", task="classification", metrics={"accuracy": 0.9}),
        mocker.MagicMock(id="model2", task="regression", metrics={"r2": 0.8})
    ]
    
    page.refresh_list(mock_controller)
    
    # Model list should have 2 rows
    assert page._model_list.rowCount() == 2
