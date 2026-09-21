import pytest
from unittest.mock import MagicMock
from ml_studio.gui.pages.models_page import ModelsPage
from ml_studio.app.container import AppContainer
from ml_studio.app.config import AppConfig

@pytest.fixture
def container():
    return AppContainer(AppConfig())

def test_models_page_init(container, qtbot):
    page = ModelsPage(container)
    qtbot.addWidget(page)
    assert page._table is not None

def test_models_page_refresh_list(container, qtbot):
    page = ModelsPage(container)
    qtbot.addWidget(page)
    registry = MagicMock()
    
    mock_model = MagicMock()
    mock_model.name = "MyModel"
    mock_model.metadata.model_id = "test-model"
    mock_model.metadata.task = MagicMock()
    mock_model.metadata.task.value = "classification"
    mock_model.metadata.version = "1.0.0"
    mock_model.metadata.created_at.strftime.return_value = "2023-01-01"
    
    registry.list_models.return_value = [mock_model]
    
    page.refresh(registry)
    assert page._table.rowCount() == 1
    
def test_models_page_on_selection_changed(container, qtbot):
    page = ModelsPage(container)
    qtbot.addWidget(page)
    
    registry = MagicMock()
    mock_model = MagicMock()
    mock_model.name = "MyModel"
    mock_model.metadata.model_id = "test-model"
    mock_model.metadata.task = MagicMock()
    mock_model.metadata.task.value = "classification"
    mock_model.metadata.version = "1.0.0"
    mock_model.metadata.created_at.strftime.return_value = "2023-01-01"
    
    registry.list_models.return_value = [mock_model]
    
    page.refresh(registry)
    
    # We want to check if selecting a row changes the details.
    page._table.selectRow(0)
    page._table.itemSelectionChanged.emit()
    
    # Check if a widget from the detail view gets updated (the mock model name)
    assert page._detail_title.text() == "MyModel"
