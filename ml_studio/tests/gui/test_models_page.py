import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock

from ml_studio.app.config import AppConfig
from ml_studio.app.container import AppContainer
from ml_studio.core.training.task import TaskType
from ml_studio.gui.pages.models_page import ModelsPage


@pytest.fixture
def container():
    return AppContainer(AppConfig())


def _make_model(name="MyModel"):
    model = MagicMock()
    model.name = name
    model.model_id = "test-model"
    model.version = 1
    model.task = TaskType.CLASSIFICATION
    model.metrics = {"f1": 0.9}
    model.dataset_id = "ds1"
    model.training_timestamp = datetime(2023, 1, 1, tzinfo=timezone.utc)
    model.tags = []
    model.hyperparameters = {}
    model.to_dict.return_value = {"name": name}
    model.artifact_dir = "/tmp/model"
    return model


def test_models_page_init(container, qtbot):
    page = ModelsPage(container)
    qtbot.addWidget(page)
    assert page._table is not None
    assert page._table.selectionBehavior() == page._table.SelectionBehavior.SelectRows


def test_models_page_refresh_list(container, qtbot):
    page = ModelsPage(container)
    qtbot.addWidget(page)
    registry = MagicMock()
    registry.list_models.return_value = [_make_model()]
    page.refresh(registry)
    assert page._table.rowCount() == 1
    # First row auto-selected
    assert page._selected_model is not None
    assert page._detail_title.text() == "MyModel"
    assert not page._selection_badge.isHidden()


def test_models_page_on_selection_changed(container, qtbot):
    page = ModelsPage(container)
    qtbot.addWidget(page)
    registry = MagicMock()
    registry.list_models.return_value = [_make_model("Alpha"), _make_model("Beta")]
    page.refresh(registry)

    page._table.selectRow(1)
    assert page._detail_title.text() == "Beta"
    assert "Selected" in page._selection_badge.text()
    assert page._btn_view.isEnabled()
