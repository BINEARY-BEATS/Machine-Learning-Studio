import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
from ml_studio.gui.pages.prepare_page import PreparePage
from ml_studio.core.dataset import Dataset
from ml_studio.core.project import Project, ProjectMetadata
from ml_studio.core.pipeline import Pipeline
from ml_studio.app.container import AppContainer
from ml_studio.app.config import AppConfig


@pytest.fixture
def container():
    return AppContainer(AppConfig())


@pytest.fixture
def page(qtbot, container):
    p = PreparePage(container)
    qtbot.addWidget(p)
    return p


def test_prepare_page_init(page):
    assert page._pipeline_list.count() == 0
    assert not page._empty.isHidden()


def test_prepare_page_set_dataset(page, qtbot):
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    dataset = Dataset(_dataframe=df, name="test")
    page.set_dataset(dataset)
    assert page.dataset is dataset
    assert page.schema_editor._table.rowCount() == 2


def test_prepare_page_set_project(page, qtbot):
    project = Project(metadata=ProjectMetadata(name="test"))
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    project.dataset = Dataset(_dataframe=df, name="test")

    page.set_project(project)
    assert page.schema_editor._table.rowCount() == 2


def test_prepare_page_save_and_clear(page, qtbot):
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    page.set_dataset(Dataset(_dataframe=df, name="test"))
    page._save_btn.click()
    page._clear_btn.click()
    assert len(page.pipeline.steps) == 0


def test_prepare_page_reorder(page, qtbot):
    from ml_studio.transforms.registry import get as get_transform

    Impute = get_transform("Impute")
    OneHot = get_transform("OneHot")

    page.pipeline.add(Impute(strategy="mean", columns=["a"]))
    page.pipeline.add(OneHot(columns=["b"]))
    page._refresh_ui()

    assert page.pipeline.steps[0].__class__.__name__ == "Impute"
    assert page.pipeline.steps[1].__class__.__name__ == "OneHot"

    page._pipeline_list.model().moveRow(
        page._pipeline_list.model().index(1, 0).parent(),
        1,
        page._pipeline_list.model().index(0, 0).parent(),
        0,
    )

    assert page.pipeline.steps[0].__class__.__name__ == "OneHot"
    assert page.pipeline.steps[1].__class__.__name__ == "Impute"


def test_prepare_page_toggle_step(page):
    from ml_studio.transforms.registry import get as get_transform

    Impute = get_transform("Impute")
    page.pipeline.add(Impute(strategy="mean", columns=["a"]))
    page._refresh_ui()
    page._toggle_step(0, False)
    assert page.pipeline.steps[0].enabled is False
