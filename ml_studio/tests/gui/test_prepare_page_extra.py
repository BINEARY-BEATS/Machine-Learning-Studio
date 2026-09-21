"""Extra prepare-page coverage against the current pipeline UI."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from ml_studio.app.config import AppConfig
from ml_studio.app.container import AppContainer
from ml_studio.core.dataset import Dataset
from ml_studio.gui.pages.prepare_page import PreparePage


@pytest.fixture
def container():
    return AppContainer(AppConfig())


@pytest.fixture
def page(qtbot, container):
    p = PreparePage(container)
    qtbot.addWidget(p)
    df = pd.DataFrame({"a": [1.0, None, 3.0], "b": ["x", "y", "x"]})
    p.set_dataset(Dataset(_dataframe=df, name="test"))
    return p


def test_prepare_page_add_step_via_dialogs(page, qtbot):
    with patch("ml_studio.gui.pages.prepare_page.TransformPickerDialog") as mock_picker, patch(
        "ml_studio.gui.pages.prepare_page.StepConfigDialog"
    ) as mock_config:
        picker = mock_picker.return_value
        picker.exec.return_value = mock_picker.DialogCode.Accepted
        picker.selected_transform = "Impute"

        config = mock_config.return_value
        config.exec.return_value = mock_config.DialogCode.Accepted
        from ml_studio.transforms.registry import get as get_transform

        config.transform_class = get_transform("Impute")
        config.final_params = {"strategy": "mean", "columns": ["a"]}

        page._add_step()
        assert len(page.pipeline.steps) == 1
        assert page.pipeline.steps[0].__class__.__name__ == "Impute"


def test_prepare_page_preview_requires_dataset(container, qtbot):
    page = PreparePage(container)
    qtbot.addWidget(page)
    page.window = MagicMock(return_value=MagicMock(_toast=MagicMock()))
    with qtbot.assertNotEmitted(page.preview_requested):
        page._on_preview()


def test_prepare_page_preview_emits_signal(page, qtbot):
    with qtbot.waitSignal(page.preview_requested, timeout=1000):
        page._on_preview()


def test_prepare_page_recipe_uses_yaml(page, qtbot):
    from PyQt6.QtWidgets import QMessageBox
    from ml_studio.core.pipeline import Pipeline
    from ml_studio.transforms.registry import get as get_transform

    if page._recipe_combo.count() <= 1:
        pytest.skip("No recipes available")

    with patch("ml_studio.gui.pages.prepare_page.QMessageBox.question") as mock_q, patch(
        "ml_studio.gui.pages.prepare_page.apply_recipe"
    ) as mock_apply:
        mock_q.return_value = QMessageBox.StandardButton.Yes
        pipe = Pipeline()
        pipe.add(get_transform("Impute")(strategy="mean", columns=["a"]))
        mock_apply.return_value = pipe

        page._on_recipe_selected(1)
        mock_apply.assert_called_once()
        schema_arg = mock_apply.call_args[0][1]
        assert isinstance(schema_arg, dict)
        assert "a" in schema_arg
