"""Train wizard validation and config tests."""

from __future__ import annotations

import pytest
from PyQt6.QtCore import Qt

from ml_studio.app.config import AppConfig
from ml_studio.app.container import AppContainer
from ml_studio.core.training.task import TaskType
from ml_studio.gui.pages.train_page import TrainPage


@pytest.fixture
def container():
    return AppContainer(AppConfig())


@pytest.fixture
def page(qtbot, container):
    p = TrainPage(container)
    qtbot.addWidget(p)
    return p


def test_train_steps_exclude_eval_save(page):
    assert "Eval" not in page.STEPS
    assert "Save" not in page.STEPS
    assert page.STEPS[-1] == "Train"


def test_validate_requires_dataset(page):
    err = page.validate_step(1)
    assert err is not None
    assert "dataset" in err.lower()


def test_validate_features_and_build_config(page):
    page.set_dataset(
        "demo",
        ["a", "b", "y"],
        "y",
        TaskType.CLASSIFICATION,
    )
    assert page.validate_step(2) is None
    config = page.build_config()
    assert config is not None
    assert config.target_column == "y"
    assert "y" not in config.feature_columns
    assert config.tune_method == "none"


def test_tune_optuna_in_config(page):
    page.set_dataset("demo", ["a", "b", "y"], "y", TaskType.REGRESSION)
    page._tune_combo.setCurrentText("Optuna")
    page._tune_trials.setValue(12)
    config = page.build_config()
    assert config.tune_method == "optuna"
    assert config.tune_trials == 12


def test_next_blocked_without_dataset(page, qtbot):
    from unittest.mock import patch

    with patch("ml_studio.gui.pages.train_page.QMessageBox.warning") as warn:
        page._goto_step(0)
        page._next_step()
        # Moving to Dataset (1) requires dataset
        warn.assert_called()
