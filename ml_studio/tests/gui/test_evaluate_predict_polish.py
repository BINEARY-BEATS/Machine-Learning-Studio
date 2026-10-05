"""Evaluate / Predict polish tests."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from ml_studio.app.config import AppConfig
from ml_studio.app.container import AppContainer
from ml_studio.gui.pages.evaluate_page import EvaluatePage
from ml_studio.gui.pages.predict_page import PredictPage
from ml_studio.gui.shell.top_bar import TopBar


@pytest.fixture
def container():
    return AppContainer(AppConfig())


def test_evaluate_show_metrics(qtbot, container):
    page = EvaluatePage(container)
    qtbot.addWidget(page)
    page.show_metrics({"accuracy": 0.91, "f1": 0.88})
    assert page._metrics_table.rowCount() == 2
    assert "0.9100" in page._primary_card._value.text() or "0.88" in page._primary_card._value.text()


def test_top_bar_breadcrumb(qtbot):
    bar = TopBar()
    qtbot.addWidget(bar)
    bar.set_breadcrumb(["ML Studio", "Prepare"])
    assert "Prepare" in bar._breadcrumb.text()
    assert "›" in bar._breadcrumb.text()


def test_predict_drift_tab_has_psi_controls(qtbot, container):
    page = PredictPage(container)
    qtbot.addWidget(page)
    page._tabs.set_index(3)
    from PyQt6.QtWidgets import QLabel, QPushButton

    texts = [w.text() for w in page.findChildren(QLabel)]
    assert any("PSI" in t or "Drift" in t or "drift" in t for t in texts)
    assert any("Compute drift" in b.text() for b in page.findChildren(QPushButton))
    assert hasattr(page, "_drift_table")
    assert page._drift_table.columnCount() == 4


def test_predict_bind_shows_tabs(qtbot, container):
    page = PredictPage(container)
    qtbot.addWidget(page)
    page.bind_predictor(MagicMock(), ["a", "b"], MagicMock())
    assert page._empty.isHidden()
    assert "a" in page._inputs
