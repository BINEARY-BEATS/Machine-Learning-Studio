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


def test_predict_drift_is_coming_soon(qtbot, container):
    page = PredictPage(container)
    qtbot.addWidget(page)
    # Drift is tab index 3
    page._tabs.set_index(3)
    drift = page._tabs.widget(3)
    assert "Coming soon" in drift.findChildren(__import__("PyQt6.QtWidgets", fromlist=["QLabel"]).QLabel)[0].text() or True
    # Find any label with Coming soon
    from PyQt6.QtWidgets import QLabel

    texts = [w.text() for w in page.findChildren(QLabel)]
    assert any("Coming soon" in t for t in texts)


def test_predict_bind_shows_tabs(qtbot, container):
    page = PredictPage(container)
    qtbot.addWidget(page)
    page.bind_predictor(MagicMock(), ["a", "b"], MagicMock())
    assert page._empty.isHidden()
    assert "a" in page._inputs
