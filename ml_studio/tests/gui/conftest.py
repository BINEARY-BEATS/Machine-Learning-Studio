"""Shared Qt test fixtures."""

from __future__ import annotations

import pytest
from PyQt6.QtWidgets import QApplication

from ml_studio.app.theme import ThemeMode, apply_theme


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    apply_theme(app, ThemeMode.LIGHT)
    yield app
