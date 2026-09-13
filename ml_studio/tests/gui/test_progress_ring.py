import pytest
from unittest.mock import MagicMock
from ml_studio.gui.widgets.progress_ring import ProgressRing
from PyQt6.QtWidgets import QWidget

def test_progress_ring(qtbot):
    ring = ProgressRing(parent=QWidget())
    qtbot.addWidget(ring)
    assert ring is not None
