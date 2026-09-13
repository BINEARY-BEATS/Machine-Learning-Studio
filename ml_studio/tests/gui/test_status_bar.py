import pytest
from unittest.mock import MagicMock
from ml_studio.gui.shell.status_bar import AppStatusBar

def test_status_bar(qtbot):
    bar = AppStatusBar()
    qtbot.addWidget(bar)
    
    bar.set_message("Test")
    assert bar._message.text() == "Test"
    
    bar.set_context("Ctx")
    assert bar._context.text() == "Ctx"
    
    bar.set_memory("Mem")
    assert bar._memory.text() == "Mem"
    
    bar.update_from_dataset(None)
    assert bar._context.text() == ""
    assert bar._memory.text() == ""
    
    dataset = MagicMock()
    dataset.name = "MyData"
    dataset.memory_bytes = 1048576 * 2
    bar.update_from_dataset(dataset)
    assert bar._context.text() == "Dataset: MyData"
    assert bar._memory.text() == "Memory: 2.0 MB"
