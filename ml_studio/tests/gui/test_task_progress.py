import pytest
from unittest.mock import MagicMock
from ml_studio.gui.widgets.task_progress import TaskProgressPanel

def test_task_progress(qtbot):
    progress = TaskProgressPanel()
    qtbot.addWidget(progress)
    
    progress.begin("Training Model", "Subtitle")
    assert progress._title.text() == "Training Model"
    
    progress.update(50, "Halfway there")
    assert progress._bar.value() == 50
    assert progress._step.text() == "Halfway there"
    
    progress.end()
    
    progress._on_cancel()
    assert not progress._cancel_btn.isEnabled()
