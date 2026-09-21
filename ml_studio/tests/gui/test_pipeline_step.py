import pytest
from ml_studio.gui.widgets.pipeline_step import PipelineStepWidget
from ml_studio.transforms.registry import get as get_transform

def test_pipeline_step_init(qtbot):
    Impute = get_transform("Impute")
    step = Impute(strategy="mean")
    widget = PipelineStepWidget(step, 0, 3)
    qtbot.addWidget(widget)
    
    assert widget._up_btn.button().isEnabled() is False
    assert widget._down_btn.button().isEnabled() is True
    assert widget._enabled.isChecked() is True

def test_pipeline_step_signals(qtbot):
    Impute = get_transform("Impute")
    step = Impute(strategy="mean")
    widget = PipelineStepWidget(step, 1, 3)
    qtbot.addWidget(widget)
    
    with qtbot.waitSignal(widget.move_up_requested, timeout=1000):
        widget._up_btn.button().click()
        
    with qtbot.waitSignal(widget.move_down_requested, timeout=1000):
        widget._down_btn.button().click()
        
    with qtbot.waitSignal(widget.edit_requested, timeout=1000):
        widget._edit_btn.button().click()
        
    with qtbot.waitSignal(widget.delete_requested, timeout=1000):
        widget._delete_btn.button().click()
        
    with qtbot.waitSignal(widget.toggle_requested, timeout=1000) as blocker:
        widget._enabled.setChecked(False)
    assert blocker.args == [False]
