import pytest
from PyQt6.QtCore import Qt
from ml_studio.gui.dialogs.transform_picker import TransformPickerDialog
from ml_studio.gui.dialogs.step_config import StepConfigDialog, MultiSelectWidget

def test_transform_picker(qtbot):
    dialog = TransformPickerDialog()
    qtbot.addWidget(dialog)
    
    assert dialog._list.count() > 0
    dialog._search.setText("impute")
    
    # At least one item should be visible, others hidden
    visible = sum(1 for i in range(dialog._list.count()) if not dialog._list.item(i).isHidden())
    assert visible > 0
    
    # Select the first visible one
    for i in range(dialog._list.count()):
        if not dialog._list.item(i).isHidden():
            dialog._list.setCurrentRow(i)
            break
            
    dialog._on_accept()
    assert dialog.selected_transform is not None

def test_multi_select_widget(qtbot):
    w = MultiSelectWidget(["a", "b", "c"])
    qtbot.addWidget(w)
    w.set_value(["a", "c"])
    assert w.get_value() == ["a", "c"]

def test_step_config_impute(qtbot):
    dialog = StepConfigDialog("Impute", {"strategy": "mean"}, ["a", "b"])
    qtbot.addWidget(dialog)
    
    # Should populate fields
    assert "strategy" in dialog._fields
    assert "columns" in dialog._fields
    
    # Modify a field
    dialog._fields["strategy"].setCurrentText("median")
    
    # Save
    dialog._on_save()
    assert dialog.final_params["strategy"] == "median"

def test_step_config_validation(qtbot):
    dialog = StepConfigDialog("Winsorize", {}, ["a", "b"])
    qtbot.addWidget(dialog)
    
    # Set invalid json to array field
    dialog._fields["limits"].setText("not json")
    dialog._on_save()
    
    # Should not close, error label should show
    assert not dialog._error_label.isHidden()
    assert "Invalid JSON" in dialog._error_label.text()
