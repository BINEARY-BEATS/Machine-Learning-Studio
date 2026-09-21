import pytest
import pandas as pd
from PyQt6.QtWidgets import QLabel
from ml_studio.core.pipeline import Pipeline
from ml_studio.core.dataset import Dataset
from ml_studio.transforms.registry import get as get_transform
from ml_studio.gui.dialogs.preview_modal import PreviewModal

def test_preview_modal(qtbot):
    pipeline = Pipeline()
    OneHot = get_transform("OneHot")
    pipeline.add(OneHot(columns=["b"]))
    
    df = pd.DataFrame({'a': [1.0, 2.0, 3.0], 'b': ['x', 'y', 'z']})
    dataset = Dataset(_dataframe=df, name="test")
    
    modal = PreviewModal(pipeline, dataset)
    qtbot.addWidget(modal)
    
    # Check if error label is shown
    err = None
    for lbl in modal.findChildren(QLabel):
        if lbl.objectName() == "DangerText":
            err = lbl
            break
            
    if err is not None:
        print("ERROR IN PREVIEW:", err.text())
        assert False, err.text()

def test_preview_modal_error(qtbot):
    pipeline = Pipeline()
    OneHot = get_transform("OneHot")
    pipeline.add(OneHot(columns=["MISSING"]))
    
    df = pd.DataFrame({'a': [1.0, 2.0, 3.0]})
    dataset = Dataset(_dataframe=df, name="test")
    
    modal = PreviewModal(pipeline, dataset)
    qtbot.addWidget(modal)
    
    err = None
    for lbl in modal.findChildren(QLabel):
        if lbl.objectName() == "DangerText":
            err = lbl
            break
            
    assert err is not None
