import pytest
from unittest.mock import MagicMock
from ml_studio.gui.shell.sidebar import Sidebar

def test_sidebar(qtbot):
    sidebar = Sidebar()
    qtbot.addWidget(sidebar)
    
    sidebar.set_active("home")
    
    sidebar.toggle_collapse()
    assert sidebar._collapsed
    
    sidebar.toggle_collapse()
    assert not sidebar._collapsed
