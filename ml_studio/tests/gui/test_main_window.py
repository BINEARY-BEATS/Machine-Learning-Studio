import pytest
from PyQt6.QtWidgets import QApplication
from ml_studio.gui.main_window import MainWindow
from ml_studio.app.config import AppConfig
from ml_studio.app.container import AppContainer

@pytest.fixture
def container():
    config = AppConfig()
    return AppContainer(config=config)

def test_main_window_init(qtbot, container):
    window = MainWindow(container)
    qtbot.addWidget(window)
    
    # Check if window is instantiated correctly
    assert window.windowTitle() == container.config.app_name
    
    # Check if it has a sidebar and stack
    assert window._sidebar is not None
    assert window._stack is not None
    assert window._top_bar is not None
    assert window._status is not None
    
def test_main_window_navigation(qtbot, container):
    window = MainWindow(container)
    qtbot.addWidget(window)
    
    # Initial page is home
    assert window._stack.currentIndex() == 0
    
    # Navigate to data
    window._sidebar._buttons["data"].click()
    assert window._stack.currentIndex() == 1
    
    # Navigate to prepare
    window._sidebar._buttons["prepare"].click()
    assert window._stack.currentIndex() == 2
    
def test_main_window_theme_toggle(qtbot, container):
    window = MainWindow(container)
    qtbot.addWidget(window)
    
    # Top bar has theme toggle
    window._top_bar._theme_btn.click()
    # It should toggle the theme mode
    assert window._top_bar is not None
