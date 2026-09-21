import pytest
import time
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import QApplication
from unittest.mock import patch

from ml_studio.gui.app_controller import AppController
from ml_studio.gui.main_window import MainWindow
from ml_studio.gui.widgets.command_palette import CommandPalette
from ml_studio.app.container import AppContainer
from ml_studio.app.config import AppConfig

@pytest.fixture
def container():
    return AppContainer(config=AppConfig())

def test_app_launches_without_error_and_fast(qtbot, container):
    start_time = time.time()
    
    main_win = MainWindow(container)
    qtbot.addWidget(main_win)
    main_win.show()
    
    qtbot.waitExposed(main_win)
    elapsed = time.time() - start_time
    
    assert elapsed < 5.0, f"App took too long to start: {elapsed:.2f}s"
    
    # Save a screenshot to prove it ran
    pixmap = main_win.grab()
    pixmap.save("app_running.png")


def test_navigate_home_to_data_to_home(qtbot, container):
    main_win = MainWindow(container)
    qtbot.addWidget(main_win)
    main_win.show()
    qtbot.waitExposed(main_win)
    
    # Verify we start at Home (index 0)
    assert main_win._stack.currentIndex() == 0
    
    # Click Data
    qtbot.mouseClick(main_win._sidebar._buttons["data"], Qt.MouseButton.LeftButton)
    assert main_win._stack.currentIndex() == 1
    
    # Click Home
    qtbot.mouseClick(main_win._sidebar._buttons["home"], Qt.MouseButton.LeftButton)
    assert main_win._stack.currentIndex() == 0


def test_open_project_updates_ui(qtbot, container):
    main_win = MainWindow(container)
    qtbot.addWidget(main_win)
    
    with patch("ml_studio.gui.app_controller.QFileDialog.getOpenFileName", return_value=("dummy/path", "")):
        with patch.object(main_win.controller.container.project_manager, "open_project", return_value=True):
            with patch("ml_studio.gui.main_window.QMessageBox.critical"):
                with patch.object(main_win.controller, "open_project", return_value=True):
                    with patch.object(main_win._pages["home"], "refresh_stats") as mock_refresh:
                        main_win._open_project()
    
    # Opening a project successfully should refresh home stats
    assert mock_refresh.call_count >= 1


def test_theme_toggle_updates_stylesheet(qtbot, container):
    main_win = MainWindow(container)
    qtbot.addWidget(main_win)
    
    initial_theme = main_win._theme
    
    # The theme toggle button is in top_bar
    qtbot.mouseClick(main_win._top_bar._theme_btn, Qt.MouseButton.LeftButton)
    
    # Verify theme changed
    new_theme = main_win._theme
    assert new_theme != initial_theme
    
    # Wait a tiny bit for Qt to apply stylesheet to qApp
    QApplication.processEvents()
    assert QApplication.instance().styleSheet() != ""


def test_command_palette_toggle_integration(qtbot, container):
    main_win = MainWindow(container)
    qtbot.addWidget(main_win)
    # Trigger command palette slot directly since QShortcut can be flaky in headless tests
    with patch("ml_studio.gui.widgets.command_palette.CommandPalette.exec") as mock_exec:
        if hasattr(main_win, "show_command_palette"):
            main_win.show_command_palette()
        elif hasattr(main_win, "_show_command_palette"):
            main_win._show_command_palette()
        else:
            qtbot.keySequence(main_win, "Ctrl+K")
            
        mock_exec.assert_called_once()
