"""MainWindow action wiring tests against the current shell API."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from ml_studio.app.config import AppConfig
from ml_studio.app.container import AppContainer
from ml_studio.gui.main_window import MainWindow


@pytest.fixture
def container():
    return AppContainer(AppConfig())


@pytest.fixture
def window(container, qtbot):
    win = MainWindow(container)
    qtbot.addWidget(win)
    return win


def test_main_window_open_project(window):
    window.controller.open_project = MagicMock(return_value=True)
    window.controller.current_dataset = None
    window._open_project()
    window.controller.open_project.assert_called_once()


def test_main_window_new_project(window):
    window.controller.new_project = MagicMock()
    window.controller.clear_pages = MagicMock()
    window.controller.current_dataset = None
    window.controller.registry.list_models = MagicMock(return_value=[])
    window._new_project()
    window.controller.new_project.assert_called_once()
    window.controller.clear_pages.assert_called_once()


def test_main_window_save_project(window):
    window.controller.save_project = MagicMock(return_value=True)
    window._save_project()
    window.controller.save_project.assert_called_once()


def test_main_window_shell_splitter_resizable(window):
    assert hasattr(window, "_shell_splitter")
    assert window._shell_splitter.count() == 2
    assert window._shell_splitter.widget(0) is window._sidebar
    assert window._shell_splitter.widget(1) is window._stack
    # Drag-friendly: sidebar is not permanently fixed-width when expanded
    assert window._sidebar.maximumWidth() >= window._sidebar.EXPANDED_WIDTH


def test_main_window_worker_error_shows_toast(window):
    with patch.object(window._toast, "show_message") as toast:
        with patch("ml_studio.gui.main_window.QMessageBox.critical"):
            window._on_worker_error("boom")
            toast.assert_called()
            assert toast.call_args.kwargs.get("variant") == "danger" or (
                len(toast.call_args.args) >= 1 and "boom" in toast.call_args.args[0]
            )


def test_main_window_worker_cancel_toast(window):
    with patch.object(window._toast, "show_message") as toast:
        window._on_worker_error("Task cancelled by user")
        toast.assert_called()
        assert toast.call_args.kwargs.get("variant") == "warning"


def test_main_window_navigation(window):
    window._navigate("prepare")
    assert window._stack.currentWidget() is window._pages["prepare"]
    assert window._current_page == "prepare"

    window._navigate("home")
    assert window._stack.currentWidget() is window._pages["home"]
    assert window._current_page == "home"


def test_main_window_nav_gate_warns_without_data(window):
    window.controller.current_dataset = None
    with patch.object(window._toast, "show_message") as toast:
        window._navigate("train")
        toast.assert_called()
        assert toast.call_args.kwargs.get("variant") == "warning"


def test_main_window_progress_panel(window):
    window._progress.begin("Training", "Fitting…")
    window._on_worker_progress(50, "Halfway")
    window._on_worker_finished()
    assert window._progress is not None


def test_main_window_dataset_loaded_updates_status(window):
    dataset = MagicMock()
    dataset.name = "demo"
    dataset.row_count = 42
    dataset.memory_usage_mb = 1.5
    with patch.object(window.controller, "on_dataset_loaded"):
        with patch.object(window._toast, "show_message"):
            with patch.object(window._status, "update_from_dataset"):
                window._on_dataset_loaded(dataset)
    assert window._followup_profile is True


def test_main_window_command_palette_handlers(window):
    with patch.object(window, "_navigate") as nav:
        window._handle_command("prepare_dataset")
        nav.assert_called_with("prepare")
        window._handle_command("train_model")
        nav.assert_called_with("train")
