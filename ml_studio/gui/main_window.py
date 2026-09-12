"""Main application window with production shell."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction, QKeySequence
from PyQt6.QtWidgets import QApplication, QHBoxLayout, QMainWindow, QMessageBox, QStackedWidget, QVBoxLayout, QWidget

from ml_studio.app.container import AppContainer
from ml_studio.app.logger import get_logger
from ml_studio.app.paths import ensure_runtime_dirs
from ml_studio.app.theme import ThemeMode, apply_theme, write_theme_files
from ml_studio.gui.app_controller import AppController
from ml_studio.gui.dialogs.import_preview import ImportPreviewDialog
from ml_studio.gui.pages.data_page import DataPage
from ml_studio.gui.pages.evaluate_page import EvaluatePage
from ml_studio.gui.pages.home import HomePage
from ml_studio.gui.pages.models_page import ModelsPage
from ml_studio.gui.pages.predict_page import PredictPage
from ml_studio.gui.pages.prepare_page import PreparePage
from ml_studio.gui.pages.settings_page import SettingsPage
from ml_studio.gui.pages.train_page import TrainPage
from ml_studio.gui.shell import AppStatusBar, Sidebar, TopBar
from ml_studio.gui.shell.navigation import NAV_ITEMS, PAGE_KEYS
from ml_studio.gui.widgets.command_palette import CommandPalette
from ml_studio.gui.widgets.task_progress import TaskProgressPanel
from ml_studio.gui.widgets.toast import Toast

logger = get_logger("main_window")

BREADCRUMBS = {item.key: item.label for item in NAV_ITEMS}


class MainWindow(QMainWindow):
    def __init__(self, container: AppContainer):
        super().__init__()
        self.container = container
        ensure_runtime_dirs()
        write_theme_files()
        self.controller = AppController(container)
        self._theme = ThemeMode.LIGHT
        self._current_page = "home"

        self.setWindowTitle("Machine Learning Studio")
        self.setMinimumSize(1200, 800)
        self.resize(1400, 900)
        self._build_ui()
        self._build_menu()
        self._connect_signals()
        self._apply_theme()
        self.controller.new_project()
        self._sync_project_name()
        self._navigate("home")

    def _build_ui(self) -> None:
        root = QWidget()
        root.setObjectName("CentralWidget")
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._top_bar = TopBar()
        layout.addWidget(self._top_bar)

        body = QWidget()
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(0)

        content_row = QWidget()
        row_layout = QVBoxLayout(content_row)
        row_layout.setContentsMargins(0, 0, 0, 0)

        h = QHBoxLayout()
        h.setSpacing(0)
        self._sidebar = Sidebar()
        self._stack = QStackedWidget()
        self._pages = {
            "home": HomePage(self.container),
            "data": DataPage(self.container),
            "prepare": PreparePage(self.container),
            "train": TrainPage(self.container),
            "evaluate": EvaluatePage(self.container),
            "predict": PredictPage(self.container),
            "models": ModelsPage(self.container),
            "settings": SettingsPage(self.container),
        }
        for key in PAGE_KEYS:
            self._stack.addWidget(self._pages[key])
        h.addWidget(self._sidebar)
        h.addWidget(self._stack, 1)
        row_layout.addLayout(h)
        body_layout.addWidget(content_row, 1)
        layout.addWidget(body, 1)

        self._status = AppStatusBar()
        self.setStatusBar(self._status)
        self._progress = TaskProgressPanel(root)
        self._toast = Toast(self)

    def _build_menu(self) -> None:
        save_action = QAction("Save", self)
        save_action.setShortcut(QKeySequence.StandardKey.Save)
        save_action.triggered.connect(self._save_project)
        self.addAction(save_action)
        palette_action = QAction("Command Palette", self)
        palette_action.setShortcut(QKeySequence("Ctrl+K"))
        palette_action.triggered.connect(self._show_command_palette)
        self.addAction(palette_action)

    def _connect_signals(self) -> None:
        self._sidebar.navigate.connect(self._navigate)
        self._top_bar.command_palette_requested.connect(self._show_command_palette)
        self._top_bar.theme_toggle_requested.connect(self._toggle_theme)
        self._top_bar.project_renamed.connect(self._rename_project)
        self._pages["home"]._new_btn.clicked.connect(self._new_project)
        self._pages["home"]._open_btn.clicked.connect(self._open_project)
        self._pages["home"]._import_btn.clicked.connect(self._import_dataset)
        self._pages["data"]._import_btn.clicked.connect(self._import_dataset)
        self._pages["data"]._optimize_btn.clicked.connect(self._optimize_memory)
        self._pages["data"]._profile_btn.clicked.connect(self._profile_dataset)
        self._pages["train"]._train_btn.clicked.connect(self._start_training)
        self._pages["models"]._refresh_btn.clicked.connect(
            lambda: self._pages["models"].refresh(self.controller.registry)
        )
        self._pages["models"].predict_requested.connect(self._on_model_predict_requested)
        self._pages["settings"]._theme_cb.currentTextChanged.connect(self._on_settings_theme)
        self._progress.cancel_requested.connect(self._cancel_training)

    def _cancel_training(self):
        if self.controller._active_worker:
            self.controller._active_worker.cancel()
        if self.controller._active_thread:
            self.controller._active_thread.requestInterruption()

    def _on_model_predict_requested(self, model_id: str) -> None:
        try:
            from ml_studio.core.inference.predictor import Predictor
            mv = self.controller.registry.get(model_id)
            if not mv:
                return
            pipeline = self.controller.registry.load_pipeline(model_id)
            self.controller.predictor = Predictor(pipeline)
            self._pages["predict"].bind_predictor(self.controller.predictor, pipeline.feature_columns, mv.task)
            self._navigate("predict")
            self._toast.show_message("Model loaded for prediction", variant="success")
        except Exception as exc:
            self._toast.show_message(f"Error loading model: {exc}", variant="error")

    def _navigate(self, key: str) -> None:
        if key not in self._pages:
            return
        self._current_page = key
        idx = PAGE_KEYS.index(key)
        self._stack.setCurrentIndex(idx)
        self._sidebar.set_active(key)
        self._top_bar.set_breadcrumb(["ML Studio", BREADCRUMBS.get(key, key.title())])
        self._pages[key].on_show()
        self._status.set_message(f"{BREADCRUMBS.get(key, key)} ready")

    def _apply_theme(self) -> None:
        app = QApplication.instance()
        if app:
            apply_theme(app, self._theme)
        self._sidebar.set_theme_mode(self._theme)
        self._top_bar.set_theme_mode(self._theme)
        for page in self._pages.values():
            if hasattr(page, "set_theme_mode"):
                page.set_theme_mode(self._theme)

    def _toggle_theme(self) -> None:
        self._theme = ThemeMode.DARK if self._theme == ThemeMode.LIGHT else ThemeMode.LIGHT
        self._pages["settings"].set_theme_selection(self._theme)
        self._apply_theme()

    def _on_settings_theme(self, _text: str) -> None:
        self._theme = self._pages["settings"].selected_theme()
        self._apply_theme()

    def _sync_project_name(self) -> None:
        pm = self.container.project_manager
        name = pm.current.name if pm.current else "Untitled Project"
        self._top_bar.set_project_name(name)

    def _rename_project(self, name: str) -> None:
        pm = self.container.project_manager
        if pm.current and name:
            pm.current.name = name
            self._pages["home"].refresh_stats(self.controller, self._pages)

    def _show_command_palette(self) -> None:
        palette = CommandPalette(self, self._theme)
        palette.command_selected.connect(self._handle_command)
        palette.exec()

    def _handle_command(self, cmd: str) -> None:
        handlers = {
            "new_project": self._new_project,
            "open_project": self._open_project,
            "save": self._save_project,
            "import_dataset": self._import_dataset,
            "profile_dataset": self._profile_dataset,
            "prepare_dataset": lambda: self._navigate("prepare"),
            "train_model": lambda: self._navigate("train"),
            "run_automl": lambda: self._navigate("train"),
            "evaluate_model": lambda: self._navigate("evaluate"),
            "predict": lambda: self._navigate("predict"),
            "model_registry": lambda: self._navigate("models"),
            "settings": lambda: self._navigate("settings"),
        }
        handler = handlers.get(cmd)
        if handler:
            handler()

    def _new_project(self) -> None:
        self.controller.new_project()
        self._sync_project_name()
        self._toast.show_message("New project created", variant="success")
        self._pages["home"].refresh_stats(self.controller, self._pages)

    def _open_project(self) -> None:
        try:
            if self.controller.open_project(self):
                self._sync_project_name()
                self._toast.show_message("Project opened", variant="success")
                self._pages["home"].refresh_stats(self.controller, self._pages)
        except Exception as exc:
            QMessageBox.critical(self, "Error", str(exc))

    def _save_project(self) -> None:
        try:
            if self.controller.save_project(self):
                self._toast.show_message("Project saved", variant="success")
        except Exception as exc:
            QMessageBox.critical(self, "Error", str(exc))

    def _import_dataset(self) -> None:
        path = self.controller.pick_dataset_path(self)
        if not path:
            return
        dialog = ImportPreviewDialog(path, self)
        if dialog.exec() != dialog.DialogCode.Accepted:
            return
        self._pages["data"].set_loading(True)
        self.controller.load_dataset(
            path,
            self._on_dataset_loaded,
            self._on_worker_progress,
            self._on_worker_error,
            self._on_worker_finished,
        )

    def _on_dataset_loaded(self, dataset) -> None:
        self._pages["data"].set_loading(False)
        self.controller.on_dataset_loaded(dataset, self._pages)
        self._status.update_from_dataset(dataset)
        self._navigate("data")
        self._toast.show_message(f"Imported {dataset.name} ({dataset.row_count:,} rows)", variant="success")

    def _optimize_memory(self) -> None:
        msg = self.controller.optimize_memory(self._pages)
        if msg:
            self._toast.show_message(msg, variant="success")

    def _profile_dataset(self) -> None:
        if self.controller.current_dataset:
            self._pages["data"]._populate_profile()
            self._toast.show_message("Profile updated")

    def _start_training(self) -> None:
        if not self.controller.validate_training(self._pages, self):
            return
        worker = self.controller.build_training_worker(self._pages)
        if worker is None:
            return
        train_page = self._pages["train"]
        self.controller.start_worker(
            worker,
            self._on_training_complete,
            self._on_worker_progress,
            self._on_worker_error,
            self._on_worker_finished,
        )
        self._progress.begin(
            f"Training {train_page.get_model_name()}",
            f"{train_page.get_task().value} · target {train_page.get_target_column()}",
        )

    def _on_training_complete(self, result) -> None:
        saved = self.controller.on_training_complete(result, self._pages)
        if not saved:
            self._toast.show_message("Model performs worse than baseline. Not saved.", variant="warning")
            return
        self._navigate("evaluate")
        self._toast.show_message("Training complete", variant="success")

    def _on_worker_progress(self, percent: int, message: str) -> None:
        self._progress.update(percent, message)
        self._status.set_message(message)

    def _on_worker_finished(self) -> None:
        self._progress.end()

    def _on_worker_error(self, msg: str) -> None:
        self._progress.end()
        self._pages["data"].set_loading(False)
        if "cancelled" in msg.lower():
            self._toast.show_message("Training cancelled by user", variant="warning")
        else:
            self._pages["data"].show_error(msg)
            self._toast.show_message(f"Training failed: {msg}", variant="error")
        QMessageBox.critical(self, "Error", msg)
        self._status.set_message("Error")

    def closeEvent(self, event) -> None:
        self.controller.cleanup_worker()
        super().closeEvent(event)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if hasattr(self, "_progress") and self.centralWidget():
            self._progress.setGeometry(self.centralWidget().rect())
