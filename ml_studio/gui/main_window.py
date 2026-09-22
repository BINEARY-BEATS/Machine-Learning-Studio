"""Main application window with production shell."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QAction, QKeySequence
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QMessageBox,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

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
        self._followup_profile = False

        self.setWindowTitle("Machine Learning Studio")
        self.setMinimumSize(900, 600)
        self.resize(1400, 900)
        self._build_ui()
        self._build_menu()
        self._connect_signals()
        self._pages["home"].wire_empty_actions(
            self._new_project, self._open_project, self._import_dataset
        )
        self._apply_theme()
        self.controller.new_project()
        self._sync_project_name()
        self._setup_autosave()
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

        self._shell_splitter = QSplitter(Qt.Orientation.Horizontal)
        self._shell_splitter.setObjectName("ShellSplitter")
        self._shell_splitter.setChildrenCollapsible(False)
        self._shell_splitter.addWidget(self._sidebar)
        self._shell_splitter.addWidget(self._stack)
        self._shell_splitter.setStretchFactor(0, 0)
        self._shell_splitter.setStretchFactor(1, 1)
        self._shell_splitter.setSizes([Sidebar.EXPANDED_WIDTH, 1180])
        self._splitter_sized = False
        body_layout.addWidget(self._shell_splitter, 1)
        layout.addWidget(body, 1)

        self._status = AppStatusBar()
        self.setStatusBar(self._status)
        self._progress = TaskProgressPanel(root)
        self._toast = Toast(self)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if not self._splitter_sized:
            self._splitter_sized = True
            content = max(400, self.width() - Sidebar.EXPANDED_WIDTH - 20)
            self._shell_splitter.setSizes([Sidebar.EXPANDED_WIDTH, content])

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
        self._pages["data"]._empty_import_btn.clicked.connect(self._import_dataset)
        self._pages["data"]._optimize_btn.clicked.connect(self._optimize_memory)
        self._pages["data"]._profile_btn.clicked.connect(self._profile_dataset)
        self._pages["train"]._train_btn.clicked.connect(self._start_training)
        self._pages["train"].train_requested.connect(self._start_training)
        self._pages["models"]._refresh_btn.clicked.connect(
            lambda: self._pages["models"].refresh(self.controller.registry)
        )
        self._pages["models"].predict_requested.connect(self._on_model_predict_requested)
        self._pages["predict"].batch_predict_requested.connect(self._run_batch_predict)
        self._pages["prepare"].preview_requested.connect(self._run_pipeline_preview)
        self._pages["settings"]._theme_cb.currentTextChanged.connect(self._on_settings_theme)
        self._progress.cancel_requested.connect(self._cancel_active_task)

    def _cancel_active_task(self) -> None:
        self.controller.cancel_active_worker()

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
            self._toast.show_message(f"Error loading model: {exc}", variant="danger")

    def _navigate(self, key: str) -> None:
        if key not in self._pages:
            return
        warn = self._nav_gate_message(key)
        if warn:
            self._toast.show_message(warn, variant="warning")
        self._current_page = key
        idx = PAGE_KEYS.index(key)
        self._stack.setCurrentIndex(idx)
        self._sidebar.set_active(key)
        self._top_bar.set_breadcrumb(["ML Studio", BREADCRUMBS.get(key, key.title())])
        self._pages[key].on_show()
        self._status.set_message(f"{BREADCRUMBS.get(key, key)} ready")

    def _nav_gate_message(self, key: str) -> str | None:
        """Soft gate: warn when opening pages without prerequisites."""
        has_data = self.controller.current_dataset is not None
        has_model = self.controller.predictor is not None or bool(
            self.controller.registry.list_models()
        )
        if key in ("prepare", "train") and not has_data:
            return "No dataset loaded yet. Import data on the Data page first."
        if key == "evaluate" and not self._pages["evaluate"]._runs and not has_model:
            return "No experiments yet. Train a model first."
        if key == "predict" and not has_model:
            return "No model loaded. Train or open a model from Models first."
        return None

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
            "run_automl": self._open_train_for_automl,
            "evaluate_model": lambda: self._navigate("evaluate"),
            "predict": lambda: self._navigate("predict"),
            "model_registry": lambda: self._navigate("models"),
            "settings": lambda: self._navigate("settings"),
        }
        handler = handlers.get(cmd)
        if handler:
            handler()

    def _open_train_for_automl(self) -> None:
        """Honest AutoML entry: open Train with Optuna suggested, not a fake runner."""
        self._navigate("train")
        train = self._pages["train"]
        idx = train._tune_combo.findText("Optuna")
        if idx >= 0:
            try:
                import optuna  # noqa: F401

                train._tune_combo.setCurrentIndex(idx)
                train._goto_step(len(train.STEPS) - 2)  # Tune step
                self._toast.show_message(
                    "Optuna tuning selected — review model, then Start Training.",
                    variant="default",
                )
            except ImportError:
                self._toast.show_message(
                    "Optuna not installed. Open Train wizard and use Grid search or None.",
                    variant="warning",
                )
        else:
            self._toast.show_message("Opened Train wizard.", variant="default")

    def _setup_autosave(self) -> None:
        self._autosave_timer = QTimer(self)
        interval = getattr(self.container.config, "autosave_interval_ms", 120_000)
        self._autosave_timer.setInterval(interval)
        self._autosave_timer.timeout.connect(self._autosave_tick)
        self._autosave_timer.start()

    def _autosave_tick(self) -> None:
        if not self._pages["settings"].autosave_enabled():
            return
        pm = self.container.project_manager
        if not pm.current or not pm.current.dirty or not pm.current.path:
            return
        try:
            self.controller.sync_session_to_project(self._pages)
            pm.save_project()
            self._status.set_message("Autosaved")
        except Exception as exc:
            logger.warning("Autosave failed: %s", exc)

    def _new_project(self) -> None:
        self.controller.new_project()
        self.controller.clear_pages(self._pages)
        self._status.update_from_dataset(None)
        self._sync_project_name()
        self._toast.show_message("New project created", variant="success")
        self._pages["home"].refresh_stats(self.controller, self._pages)
        self._navigate("home")

    def _open_project(self) -> None:
        try:
            if self.controller.open_project(self, self._pages):
                self._sync_project_name()
                if self.controller.current_dataset:
                    self._status.update_from_dataset(self.controller.current_dataset)
                    self._followup_profile = True
                    self._profile_dataset(auto=True)
                    self._navigate("data")
                else:
                    self._navigate("home")
                self._toast.show_message("Project opened", variant="success")
                self._pages["home"].refresh_stats(self.controller, self._pages)
        except Exception as exc:
            QMessageBox.critical(self, "Error", str(exc))

    def _save_project(self) -> None:
        try:
            if self.controller.save_project(self, self._pages):
                self._toast.show_message("Project saved", variant="success")
                self._status.set_message("Saved")
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
        self._followup_profile = True

    def _optimize_memory(self) -> None:
        worker = self.controller.build_optimize_worker()
        if worker is None:
            self._toast.show_message("Import a dataset first.", variant="warning")
            return
        self._progress.begin("Optimize memory", "Downcasting dtypes…")
        self.controller.start_worker(
            worker,
            self._on_optimize_complete,
            self._on_worker_progress,
            self._on_worker_error,
            self._on_worker_finished,
        )

    def _on_optimize_complete(self, result) -> None:
        msg = self.controller.apply_optimize_result(result, self._pages)
        self._status.update_from_dataset(self.controller.current_dataset)
        self._toast.show_message(msg, variant="success")
        self._followup_profile = True

    def _profile_dataset(self, auto: bool = False) -> None:
        worker = self.controller.build_profiling_worker()
        if worker is None:
            if not auto:
                self._toast.show_message("Import a dataset first.", variant="warning")
            return
        self._pages["data"].set_loading(True, "Profiling dataset…")
        self._progress.begin("Profiling dataset", "Computing statistics and quality issues…")
        self.controller.start_worker(
            worker,
            self._on_profile_complete,
            self._on_worker_progress,
            self._on_worker_error,
            self._on_worker_finished,
        )

    def _on_profile_complete(self, result) -> None:
        self._pages["data"].set_loading(False)
        self._pages["data"].apply_profile_result(result)
        self._toast.show_message("Profile updated", variant="success")

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
        saved, message = self.controller.on_training_complete(result, self._pages)
        if not saved:
            self._toast.show_message(message, variant="warning")
            QMessageBox.warning(self, "Quality gate", message)
            return
        self._navigate("evaluate")
        self._toast.show_message(message, variant="success")

    def _run_batch_predict(self, path: str) -> None:
        worker = self.controller.build_batch_predict_worker(Path(path))
        if worker is None:
            self._toast.show_message("No model loaded for prediction.", variant="warning")
            return
        self._progress.begin("Batch prediction", Path(path).name)
        self.controller.start_worker(
            worker,
            self._on_batch_predict_complete,
            self._on_worker_progress,
            self._on_worker_error,
            self._on_worker_finished,
        )

    def _on_batch_predict_complete(self, out_path) -> None:
        self._toast.show_message(f"Predictions saved to {out_path}", variant="success")
        QMessageBox.information(self, "Success", f"Batch predictions saved to:\n{out_path}")

    def _run_pipeline_preview(self) -> None:
        prepare = self._pages["prepare"]
        if prepare.dataset is None:
            self._toast.show_message("Load a dataset first to preview.", variant="warning")
            return
        worker = self.controller.build_preview_worker(prepare.pipeline, prepare.dataset)
        self._progress.begin("Pipeline preview", "Sampling and fitting steps…")
        self.controller.start_worker(
            worker,
            self._on_preview_complete,
            self._on_worker_progress,
            self._on_worker_error,
            self._on_worker_finished,
        )

    def _on_preview_complete(self, preview) -> None:
        from ml_studio.gui.dialogs.preview_modal import PreviewModal

        modal = PreviewModal(preview_result=preview, parent=self)
        modal.exec()

    def _on_worker_progress(self, percent: int, message: str) -> None:
        self._progress.update(percent, message)
        self._status.set_message(message)

    def _on_worker_finished(self) -> None:
        self._progress.end()
        self._pages["data"].set_loading(False)
        if self._followup_profile:
            self._followup_profile = False
            self._profile_dataset(auto=True)

    def _on_worker_error(self, msg: str) -> None:
        self._progress.end()
        self._pages["data"].set_loading(False)
        if "cancelled" in msg.lower():
            self._toast.show_message("Task cancelled", variant="warning")
            self._status.set_message("Cancelled")
            return
        self._toast.show_message(f"Task failed: {msg}", variant="danger")
        QMessageBox.critical(self, "Error", msg)
        self._status.set_message("Error")

    def closeEvent(self, event) -> None:
        self.controller.cleanup_worker()
        super().closeEvent(event)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if hasattr(self, "_progress") and self.centralWidget():
            self._progress.setGeometry(self.centralWidget().rect())
