"""Background task and project actions for the main window."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtWidgets import QFileDialog, QMessageBox

from ml_studio.app.container import AppContainer
from ml_studio.app.logger import get_logger
from ml_studio.core.persistence.model_registry import ModelRegistry
from ml_studio.core.profiling import optimize_dtypes
from ml_studio.core.training.data_prep import prepare_for_training, suggest_training_columns
from ml_studio.core.training.task import TaskType
from ml_studio.core.training.trainer import TrainingConfig
from ml_studio.gui.workers.dataset_worker import DatasetLoadWorker
from ml_studio.gui.workers.training_worker import TrainingWorker

logger = get_logger("app_controller")


class AppController:
    """Coordinates project, dataset, and training workflows."""

    def __init__(self, container: AppContainer, registry_dir: Path | None = None) -> None:
        self.container = container
        self.registry = ModelRegistry(registry_dir or Path.home() / ".mlstudio" / "models")
        self.current_dataset = None
        self.current_result = None
        self.predictor = None
        self._active_worker = None
        self._active_thread = None

    def cleanup_worker(self) -> None:
        self._active_worker = None
        if self._active_thread is not None:
            if self._active_thread.isRunning():
                self._active_thread.quit()
                self._active_thread.wait(3000)
            self._active_thread = None

    def start_worker(self, worker, on_result, progress_cb, error_cb, finished_cb) -> None:
        self.cleanup_worker()
        self._active_worker = worker
        worker.progress.connect(progress_cb)
        worker.result.connect(on_result)
        worker.error.connect(error_cb)
        worker.finished.connect(finished_cb)
        self._active_thread = worker.run_in_thread()

    def new_project(self) -> None:
        self.container.project_manager.new_project()

    def open_project(self, parent) -> bool:
        path, _ = QFileDialog.getOpenFileName(
            parent, "Open Project", str(Path.home()), "ML Studio Project (*.mlstudio)"
        )
        if not path:
            return False
        self.container.project_manager.open_project(Path(path))
        return True

    def save_project(self, parent) -> bool:
        pm = self.container.project_manager
        if not pm.current:
            return False
        if pm.current.path:
            pm.save_project()
            return True
        path, _ = QFileDialog.getSaveFileName(
            parent, "Save Project", str(Path.home()), "ML Studio Project (*.mlstudio)"
        )
        if not path:
            return False
        pm.save_project(Path(path))
        return True

    def pick_dataset_path(self, parent) -> Path | None:
        path, _ = QFileDialog.getOpenFileName(
            parent,
            "Import Dataset",
            str(Path.home()),
            "Data Files (*.csv *.xlsx *.xls *.json *.parquet *.feather *.arrow *.orc *.db *.sqlite)",
        )
        return Path(path) if path else None

    def load_dataset(self, path: Path, on_loaded, progress_cb, error_cb, finished_cb) -> None:
        worker = DatasetLoadWorker(path)
        self.start_worker(worker, on_loaded, progress_cb, error_cb, finished_cb)

    def on_dataset_loaded(self, dataset, pages) -> None:
        self.current_dataset = dataset
        pages["data"].set_dataset(dataset)
        try:
            target, _features, task = suggest_training_columns(dataset.dataframe)
            pages["train"].set_dataset(
                dataset.name,
                list(dataset.dataframe.columns),
                target,
                task,
            )
        except ValueError:
            pages["train"].set_dataset(
                dataset.name,
                list(dataset.dataframe.columns),
                str(dataset.dataframe.columns[-1]),
                TaskType.REGRESSION,
            )
        pages["home"].refresh_stats(self, pages)

    def optimize_memory(self, pages) -> str | None:
        if not self.current_dataset:
            return None
        optimized, report = optimize_dtypes(self.current_dataset.dataframe)
        self.current_dataset.set_dataframe(optimized, reason="optimize_dtypes")
        pages["data"].set_dataset(self.current_dataset)
        return f"Memory reduced by {report['reduction_pct']:.1f}%"

    def build_training_worker(self, pages) -> TrainingWorker | None:
        if not self.current_dataset:
            return None
        train_page = pages["train"]
        config = train_page.build_config()
        if config is None:
            return None
        prepared_df, target, features, _ = prepare_for_training(
            self.current_dataset.dataframe,
            config.task,
            target_column=config.target_column,
            feature_columns=config.feature_columns or None,
        )
        config.target_column = target
        config.feature_columns = features
        preprocessing = pages["prepare"].pipeline if pages["prepare"].pipeline.nodes else None
        return TrainingWorker(prepared_df, config, preprocessing)

    def on_training_complete(self, result, pages) -> bool:
        from ml_studio.core.training.registry import MODEL_REGISTRY

        if result.task == TaskType.REGRESSION and result.metrics.get("r2", 0) < 0:
            return False
        if result.task == TaskType.CLASSIFICATION:
            acc = result.metrics.get("accuracy", 0)
            baseline = result.metrics.get("baseline_accuracy", 0)
            if acc < baseline:
                return False

        self.current_result = result
        meta = MODEL_REGISTRY.get(result.model_id)
        model_display = meta.name if meta else result.model_id
        dataset_name = self.current_dataset.name if self.current_dataset else "Unknown"
        mv = self.registry.register(result, name=model_display)
        pages["evaluate"].add_run(
            result,
            model_display_name=model_display,
            dataset_name=dataset_name,
            registry_id=mv.model_id,
        )
        pages["models"].refresh(self.registry)
        pages["home"].refresh_stats(self, pages)
        try:
            from ml_studio.core.inference.predictor import Predictor

            self.predictor = Predictor(self.registry.load_pipeline(mv.model_id))
            pages["predict"].bind_predictor(self.predictor, result.feature_columns, result.task)
        except Exception as exc:
            logger.warning("Could not load predictor: %s", exc)
        return True

    def validate_training(self, pages, parent) -> bool:
        if not self.current_dataset:
            QMessageBox.warning(parent, "Warning", "Import a dataset first.")
            return False
        config = pages["train"].build_config()
        if config is None:
            QMessageBox.warning(parent, "Warning", "Complete the training wizard steps.")
            return False
        return True
