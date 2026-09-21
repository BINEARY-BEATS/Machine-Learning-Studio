"""Background task and project actions for the main window."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtWidgets import QFileDialog, QMessageBox

from ml_studio.app.container import AppContainer
from ml_studio.app.logger import get_logger
from ml_studio.core.persistence.model_registry import ModelRegistry
from ml_studio.core.pipeline import Pipeline
from ml_studio.core.training.data_prep import suggest_training_columns
from ml_studio.core.training.task import TaskType
from ml_studio.gui.workers.batch_predict_worker import BatchPredictWorker
from ml_studio.gui.workers.dataset_worker import DatasetLoadWorker
from ml_studio.gui.workers.optimize_worker import OptimizeWorker
from ml_studio.gui.workers.preview_worker import PreviewWorker
from ml_studio.gui.workers.profiling_worker import ProfilingWorker
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

    def cancel_active_worker(self) -> None:
        if self._active_worker is not None:
            self._active_worker.cancel()
        if self._active_thread is not None and self._active_thread.isRunning():
            self._active_thread.requestInterruption()

    def new_project(self) -> None:
        self.container.project_manager.new_project()
        self.current_dataset = None
        self.current_result = None
        self.predictor = None

    def clear_pages(self, pages) -> None:
        if "data" in pages:
            pages["data"].set_dataset(None)
        if "prepare" in pages:
            pages["prepare"].set_dataset(None)
            pages["prepare"].pipeline = Pipeline()
            pages["prepare"].schema_overrides = {}
            pages["prepare"]._refresh_ui()
        if "home" in pages:
            pages["home"].refresh_stats(self, pages)

    def open_project(self, parent, pages=None) -> bool:
        path, _ = QFileDialog.getOpenFileName(
            parent, "Open Project", str(Path.home()), "ML Studio Project (*.mlstudio)"
        )
        if not path:
            return False
        project = self.container.project_manager.open_project(Path(path))
        if pages is not None:
            self.hydrate_from_project(project, pages)
        return True

    def save_project(self, parent, pages=None) -> bool:
        pm = self.container.project_manager
        if not pm.current:
            return False
        if pages is not None:
            self.sync_session_to_project(pages)
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

    def sync_session_to_project(self, pages) -> None:
        """Copy live controller/page state onto Project before save."""
        pm = self.container.project_manager
        if not pm.current:
            return
        project = pm.current
        project.dataset = self.current_dataset
        prepare = pages.get("prepare")
        if prepare is not None:
            project.pipeline = prepare.pipeline
            project.schema_overrides = dict(prepare.schema_overrides)
        project.mark_dirty()

    def hydrate_from_project(self, project, pages) -> None:
        """Restore UI + controller state from an opened Project."""
        self.current_dataset = project.dataset
        self.current_result = None

        if project.dataset is not None:
            self.on_dataset_loaded(project.dataset, pages)
            prepare = pages.get("prepare")
            if prepare is not None:
                prepare.pipeline = project.pipeline or Pipeline()
                prepare.schema_overrides = dict(project.schema_overrides or {})
                prepare.set_dataset(project.dataset, prepare.schema_overrides)
                prepare._refresh_ui()
        else:
            self.clear_pages(pages)
            prepare = pages.get("prepare")
            if prepare is not None and project.pipeline is not None:
                prepare.pipeline = project.pipeline
                prepare.schema_overrides = dict(project.schema_overrides or {})
                prepare._refresh_ui()
        project.mark_clean()

    def mark_project_dirty(self) -> None:
        pm = self.container.project_manager
        if pm.current:
            pm.current.mark_dirty()

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
        self.mark_project_dirty()
        pages["data"].set_dataset(dataset)
        if "prepare" in pages:
            pages["prepare"].set_dataset(dataset)
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

    def build_profiling_worker(self) -> ProfilingWorker | None:
        if not self.current_dataset:
            return None
        return ProfilingWorker(self.current_dataset)

    def build_optimize_worker(self) -> OptimizeWorker | None:
        if not self.current_dataset:
            return None
        return OptimizeWorker(self.current_dataset.dataframe)

    def apply_optimize_result(self, result: dict, pages) -> str:
        optimized = result["dataframe"]
        report = result["report"]
        self.current_dataset.set_dataframe(optimized, reason="optimize_dtypes")
        self.mark_project_dirty()
        pages["data"].set_dataset(self.current_dataset)
        if "prepare" in pages:
            pages["prepare"].set_dataset(self.current_dataset)
        return f"Memory reduced by {report['reduction_pct']:.1f}%"

    def build_training_worker(self, pages) -> TrainingWorker | None:
        if not self.current_dataset:
            return None
        train_page = pages["train"]
        config = train_page.build_config()
        if config is None:
            return None
        prepare = pages.get("prepare")
        preprocessing = None
        if prepare is not None and prepare.pipeline.steps:
            active = [s for s in prepare.pipeline.steps if getattr(s, "enabled", True)]
            if active:
                preprocessing = Pipeline(steps=list(active))
        # Raw dataframe — preparation runs inside TrainingWorker
        return TrainingWorker(
            self.current_dataset.dataframe,
            config,
            preprocessing,
            prepare_data=True,
        )

    def build_batch_predict_worker(self, path: Path) -> BatchPredictWorker | None:
        if not self.predictor:
            return None
        return BatchPredictWorker(self.predictor, path)

    def build_preview_worker(self, pipeline, dataset) -> PreviewWorker:
        return PreviewWorker(pipeline, dataset)

    def on_training_complete(self, result, pages) -> tuple[bool, str]:
        from ml_studio.core.training.registry import MODEL_REGISTRY

        if result.task == TaskType.REGRESSION:
            r2 = result.metrics.get("r2", 0)
            if r2 is not None and r2 < 0:
                return (
                    False,
                    f"Quality gate failed: R²={r2:.3f} is below 0 "
                    "(worse than predicting the mean). Model was not saved.",
                )
        if result.task == TaskType.CLASSIFICATION:
            acc = result.metrics.get("accuracy", 0)
            baseline = result.metrics.get("baseline_accuracy", 0)
            if acc is not None and baseline is not None and acc < baseline:
                return (
                    False,
                    f"Quality gate failed: accuracy {acc:.3f} is below "
                    f"baseline {baseline:.3f}. Model was not saved.",
                )

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

        primary = (
            result.metrics.get("r2")
            or result.metrics.get("f1")
            or result.metrics.get("accuracy")
            or result.metrics.get("silhouette")
        )
        score_txt = f"{primary:.3f}" if isinstance(primary, float) else "done"
        return True, f"Training complete — {model_display} (score {score_txt})"

    def validate_training(self, pages, parent) -> bool:
        if not self.current_dataset:
            QMessageBox.warning(parent, "Warning", "Import a dataset first.")
            return False
        train = pages["train"]
        err = train.validate_step(len(train.STEPS) - 1) if hasattr(train, "validate_step") else None
        if err:
            QMessageBox.warning(parent, "Complete the wizard", err)
            return False
        config = train.build_config()
        if config is None:
            QMessageBox.warning(parent, "Warning", "Complete the training wizard steps.")
            return False
        return True
