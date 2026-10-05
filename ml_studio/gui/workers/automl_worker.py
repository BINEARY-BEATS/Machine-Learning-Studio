"""AutoML worker — runs leaderboard off the UI thread."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from ml_studio.core.training.automl import AutoMLRunner
from ml_studio.core.training.data_prep import EncodingBundle
from ml_studio.core.training.trainer import TrainingConfig
from ml_studio.gui.workers.base_worker import WorkerBase


class AutoMLWorker(WorkerBase):
    """Prepare a train/val split and run AutoMLRunner with live progress."""

    def __init__(
        self,
        df: pd.DataFrame,
        config: TrainingConfig,
        *,
        max_models: int = 5,
        max_runtime_seconds: float = 180,
    ):
        super().__init__()
        self.df = df
        self.config = config
        self.max_models = max_models
        self.max_runtime_seconds = max_runtime_seconds
        self._runner: AutoMLRunner | None = None

    def cancel(self) -> None:
        super().cancel()
        if self._runner is not None:
            self._runner.cancel()

    def do_work(self):
        config = self.config
        df = self.df
        self.progress.emit(5, "Preparing AutoML split…")
        if self.is_cancelled:
            raise RuntimeError("AutoML cancelled")

        # Cap rows so CV folds finish quickly and Cancel can interrupt sooner.
        max_rows = 3_000
        if len(df) > max_rows:
            df = df.sample(n=max_rows, random_state=config.random_state).reset_index(
                drop=True
            )
            self.progress.emit(
                8,
                f"AutoML using a {max_rows:,}-row sample for speed…",
            )

        X = df[config.feature_columns]
        y = df[config.target_column]
        X_tr, X_va, y_tr, y_va = train_test_split(
            X,
            y,
            test_size=config.test_size,
            random_state=config.random_state,
        )
        self.progress.emit(12, "Encoding features for AutoML (train split only)…")
        enc = EncodingBundle().fit(X_tr, y_tr, config.task, config.target_column)
        X_tr_e = enc.transform_features(X_tr)
        X_va_e = enc.transform_features(X_va)
        y_tr_e = enc.transform_target(y_tr)
        y_va_e = enc.transform_target(y_va)

        self._runner = AutoMLRunner(
            config.task,
            max_models=self.max_models,
            max_runtime_seconds=self.max_runtime_seconds,
            n_cv_splits=min(3, max(2, int(config.cv_splits))),
        )

        def on_progress(pct: int, msg: str) -> None:
            if self.is_cancelled:
                self._runner.cancel()
            # Map AutoML 0–100 into overall 15–95
            mapped = 15 + int(0.80 * max(0, min(100, pct)))
            self.progress.emit(mapped, msg)

        self.progress.emit(15, f"AutoML: comparing up to {self.max_models} models…")
        result = self._runner.run(
            X_tr_e,
            y_tr_e,
            X_va_e,
            y_va_e,
            progress_callback=on_progress,
        )
        if self.is_cancelled:
            raise RuntimeError("AutoML cancelled")
        self.progress.emit(100, f"AutoML complete — {len(result.entries)} model(s)")
        return result
