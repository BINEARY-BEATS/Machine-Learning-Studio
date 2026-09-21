"""Training worker — prepares data then trains off the UI thread."""

from __future__ import annotations

import pandas as pd

from ml_studio.core.training.data_prep import prepare_for_training
from ml_studio.core.training.trainer import Trainer, TrainingConfig
from ml_studio.gui.workers.base_worker import WorkerBase


class TrainingWorker(WorkerBase):
    def __init__(
        self,
        df: pd.DataFrame,
        config: TrainingConfig,
        preprocessing=None,
        *,
        prepare_data: bool = True,
    ):
        super().__init__()
        self.df = df
        self.config = config
        self.preprocessing = preprocessing
        self.prepare_data = prepare_data
        self._trainer = Trainer()

    def do_work(self):
        def progress(pct, msg):
            if self.is_cancelled:
                self._trainer.cancel()
            self.progress.emit(pct, msg)

        df = self.df
        config = self.config
        if self.prepare_data:
            progress(5, "Preparing training data…")
            if self.is_cancelled:
                raise RuntimeError("Training cancelled")
            prepared_df, target, features, _ = prepare_for_training(
                df,
                config.task,
                target_column=config.target_column,
                feature_columns=config.feature_columns or None,
            )
            config.target_column = target
            config.feature_columns = features
            df = prepared_df
            progress(15, "Starting model training…")

        return self._trainer.train(
            df, config, self.preprocessing, progress_callback=progress
        )
