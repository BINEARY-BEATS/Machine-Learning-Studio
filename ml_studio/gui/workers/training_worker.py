"""Training worker."""

from ml_studio.core.training.trainer import Trainer, TrainingConfig
from ml_studio.gui.workers.base_worker import WorkerBase


class TrainingWorker(WorkerBase):
    def __init__(self, df, config: TrainingConfig, preprocessing=None):
        super().__init__()
        self.df = df
        self.config = config
        self.preprocessing = preprocessing
        self._trainer = Trainer()

    def do_work(self):
        def progress(pct, msg):
            if self.is_cancelled:
                self._trainer.cancel()
            self.progress.emit(pct, msg)

        return self._trainer.train(
            self.df, self.config, self.preprocessing, progress_callback=progress
        )
