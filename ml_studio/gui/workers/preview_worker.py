"""Pipeline preview worker (samples first N rows)."""

from __future__ import annotations

from ml_studio.gui.workers.base_worker import WorkerBase


class PreviewWorker(WorkerBase):
    def __init__(self, pipeline, dataset, sample_rows: int = 500):
        super().__init__()
        self.pipeline = pipeline
        self.dataset = dataset
        self.sample_rows = sample_rows

    def do_work(self):
        self.progress.emit(10, f"Sampling {self.sample_rows} rows…")
        if self.is_cancelled:
            raise RuntimeError("Preview cancelled")
        df = self.dataset.dataframe.head(self.sample_rows)
        self.progress.emit(40, "Fitting pipeline preview…")
        preview = self.pipeline.preview(df)
        self.progress.emit(100, "Preview ready")
        return preview
