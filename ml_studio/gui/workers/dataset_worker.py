"""Dataset loading worker."""

from pathlib import Path

from ml_studio.core.ingestion import load_dataset_from_path
from ml_studio.gui.workers.base_worker import WorkerBase


class DatasetLoadWorker(WorkerBase):
    def __init__(self, path: Path):
        super().__init__()
        self.path = path

    def do_work(self):
        self.progress.emit(5, f"Opening {self.path.name}…")
        if self.is_cancelled:
            return None
        self.progress.emit(25, "Detecting format and reading rows…")
        ds = load_dataset_from_path(self.path)
        self.progress.emit(
            100,
            f"Loaded {ds.row_count:,} rows × {ds.column_count} columns",
        )
        return ds
