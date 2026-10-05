"""Dataset loading worker."""

from pathlib import Path
from typing import Any

from ml_studio.core.ingestion import load_dataset_from_path
from ml_studio.gui.workers.base_worker import WorkerBase


class DatasetLoadWorker(WorkerBase):
    def __init__(self, path: Path, options: dict[str, Any] | None = None):
        super().__init__()
        self.path = path
        self.options = dict(options or {})

    def do_work(self):
        self.progress.emit(5, f"Opening {self.path.name}…")
        if self.is_cancelled:
            return None
        self.progress.emit(25, "Detecting format and reading rows…")
        ds = load_dataset_from_path(self.path, **self.options)
        self.progress.emit(
            100,
            f"Loaded {ds.row_count:,} rows × {ds.column_count} columns",
        )
        return ds
