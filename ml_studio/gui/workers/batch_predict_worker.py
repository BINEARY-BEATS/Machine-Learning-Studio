"""Batch prediction worker (chunked, cancellable)."""

from __future__ import annotations

from pathlib import Path

from ml_studio.gui.workers.base_worker import WorkerBase


class BatchPredictWorker(WorkerBase):
    def __init__(self, predictor, path: Path, output_path: Path | None = None):
        super().__init__()
        self.predictor = predictor
        self.path = Path(path)
        self.output_path = output_path

    def do_work(self):
        self.progress.emit(5, f"Reading {self.path.name}…")

        def progress(pct: int, msg: str) -> None:
            self.progress.emit(pct, msg)

        def cancel_check() -> bool:
            return self.is_cancelled

        out = self.predictor.predict_batch(
            self.path,
            output_path=self.output_path,
            progress_callback=progress,
            cancel_check=cancel_check,
        )
        self.progress.emit(100, "Batch prediction complete")
        return out
