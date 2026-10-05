"""Batch prediction worker (chunked, cancellable) with column preflight."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ml_studio.gui.workers.base_worker import WorkerBase


class BatchPredictWorker(WorkerBase):
    def __init__(self, predictor, path: Path, output_path: Path | None = None):
        super().__init__()
        self.predictor = predictor
        self.path = Path(path)
        self.output_path = Path(output_path) if output_path else None

    @staticmethod
    def preflight(path: Path, expected_columns: list[str]) -> tuple[list[str], list[str]]:
        """Read header only; return (missing, extra) vs expected feature columns."""
        path = Path(path)
        suffix = path.suffix.lower()
        if suffix == ".csv":
            header = list(pd.read_csv(path, nrows=0).columns)
        elif suffix in {".parquet", ".pq"}:
            header = list(pd.read_parquet(path).head(0).columns)
        else:
            raise ValueError(f"Unsupported batch format: {suffix}")
        expected = list(expected_columns)
        missing = [c for c in expected if c not in header]
        extra = [c for c in header if c not in expected]
        return missing, extra

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
