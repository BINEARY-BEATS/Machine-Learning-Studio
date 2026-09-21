"""Memory dtype optimization worker."""

from __future__ import annotations

import pandas as pd

from ml_studio.core.profiling import optimize_dtypes
from ml_studio.gui.workers.base_worker import WorkerBase


class OptimizeWorker(WorkerBase):
    def __init__(self, dataframe: pd.DataFrame):
        super().__init__()
        self.dataframe = dataframe

    def do_work(self):
        self.progress.emit(10, "Analyzing dtypes…")
        if self.is_cancelled:
            raise RuntimeError("Optimize cancelled")
        optimized, report = optimize_dtypes(self.dataframe)
        self.progress.emit(100, "Optimization complete")
        return {"dataframe": optimized, "report": report}
