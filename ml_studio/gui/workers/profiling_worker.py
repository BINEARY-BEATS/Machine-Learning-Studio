"""Profiling worker — stats + quality issues off the UI thread."""

from ml_studio.core.profiling import detect_quality_issues, profile_dataset
from ml_studio.gui.workers.base_worker import WorkerBase


class ProfilingWorker(WorkerBase):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def do_work(self):
        if self.is_cancelled:
            raise RuntimeError("Profiling cancelled")
        self.progress.emit(20, "Computing column statistics…")
        profile = profile_dataset(self.dataset)
        if self.is_cancelled:
            raise RuntimeError("Profiling cancelled")
        self.progress.emit(70, "Detecting quality issues…")
        issues = detect_quality_issues(self.dataset)
        self.progress.emit(100, "Profile complete")
        return {"profile": profile, "issues": issues}
