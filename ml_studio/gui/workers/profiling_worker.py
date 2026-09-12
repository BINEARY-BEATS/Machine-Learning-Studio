"""Profiling worker."""

from ml_studio.core.profiling import profile_dataset
from ml_studio.gui.workers.base_worker import WorkerBase


class ProfilingWorker(WorkerBase):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def do_work(self):
        self.progress.emit(30, "Computing statistics...")
        profile = profile_dataset(self.dataset)
        self.progress.emit(100, "Complete")
        return profile
