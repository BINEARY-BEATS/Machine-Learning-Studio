import pytest
from unittest.mock import MagicMock, patch
from ml_studio.gui.workers.base_worker import WorkerBase
from ml_studio.gui.workers.dataset_worker import DatasetLoadWorker
from ml_studio.gui.workers.training_worker import TrainingWorker

class DummyWorker(WorkerBase):
    def do_work(self):
        self.progress.emit(50, "Halfway")
        return "Success"

def test_base_worker_success():
    worker = DummyWorker()
    prog_vals = []
    res_val = None
    
    worker.progress.connect(lambda p, m: prog_vals.append(p))
    worker.result.connect(lambda r: setattr(worker, "test_res", r))
    
    worker._execute()
    
    assert prog_vals == [50]
    assert worker.test_res == "Success"
    
def test_base_worker_error():
    class ErrorWorker(WorkerBase):
        def do_work(self):
            raise ValueError("Test error")
            
    worker = ErrorWorker()
    err_val = None
    worker.error.connect(lambda e: setattr(worker, "test_err", e))
    
    worker._execute()
    assert "Test error" in worker.test_err

@patch("ml_studio.gui.workers.dataset_worker.load_dataset_from_path")
def test_dataset_load_worker(mock_load):
    from pathlib import Path
    worker = DatasetLoadWorker(Path("dummy.csv"))
    mock_instance = MagicMock()
    mock_instance.row_count = 100
    mock_instance.column_count = 5
    mock_load.return_value = mock_instance
    
    res = worker.do_work()
    
    mock_load.assert_called_once_with(Path("dummy.csv"))
    assert res == mock_instance

def test_training_worker():
    worker = TrainingWorker(MagicMock(), MagicMock(), MagicMock())
    worker._trainer = MagicMock()
    worker._trainer.train.return_value = "trained_model"
    
    res = worker.do_work()
    
    worker._trainer.train.assert_called_once()
    assert res == "trained_model"
