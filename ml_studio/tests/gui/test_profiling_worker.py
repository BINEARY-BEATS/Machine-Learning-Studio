import pytest
from unittest.mock import MagicMock, patch
from ml_studio.gui.workers.profiling_worker import ProfilingWorker

@patch("ml_studio.gui.workers.profiling_worker.profile_dataset")
def test_profiling_worker(mock_profile):
    worker = ProfilingWorker(MagicMock())
    mock_profile.return_value = "ProfileResult"
    
    res = worker.do_work()
    mock_profile.assert_called_once()
    assert res == "ProfileResult"
