import pytest
from unittest.mock import MagicMock, patch
from ml_studio.gui.workers.profiling_worker import ProfilingWorker


@patch("ml_studio.gui.workers.profiling_worker.detect_quality_issues")
@patch("ml_studio.gui.workers.profiling_worker.profile_dataset")
def test_profiling_worker(mock_profile, mock_issues):
    worker = ProfilingWorker(MagicMock())
    mock_profile.return_value = "ProfileResult"
    mock_issues.return_value = [{"type": "missing"}]

    res = worker.do_work()
    mock_profile.assert_called_once()
    mock_issues.assert_called_once()
    assert res == {"profile": "ProfileResult", "issues": [{"type": "missing"}]}
