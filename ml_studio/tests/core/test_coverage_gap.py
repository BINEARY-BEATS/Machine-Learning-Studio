import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock

from ml_studio.core.ingestion import RemoteFileSource, load_dataset_from_url
from ml_studio.api import Project

def test_remote_formats():
    # Test all branches of RemoteFileSource
    with patch("fsspec.open"):
        with patch("pandas.read_parquet") as mock_pq:
            load_dataset_from_url("s3://test.parquet")
            mock_pq.assert_called_once()
        with patch("pandas.read_json") as mock_json:
            load_dataset_from_url("http://test.json")
            mock_json.assert_called_once()
        with patch("pandas.read_json") as mock_jsonl:
            load_dataset_from_url("s3://test.jsonl")
            mock_jsonl.assert_called_once()
        with patch("pandas.read_feather") as mock_feather:
            load_dataset_from_url("s3://test.feather")
            mock_feather.assert_called_once()
        with patch("pandas.read_orc") as mock_orc:
            load_dataset_from_url("s3://test.orc")
            mock_orc.assert_called_once()
        with patch("pandas.read_excel") as mock_excel:
            load_dataset_from_url("s3://test.xlsx")
            mock_excel.assert_called_once()
        with pytest.raises(ValueError, match="Unsupported remote format"):
            load_dataset_from_url("s3://test.badext")

def test_api_missing_file_open(tmp_path):
    import os
    orig = os.getcwd()
    os.chdir(tmp_path)
    try:
        p = Project.create("test_missing_file_open")
        p.save()
        p2 = Project.open("test_missing_file_open")
        assert p2.name == "test_missing_file_open"
    finally:
        os.chdir(orig)
    
def test_api_get_xy_errors():
    p = Project.create("test_xy")
    from ml_studio.core.dataset import Dataset
    p.dataset = Dataset("test", "test")
    p.dataset.set_dataframe(pd.DataFrame({"a": [1, 2], "b": [3, 4]}), reason="test")
    # target not set
    with pytest.raises(ValueError, match="Target column not set"):
        p.get_xy()
