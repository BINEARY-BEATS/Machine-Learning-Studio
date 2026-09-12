import pytest
import pandas as pd
from unittest.mock import patch, MagicMock

from ml_studio.core.ingestion import (
    load_dataset_from_url,
    load_dataset_from_path,
    LocalFileSource
)

@patch("fsspec.open")
def test_remote_csv(mock_open):
    # Mock fsspec
    mock_file = MagicMock()
    mock_open.return_value.__enter__.return_value = mock_file
    
    with patch("pandas.read_csv") as mock_read_csv:
        df = pd.DataFrame({"a": [1]})
        mock_read_csv.return_value = df
        
        ds = load_dataset_from_url("s3://bucket/test.csv")
        assert ds.row_count == 1
        assert ds.source_type == "remote"
        mock_read_csv.assert_called_once()

@patch("sqlalchemy.create_engine")
def test_sql_load(mock_engine):
    mock_conn = MagicMock()
    mock_engine.return_value.connect.return_value.__enter__.return_value = mock_conn
    
    with patch("pandas.read_sql") as mock_read_sql:
        mock_read_sql.return_value = pd.DataFrame({"a": [1, 2]})
        
        ds = load_dataset_from_url("sqlite:///my.db", query="SELECT * FROM data")
        assert ds.row_count == 2
        assert ds.source_type == "sql"
        mock_read_sql.assert_called_with("SELECT * FROM data", mock_engine.return_value)
        
        ds2 = load_dataset_from_url("sqlite:///my.db")
        assert ds2.source_type == "sql"
        mock_read_sql.assert_called_with("SELECT * FROM mytable", mock_engine.return_value)

def test_local_unsupported():
    with pytest.raises((ValueError, FileNotFoundError, Exception)):
        load_dataset_from_url("http://example.com/data.xyz")

def test_supported_extensions():
    assert ".csv" in LocalFileSource.SUPPORTED
    assert ".parquet" in LocalFileSource.SUPPORTED
    assert ".jsonl" in LocalFileSource.SUPPORTED
    assert ".feather" in LocalFileSource.SUPPORTED
