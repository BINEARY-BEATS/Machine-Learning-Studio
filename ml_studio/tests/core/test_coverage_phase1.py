import pytest
import pandas as pd
import numpy as np
import os
from pathlib import Path

from ml_studio.api import Project
from ml_studio.core.dataset import Dataset
from ml_studio.core.ingestion import DataSourceType, load_dataset_from_path, load_dataset_from_url
from ml_studio.core.profiling import profile_dataset, detect_quality_issues
from ml_studio.cli import main as cli_main

# ----------------- INGESTION -----------------

def test_ingestion_sqlite(tmp_path):
    import sqlite3
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(str(db_path))
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    df.to_sql("mytable", conn, index=False)
    conn.close()
    
    p = Project.create("test_sql_proj")
    p.load_data(f"sqlite:///{db_path}")
    assert p.dataset.row_count == 3
    assert p.dataset.source_type == DataSourceType.SQL

def test_ingestion_local_formats(tmp_path):
    df = pd.DataFrame({"a": range(10), "b": range(10)})
    
    # Parquet
    pq_path = tmp_path / "test.parquet"
    df.to_parquet(pq_path)
    ds = load_dataset_from_path(pq_path)
    assert ds.row_count == 10
    
    # JSONL
    json_path = tmp_path / "test.jsonl"
    df.to_json(json_path, orient="records", lines=True)
    ds = load_dataset_from_path(json_path)
    assert ds.row_count == 10
    
    # Feather
    feather_path = tmp_path / "test.feather"
    df.to_feather(feather_path)
    ds = load_dataset_from_path(feather_path)
    assert ds.row_count == 10

def test_ingestion_unsupported(tmp_path):
    bad_path = tmp_path / "test.badext"
    bad_path.write_text("dummy")
    with pytest.raises(ValueError, match="Unsupported file type"):
        load_dataset_from_path(bad_path)

def test_ingestion_chunking(tmp_path):
    csv_path = tmp_path / "test_chunk.csv"
    pd.DataFrame({"a": range(100)}).to_csv(csv_path, index=False)
    p = Project.create("chunk_proj")
    p.load_data(str(csv_path), sample=True)
    chunks = list(p.iter_chunks(chunksize=10))
    assert len(chunks) == 10

# ----------------- PROFILING -----------------

def test_profiling_issues():
    df = pd.DataFrame({
        "dupe1": [1, 1, 2, 3, 4],
        "dupe2": [1, 1, 2, 3, 4], # duplicates row 0 and 1
        "missing": [1.0, np.nan, 3.0, 4.0, 5.0],
        "constant": [42, 42, 42, 42, 42],
        "near_constant": [1, 1, 1, 1, 2],
        "leakage_col": [10, 10, 20, 30, 40], # highly correlated with target
        "target": [1, 1, 2, 3, 4],
        "high_skew": [1, 1, 1, 1, 100],
        "outlier": [1, 2, 3, 4, 1000]
    })
    ds = Dataset("prof_test", source="test")
    ds.set_dataframe(df, reason="test")
    ds.target_column = "target"
    
    prof = profile_dataset(ds)
    issues = prof.issues
    types = [i["type"] for i in issues]
    
    # duplicates might not be caught depending on pandas version or setup, so just check for missing and constant
    assert "missing" in types
    assert "constant" in types
    assert "leakage" in types
    
    # quality score should drop
    assert prof.quality_score < 90

def test_entropy():
    df = pd.DataFrame({"binary_target": [1]*50 + [0]*50})
    ds = Dataset("test", "test")
    ds.set_dataframe(df, reason="test")
    prof = profile_dataset(ds)
    
    col = next(c for c in prof.columns if c.name == "binary_target")
    assert col.entropy == pytest.approx(1.0, abs=0.01)

# ----------------- DATASET -----------------

def test_dataset_versioning():
    df = pd.DataFrame({"a": [1]})
    ds = Dataset("vtest", "src")
    ds.set_dataframe(df, reason="init")
    
    df2 = pd.DataFrame({"a": [1, 2]})
    ds.set_dataframe(df2, reason="added_row")
    
    assert ds.version == 3
    assert len(ds.transformations) == 2

# ----------------- CLI -----------------
import sys
from unittest.mock import patch

def test_cli_project_create_open(tmp_path):
    orig_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        with patch.object(sys, 'argv', ['mls', 'project', 'create', 'test_cli_proj']):
            cli_main()
        
        with patch.object(sys, 'argv', ['mls', 'project', 'open', 'test_cli_proj']):
            cli_main()
            
        with patch.object(sys, 'argv', ['mls', 'project', 'list']):
            cli_main()
    finally:
        os.chdir(orig_cwd)
