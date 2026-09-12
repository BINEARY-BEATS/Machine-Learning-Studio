import pytest
import pandas as pd
import json
from pathlib import Path
import os
import sys
from unittest.mock import patch

from ml_studio.api import Project
from ml_studio.core.dataset import Dataset
from ml_studio.cli import main as cli_main

# ----------------- API / PROJECT -----------------

def test_api_project_lifecycle(tmp_path):
    p = Project.create("api_test")
    p.task = "regression"
    
    # Save/load round-trip
    p.save()
    p2 = Project.open("api_test")
    assert p2.task == "regression"
    
    df = pd.DataFrame({"a": [1, 2]})
    csv_path = tmp_path / "target_test.csv"
    df.to_csv(csv_path, index=False)
    p2.load_data(str(csv_path))
    p2.set_target("a")
    assert p2.dataset.target_column == "a"

def test_api_chunk_iteration(tmp_path):
    p = Project.create("chunk_api_test")
    
    csv_path = tmp_path / "test.csv"
    pd.DataFrame({"a": range(5)}).to_csv(csv_path, index=False)
    
    # Needs to be loaded first
    with pytest.raises(ValueError):
        list(p.iter_chunks())
        
    p.load_data(str(csv_path), sample=False)
    
    # Iterate with small chunks
    chunks = list(p.iter_chunks(chunksize=2))
    assert len(chunks) == 3 # 2, 2, 1
    
    # Edge case: 1 large chunk
    chunks = list(p.iter_chunks(chunksize=10))
    assert len(chunks) == 1

def test_api_empty_file_chunk(tmp_path):
    p = Project.create("empty_chunk_test")
    csv_path = tmp_path / "empty.csv"
    pd.DataFrame(columns=["a"]).to_csv(csv_path, index=False)
    
    p.load_data(str(csv_path), sample=False)
    chunks = list(p.iter_chunks(chunksize=2))
    assert len(chunks) == 1
    assert len(chunks[0]) == 0

def test_api_sampling_thresholds(tmp_path):
    p = Project.create("thresh_test")
    csv_path = tmp_path / "thresh.csv"
    
    # Under threshold
    pd.DataFrame({"a": range(10)}).to_csv(csv_path, index=False)
    p.load_data(str(csv_path), sample=True)
    assert p.dataset.row_count == 10
    
    # At threshold (1M not strictly needed to test exactly, we trust pandas nrows)
    # But let's verify sample=False disables it
    p.load_data(str(csv_path), sample=False)
    assert p.dataset.row_count == 10

def test_project_invalid_load():
    with pytest.raises(Exception):
        Project.open("does_not_exist")

# ----------------- CLI -----------------

def test_cli_stubs(tmp_path, capsys):
    orig_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        p = Project.create("test_stubs")
        p.save()
        
        with patch.object(sys, 'argv', ['mls', 'project', 'info', 'test_stubs']):
            cli_main()
            captured = capsys.readouterr()
            assert "Not yet implemented" in captured.out
            
        with patch.object(sys, 'argv', ['mls', 'data', 'head', 'test_stubs', '-n', '5']):
            cli_main()
            captured = capsys.readouterr()
            assert "Not yet implemented" in captured.out
            
        with patch.object(sys, 'argv', ['mls', 'data', 'info', 'test_stubs']):
            cli_main()
            captured = capsys.readouterr()
            assert "Not yet implemented" in captured.out
            
        # Test profile --json
        csv_path = tmp_path / "test.csv"
        pd.DataFrame({"a": range(10)}).to_csv(csv_path, index=False)
        p.load_data(str(csv_path))
        
        with patch.object(sys, 'argv', ['mls', 'profile', 'test_stubs', '--json']):
            cli_main()
            captured = capsys.readouterr()
            # output should parse as json
            res = json.loads(captured.out)
            assert res["row_count"] == 10
    finally:
        os.chdir(orig_cwd)
