import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from ml_studio.core.pipeline import Pipeline
from ml_studio.transforms.missing import Impute
from ml_studio.transforms.scaling import Standard
from ml_studio.core.schema import ColumnRole

def test_pipeline_serialization():
    p = Pipeline([
        Impute(strategy="median", columns=["A"]),
        Standard(columns=["A", "B"])
    ])
    d = p.to_dict()
    assert len(d["steps"]) == 2
    
    p2 = Pipeline.from_dict(d)
    assert len(p2.steps) == 2
    assert p2.steps[0].__class__.__name__ == "Impute"
    assert p2.steps[1].__class__.__name__ == "Standard"

def test_pipeline_hash_stability():
    p1 = Pipeline([Impute(strategy="median", columns=["A"])])
    h1 = p1.hash()
    
    p2 = Pipeline([Impute(strategy="median", columns=["A"])])
    h2 = p2.hash()
    
    assert h1 == h2
    
    # After fit, state changes, so hash might change, but should be stable for same state
    df = pd.DataFrame({"A": [1, 2, 3]})
    p1.fit(df)
    p2.fit(df)
    assert p1.hash() == p2.hash()

def test_pipeline_fit_transform_equiv():
    df = pd.DataFrame({"A": [1, 2, np.nan, 4], "B": [10, 20, 30, 40]})
    
    p1 = Pipeline([Impute(strategy="mean"), Standard()])
    res1 = p1.fit_transform(df)
    
    p2 = Pipeline([Impute(strategy="mean"), Standard()])
    p2.fit(df)
    res2 = p2.transform(df)
    
    pd.testing.assert_frame_equal(res1, res2)

def test_pipeline_preview():
    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    p = Pipeline([Standard(columns=["A"])])
    
    preview = p.preview(df)
    assert preview.original_shape == (3, 2)
    assert len(preview.steps) == 1
    assert preview.steps[0]["name"] == "Standard"

def test_pipeline_column_roles():
    df = pd.DataFrame({"feat1": [1, 2], "targ": [0, 1]})
    df.attrs["roles"] = {"feat1": "feature", "targ": "target"}
    
    # Explicit attempt to scale target should fail
    p = Pipeline([Standard(columns=["targ"])])
    with pytest.raises(ValueError, match="attempts to modify column"):
        p.fit(df)
    
def test_pipeline_column_ordering_stability():
    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]})
    p = Pipeline([
        Impute(strategy="mean", columns=["C"]),
        Standard(columns=["A", "B"])
    ])
    
    res1 = p.fit_transform(df)
    order1 = res1.columns.tolist()
    
    res2 = p.transform(df)
    order2 = res2.columns.tolist()
    
    assert order1 == order2
    
    # Save/load
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "pipe.json")
        p.save(path)
        p_loaded = Pipeline.load(path)
        
    res3 = p_loaded.transform(df)
    order3 = res3.columns.tolist()
    
    assert order1 == order3
