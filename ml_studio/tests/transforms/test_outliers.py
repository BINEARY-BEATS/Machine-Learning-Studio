import pytest
import pandas as pd
import numpy as np
from ml_studio.transforms.outliers import IQRCap, ZScoreCap, Winsorize, IsolationForestFilter

def test_iqrcap():
    df = pd.DataFrame({"A": [1, 2, 3, 4, 100]})
    t = IQRCap(factor=1.5)
    res = t.fit_transform(df)
    assert res["A"].iloc[-1] < 100
    
def test_zscorecap():
    df = pd.DataFrame({"A": [1, 2, 3, 4, 100]})
    t = ZScoreCap(threshold=1.5)
    res = t.fit_transform(df)
    assert res["A"].iloc[-1] < 100
    
def test_winsorize():
    df = pd.DataFrame({"A": list(range(100)) + [1000]})
    t = Winsorize(limits=(0.01, 0.05)) # Drop top 5%
    res = t.fit_transform(df)
    assert res["A"].max() < 1000
    
def test_isolation_forest():
    df = pd.DataFrame({"A": list(range(100)) + [1000, -1000]})
    t = IsolationForestFilter(contamination=0.05)
    res = t.fit_transform(df)
    assert len(res) < len(df)
    assert 1000 not in res["A"].values
    
def test_isolation_forest_deterministic():
    df = pd.DataFrame({"A": list(range(100)) + [1000, -1000]})
    t1 = IsolationForestFilter(contamination=0.05)
    res1 = t1.fit_transform(df.copy())
    
    t2 = IsolationForestFilter(contamination=0.05)
    res2 = t2.fit_transform(df.copy())
    
    pd.testing.assert_frame_equal(res1, res2)
