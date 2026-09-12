import pytest
import pandas as pd
import numpy as np
from ml_studio.transforms.selection import (
    VarianceThreshold, CorrelationDrop, MISelect, ANOVASelect, 
    Chi2Select, RFESelect, PermutationSelect, SHAPSelect, VIFDrop
)

def test_variance_threshold():
    df = pd.DataFrame({"A": [1, 1, 1], "B": [1, 2, 3]})
    t = VarianceThreshold(threshold=0.1)
    res = t.fit_transform(df)
    assert "A" not in res.columns
    assert "B" in res.columns
    
def test_correlation_drop():
    df = pd.DataFrame({"A": [1, 2, 3], "B": [2, 4, 6], "C": [1, 0, 1]})
    t = CorrelationDrop(threshold=0.95)
    res = t.fit_transform(df)
    # A and B are perfectly correlated. B should be dropped.
    assert "B" not in res.columns
    assert "A" in res.columns
    assert "C" in res.columns
    
def test_mi_select():
    df = pd.DataFrame({"A": [0, 1, 0, 1, 0, 1], "B": [6, 5, 4, 3, 2, 1], "C": [1, 1, 1, 1, 1, 1]})
    y = pd.Series([0, 1, 0, 1, 0, 1])
    t = MISelect(k=1) # Keep top 1
    res = t.fit_transform(df, y)
    assert res.shape[1] == 1
    assert "C" not in res.columns # C has no info
    
def test_anova_select():
    df = pd.DataFrame({"A": [1, 10, 1, 10], "B": [1, 2, 3, 4]})
    y = pd.Series([0, 1, 0, 1])
    t = ANOVASelect(k=1)
    res = t.fit_transform(df, y)
    assert "A" in res.columns
    assert "B" not in res.columns
    
def test_chi2_select():
    df = pd.DataFrame({"A": [10, 20, 10, 20], "B": [1, 1, 1, 1]})
    y = pd.Series(["cls1", "cls2", "cls1", "cls2"])  # Force classification detection
    t = Chi2Select(k=1)
    res = t.fit_transform(df, y)
    assert "A" in res.columns
    
def test_rfe_select():
    df = pd.DataFrame({"A": [1, 2, 3, 4], "B": [4, 3, 2, 1], "C": [0, 0, 0, 0]})
    y = pd.Series([1, 2, 3, 4])
    t = RFESelect(k=2)
    res = t.fit_transform(df, y)
    assert "C" not in res.columns
    assert "A" in res.columns
    
def test_permutation_select():
    df = pd.DataFrame({"A": [1, 2, 3, 4], "B": [0, 1, 0, 1]})
    y = pd.Series([1, 2, 3, 4])
    t = PermutationSelect(k=1)
    res = t.fit_transform(df, y)
    assert "A" in res.columns
    assert "B" not in res.columns

def test_vif_drop():
    df = pd.DataFrame({"A": [1, 2, 3, 4], "B": [2, 4, 6, 8], "C": [1, 0, 1, 0]})
    t = VIFDrop(threshold=5.0)
    res = t.fit_transform(df)
    # A and B are perfectly collinear, VIF is infinite. One will be dropped.
    assert len(res.columns) == 2
