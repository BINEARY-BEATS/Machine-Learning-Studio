import pytest
import pandas as pd
import numpy as np
from ml_studio.transforms.feature_eng import (
    Polynomial, Ratios, Differences, DateParts, 
    CyclicalEncoding, Binning, LogTransform, 
    SqrtTransform, InteractionTerms
)

def test_polynomial():
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    t = Polynomial(degree=2, interaction_only=False, include_bias=False)
    res = t.fit_transform(df)
    # A, B, A^2, AB, B^2 -> 5 features
    assert res.shape[1] == 5
    assert "A" in res.columns
    assert "B" in res.columns
    
def test_ratios():
    df = pd.DataFrame({"A": [10, 20], "B": [2, 5]})
    t = Ratios(pairs=[("A", "B")])
    res = t.fit_transform(df)
    assert "A_ratio_B" in res.columns
    assert res["A_ratio_B"].iloc[0] == 5.0
    
def test_differences():
    df = pd.DataFrame({"A": [10, 20], "B": [2, 5]})
    t = Differences(pairs=[("A", "B")])
    res = t.fit_transform(df)
    assert "A_diff_B" in res.columns
    assert res["A_diff_B"].iloc[0] == 8.0

def test_date_parts():
    df = pd.DataFrame({"dt": pd.to_datetime(["2023-01-01", "2023-12-31"])})
    t = DateParts()
    res = t.fit_transform(df)
    assert "dt_year" in res.columns
    assert res["dt_year"].iloc[0] == 2023
    assert res["dt_month"].iloc[1] == 12

def test_cyclical_encoding():
    df = pd.DataFrame({"month": [1, 6, 12]})
    t = CyclicalEncoding(period=12)
    res = t.fit_transform(df)
    assert "month_sin" in res.columns
    assert "month_cos" in res.columns
    assert "month" not in res.columns
    
def test_binning():
    df = pd.DataFrame({"A": list(range(100))})
    t = Binning(n_bins=4, strategy="equal_width")
    res = t.fit_transform(df)
    assert res["A"].nunique() == 4
    
def test_log_transform():
    df = pd.DataFrame({"A": [0, 1, 2]})
    t = LogTransform(offset=1.0)
    res = t.fit_transform(df)
    # log1p(0+1) = log(2)
    assert np.isclose(res["A"].iloc[0], np.log1p(1.0))
    
def test_sqrt_transform():
    df = pd.DataFrame({"A": [0, 4, 9]})
    t = SqrtTransform()
    res = t.fit_transform(df)
    assert res["A"].iloc[1] == 2.0

def test_interaction_terms():
    df = pd.DataFrame({"A": [1, 2, 3, 4, 5, 6], "B": [4, 5, 6, 7, 8, 9], "C": [7, 8, 9, 10, 11, 12]})
    y = pd.Series([1, 0, 1, 0, 1, 0])
    t = InteractionTerms(top_k=1)
    res = t.fit_transform(df, y)
    # Should have added exactly 1 interaction term
    assert res.shape[1] == 4
