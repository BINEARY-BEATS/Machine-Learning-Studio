import pytest
import pandas as pd
import numpy as np
from ml_studio.transforms.scaling import Standard, MinMax, Robust, Quantile, Power, GaussRank

def test_standard_scaler():
    df = pd.DataFrame({"A": [1, 2, 3]})
    t = Standard()
    res = t.fit_transform(df)
    assert np.isclose(res["A"].mean(), 0.0)
    assert np.isclose(res["A"].std(ddof=0), 1.0)
    
def test_minmax_scaler():
    df = pd.DataFrame({"A": [1, 2, 3]})
    t = MinMax(feature_range=(0, 10))
    res = t.fit_transform(df)
    assert res["A"].min() == 0.0
    assert res["A"].max() == 10.0

def test_robust_scaler():
    df = pd.DataFrame({"A": [1, 2, 3, 100]})
    t = Robust()
    res = t.fit_transform(df)
    assert "A" in res.columns
    
def test_quantile_scaler():
    df = pd.DataFrame({"A": np.random.normal(0, 1, 100)})
    t = Quantile(n_quantiles=10)
    res = t.fit_transform(df)
    assert "A" in res.columns
    
def test_power_scaler():
    df = pd.DataFrame({"A": [1, 2, 3, 4, 5]})
    t = Power(method="yeo-johnson")
    res = t.fit_transform(df)
    assert "A" in res.columns

    # Box-cox failure on <=0
    df2 = pd.DataFrame({"A": [0, 1, 2]})
    t2 = Power(method="box-cox")
    with pytest.raises(ValueError):
        t2.fit(df2)

def test_gauss_rank():
    df = pd.DataFrame({"A": [10, 20, 30, 40, 50]})
    t = GaussRank()
    res = t.fit_transform(df)
    # Ranks should map to norm.ppf
    assert "A" in res.columns
    assert res["A"].iloc[0] < 0  # smallest is negative
    assert res["A"].iloc[-1] > 0 # largest is positive
    
def test_serialize_standard():
    df = pd.DataFrame({"A": [1, 2, 3]})
    t = Standard()
    t.fit(df)
    d = t.to_dict()
    t2 = Standard.from_dict(d)
    assert t2.means_ == t.means_
