import pytest
import pandas as pd
import numpy as np
from ml_studio.transforms.missing import Impute, DropRows, DropColumns, FillForward, FillBackward

def test_impute_schema():
    schema = Impute.get_schema()
    assert schema["type"] == "object"
    assert "strategy" in schema["properties"]

def test_impute_fit_transform():
    df = pd.DataFrame({"A": [1, 2, np.nan, 4], "B": ["x", "y", np.nan, "x"]})
    
    # Median for numeric, mode for categorical
    transform = Impute(strategy="median", columns=["A", "B"])
    res = transform.fit_transform(df)
    
    assert res["A"].isna().sum() == 0
    assert res["A"].iloc[2] == 2.0  # Median of 1,2,4 is 2.0 (wait, median of 1,2,4 is 2.0. 1, 2, 4 -> 2)
    assert res["B"].isna().sum() == 0
    assert res["B"].iloc[2] == "x"  # Mode of x,y,x is x

def test_impute_serialize():
    df = pd.DataFrame({"A": [1, 2, np.nan, 4]})
    transform = Impute(strategy="mean")
    transform.fit(df)
    
    d = transform.to_dict()
    assert d["imputers_"]["A"] == (1+2+4)/3
    
    t2 = Impute.from_dict(d)
    assert t2.imputers_["A"] == transform.imputers_["A"]
    assert t2._is_fitted

def test_impute_leakage():
    df_train = pd.DataFrame({"A": [10, 20, np.nan, 40]}) # mean = 70/3 = 23.33
    df_test = pd.DataFrame({"A": [100, 200, np.nan, 400]}) # mean = 700/3 = 233.33
    
    transform = Impute(strategy="mean")
    transform.fit(df_train)
    
    # Transform test set
    res_test = transform.transform(df_test)
    
    # Test should be filled with TRAIN mean, not test mean
    assert np.isclose(res_test["A"].iloc[2], 23.333333)
    
def test_drop_rows():
    df = pd.DataFrame({
        "A": [1, np.nan, np.nan],
        "B": [1, 2, np.nan]
    })
    
    t = DropRows(threshold=0.6) # Drop if > 60% missing (2 cols, 1 missing = 50%, 2 missing = 100%)
    res = t.fit_transform(df)
    assert len(res) == 2
    assert res.index.tolist() == [0, 1]
    
def test_drop_columns():
    df = pd.DataFrame({
        "A": [1, 2, 3],
        "B": [1, np.nan, np.nan]
    })
    
    t = DropColumns(threshold=0.5)
    res = t.fit_transform(df)
    assert "B" not in res.columns
    assert "A" in res.columns
    
def test_fill_forward():
    df = pd.DataFrame({"A": [1, np.nan, 3]})
    t = FillForward()
    res = t.fit_transform(df)
    assert res["A"].iloc[1] == 1.0
    
def test_fill_backward():
    df = pd.DataFrame({"A": [1, np.nan, 3]})
    t = FillBackward()
    res = t.fit_transform(df)
    assert res["A"].iloc[1] == 3.0
