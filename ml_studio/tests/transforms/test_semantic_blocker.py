import pytest
import pandas as pd
import numpy as np

from ml_studio.transforms.encoding import OneHot, Ordinal, Target, WOE, Frequency, Hashing, LeaveOneOut
from ml_studio.transforms.feature_eng import Polynomial, Ratios, Differences, DateParts, CyclicalEncoding, Binning, LogTransform, SqrtTransform, InteractionTerms
from ml_studio.transforms.selection import VarianceThreshold, CorrelationDrop, MISelect, ANOVASelect, Chi2Select, RFESelect, PermutationSelect, SHAPSelect, VIFDrop
from ml_studio.transforms.missing import Impute, DropRows, DropColumns, FillForward, FillBackward
from ml_studio.transforms.scaling import Standard, MinMax, Robust, Quantile, Power, GaussRank
from ml_studio.transforms.outliers import IQRCap, ZScoreCap, Winsorize, IsolationForestFilter

def test_impute_median_correct_value():
    df = pd.DataFrame({"a": [1.0, 3.0, np.nan, 5.0]})
    res = Impute(strategy="median").fit_transform(df)
    assert res["a"].iloc[2] == 3.0

def test_impute_knn_picks_nearest():
    df = pd.DataFrame({"a": [1.0, 2.0, np.nan, 100.0], "b": [1.0, 2.0, 3.0, 100.0]})
    res = Impute(strategy="knn", knn_neighbors=1).fit_transform(df)
    assert res["a"].iloc[2] == 2.0

def test_impute_mode():
    df = pd.DataFrame({"a": ["X", "X", np.nan, "Y"]})
    res = Impute(strategy="mode").fit_transform(df)
    assert res["a"].iloc[2] == "X"
    
def test_impute_mean():
    df = pd.DataFrame({"a": [1.0, 3.0, np.nan]})
    res = Impute(strategy="mean").fit_transform(df)
    assert res["a"].iloc[2] == 2.0

def test_minmax_output_range():
    df = pd.DataFrame({"a": [0.0, 50.0, 100.0]})
    res = MinMax(columns=["a"]).fit_transform(df)
    assert res["a"].min() == 0.0
    assert res["a"].max() == 1.0
    assert res["a"].iloc[1] == 0.5

def test_dateparts_adds_correct_columns():
    df = pd.DataFrame({"dt": pd.to_datetime(["2023-06-15"])})
    res = DateParts(columns=["dt"], parts=["year", "month", "day"]).fit_transform(df)
    assert res["dt_year"].iloc[0] == 2023
    assert res["dt_month"].iloc[0] == 6
    assert res["dt_day"].iloc[0] == 15

def test_schemas_and_to_dict():
    df = pd.DataFrame({'a': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 'b': ['A', 'B', 'A', 'B', 'A', 'B']})
    y = pd.Series([1, 0, 1, 0, 1, 0])
    
    transforms = [
        (Standard(), df), (MinMax(), df), (Robust(), df), (Quantile(), df), (Power(), df), (GaussRank(), df),
        (IQRCap(), df), (ZScoreCap(), df), (Winsorize(), df), (IsolationForestFilter(), df),
        (OneHot(), df), (Ordinal(), df), (Frequency(), df), (Hashing(columns=['b']), df),
        (Target(columns=['b']), df, y), (WOE(columns=['b']), df, y), (LeaveOneOut(columns=['b']), df, y),
        (Polynomial(columns=['a']), df), 
        (Binning(columns=['a']), df), (LogTransform(columns=['a']), df), (SqrtTransform(columns=['a']), df), (InteractionTerms(columns=['a']), df, y),
        (VarianceThreshold(), df), (CorrelationDrop(), df), (MISelect(k=1), df, y), (ANOVASelect(k=1), df, y), 
        (RFESelect(k=1), df, y), (PermutationSelect(k=1), df, y), (SHAPSelect(k=1), df, y), (VIFDrop(), df)
    ]
    
    for t_spec in transforms:
        t = t_spec[0]
        args = t_spec[1:]
        t.fit(*args)
        # Test serialization
        d = t.to_dict()
        assert isinstance(d, dict)
        schema = t.get_schema()
        assert isinstance(schema, dict)

def test_missing_drop_columns():
    df = pd.DataFrame({'a': [1, np.nan, np.nan, np.nan], 'b': [1, 2, 3, 4]})
    t = DropColumns(threshold=0.5)
    res = t.fit_transform(df)
    assert 'a' not in res.columns
    assert 'b' in res.columns

def test_missing_drop_rows():
    df = pd.DataFrame({'a': [1, np.nan, 3, 4]})
    t = DropRows(threshold=0.5)
    res = t.fit_transform(df)
    assert len(res) == 3

def test_chi2_classification():
    df = pd.DataFrame({'a': [1, 2, 3, 4], 'b': [1, 1, 1, 1]})
    y = pd.Series([1, 0, 1, 0], dtype='category')
    t = Chi2Select(k=1)
    res = t.fit_transform(df, y)
    assert res.shape[1] == 1

def test_unseen_categories():
    df_train = pd.DataFrame({'cat': ['A', 'B']})
    df_test = pd.DataFrame({'cat': ['A', 'C']})
    
    t = Ordinal(columns=['cat'])
    t.fit(df_train)
    res = t.transform(df_test)
    assert res['cat'].iloc[1] == -1 # Assuming -1 for unknown
    
    t = OneHot(columns=['cat'])
    t.fit(df_train)
    res = t.transform(df_test)
    assert 'cat_C' not in res.columns
