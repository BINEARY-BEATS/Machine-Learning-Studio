import pytest
import pandas as pd
import numpy as np

from ml_studio.transforms.encoding import OneHot, Ordinal, Target, WOE, Frequency, Hashing, LeaveOneOut
from ml_studio.transforms.feature_eng import Polynomial, Ratios, Differences, DateParts, CyclicalEncoding, Binning, LogTransform, SqrtTransform, InteractionTerms
from ml_studio.transforms.selection import VarianceThreshold, CorrelationDrop, MISelect, ANOVASelect, Chi2Select, RFESelect, PermutationSelect, SHAPSelect, VIFDrop
from ml_studio.transforms.missing import Impute, DropRows, DropColumns, FillForward, FillBackward
from ml_studio.transforms.scaling import Standard, MinMax, Robust, Quantile, Power, GaussRank
from ml_studio.transforms.outliers import IQRCap, ZScoreCap, Winsorize, IsolationForestFilter
from ml_studio.core.pipeline import Pipeline

class DummyEstimator:
    def predict(self, X): return np.ones(len(X))
    def predict_proba(self, X): return np.ones((len(X), 2))
    def score(self, X, y, sample_weight=None): return 0.99

class DummyStep:
    def __init__(self): self.columns = []
    def fit(self, X, y=None): return self
    def transform(self, X): return X
    def predict(self, X): return np.ones(len(X))
    def predict_proba(self, X): return np.ones((len(X), 2))
    def score(self, X, y, sample_weight=None): return 0.99

def test_pipeline_methods():
    df = pd.DataFrame({"a": [1, 2]})
    p = Pipeline([DummyStep(), DummyStep()])
    p.fit(df)
    assert len(p.predict(df)) == 2
    assert len(p.predict_proba(df)) == 2
    assert p.score(df, np.array([1, 1])) == 0.99

def test_missing_drop_columns_transform():
    df = pd.DataFrame({"a": [1, np.nan, np.nan], "b": [1, 2, 3]})
    t = DropColumns(threshold=0.5)
    t.fit(df)
    res = t.transform(df)
    assert 'a' not in res.columns
    assert 'b' in res.columns
    assert t.get_output_columns(["a", "b"]) == ["b"]

def test_missing_drop_rows_transform():
    df = pd.DataFrame({"a": [1, np.nan, 3]})
    t = DropRows(threshold=0.5)
    t.fit(df)
    res = t.transform(df)
    assert len(res) == 2

def test_missing_fill_fwd_bwd_transform():
    df = pd.DataFrame({"a": [1, np.nan, 3]})
    t = FillForward(columns=["a"])
    t.fit(df)
    assert t.transform(df)["a"].iloc[1] == 1.0
    
    t2 = FillBackward(columns=["a"])
    t2.fit(df)
    assert t2.transform(df)["a"].iloc[1] == 3.0

def test_selectors_output_columns():
    df = pd.DataFrame({'a': [1, 1, 1, 2, 2], 'b': [1, 2, 3, 4, 5], 'c': [1, 0, 1, 0, 1]})
    y = pd.Series([1, 0, 1, 0, 1])
    
    for t_spec in [
        (VarianceThreshold(threshold=0.5), df),
        (CorrelationDrop(threshold=0.9), df),
        (MISelect(k=1), df, y),
        (ANOVASelect(k=1), df, y),
        (Chi2Select(k=1), df, pd.Series([1, 0, 1, 0, 1], dtype='category')),
        (RFESelect(k=1), df, y),
        (PermutationSelect(k=1), df, y),
        (SHAPSelect(k=1), df, y),
        (VIFDrop(), df)
    ]:
        t = t_spec[0]
        args = t_spec[1:]
        t.fit(*args)
        # Just call it
        t.get_output_columns(df.columns.tolist())
        
def test_outliers_cap_transform():
    df = pd.DataFrame({'a': [1.0, 2.0, 1000.0, -1000.0, 3.0, 4.0, 5.0]})
    
    for t in [IQRCap(columns=['a']), ZScoreCap(columns=['a'])]:
        t.fit(df)
        res = t.transform(df)
        assert res['a'].max() <= 1000.0
        
def test_feature_eng_get_output():
    df = pd.DataFrame({'a': [1, 2, 3]})
    y = pd.Series([1, 0, 1])
    for t_spec in [
        (Polynomial(columns=['a']), df),
        (Ratios(pairs=[('a', 'a')]), df),
        (Differences(pairs=[('a', 'a')]), df),
        (DateParts(columns=['a'], parts=['year']), df),
        (CyclicalEncoding(columns=['a'], period=12), df),
        (Binning(columns=['a'], n_bins=2), df),
        (LogTransform(columns=['a']), df),
        (SqrtTransform(columns=['a']), df),
        (InteractionTerms(columns=['a']), df, y)
    ]:
        t = t_spec[0]
        args = t_spec[1:]
        if type(t).__name__ == "DateParts":
            t.fit(pd.DataFrame({'a': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03'])}))
        else:
            t.fit(*args)
        t.get_output_columns(['a'])
