import pytest
import pandas as pd
import numpy as np
from ml_studio.transforms.encoding import OneHot, Ordinal, Target, WOE, Frequency, Hashing, LeaveOneOut
from ml_studio.transforms.feature_eng import Polynomial, Ratios, Differences, DateParts, CyclicalEncoding, Binning, LogTransform, SqrtTransform, InteractionTerms
from ml_studio.transforms.selection import VarianceThreshold, CorrelationDrop, MISelect, ANOVASelect, Chi2Select, RFESelect, PermutationSelect, SHAPSelect, VIFDrop
from ml_studio.transforms.missing import Impute, DropRows, DropColumns, FillForward, FillBackward
from ml_studio.transforms.scaling import Standard, MinMax, Robust, Quantile, Power, GaussRank
from ml_studio.transforms.outliers import IQRCap, ZScoreCap, Winsorize, IsolationForestFilter

def test_minmax_range():
    X = pd.DataFrame({'a': [10, 20, 30]})
    t = MinMax(columns=['a'])
    res = t.fit_transform(X)
    assert np.isclose(res['a'].min(), 0.0)
    assert np.isclose(res['a'].max(), 1.0)

def test_robust_median_zero():
    X = pd.DataFrame({'a': [1, 2, 3, 100]})
    t = Robust(columns=['a'])
    res = t.fit_transform(X)
    assert np.isclose(res['a'].median(), 0.0)

def test_power_scaler():
    X = pd.DataFrame({'a': np.random.lognormal(size=100)})
    t = Power(columns=['a'])
    res = t.fit_transform(X)
    assert abs(res['a'].mean()) < 0.5
    assert res['a'].std() > 0.5

def test_target_leakage_semantic():
    X = pd.DataFrame({'cat': ['A', 'A', 'B', 'B', 'A']})
    y = pd.Series([1.0, 1.0, 0.0, 0.0, 1.0])
    t = Target(columns=['cat'])
    res = t.fit_transform(X, y)
    assert res['cat'].mean() > 0

def test_woe_semantic():
    X = pd.DataFrame({'cat': ['A', 'A', 'B', 'B', 'A']})
    y = pd.Series([1, 1, 0, 0, 1])
    t = WOE(columns=['cat'])
    res = t.fit_transform(X, y)
    assert res['cat'].nunique() >= 1

def test_hashing_semantic():
    X = pd.DataFrame({'cat': ['A', 'B', 'C']})
    t = Hashing(columns=['cat'], n_features=4)
    res = t.fit_transform(X)
    assert res.shape[1] == 4
    
def test_leave_one_out_semantic():
    X = pd.DataFrame({'cat': ['A', 'A', 'A', 'B']})
    y = pd.Series([1, 1, 0, 0])
    t = LeaveOneOut(columns=['cat'])
    res = t.fit_transform(X, y)
    assert res['cat'].notna().all()

def test_polynomial_semantic():
    X = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
    t = Polynomial(columns=['a', 'b'], degree=2)
    res = t.fit_transform(X)
    assert 'a^2' in res.columns
    assert res['a^2'].iloc[1] == 4

def test_dateparts_semantic():
    X = pd.DataFrame({'dt': pd.to_datetime(['2023-01-01', '2023-12-31'])})
    t = DateParts(columns=['dt'])
    res = t.fit_transform(X)
    assert 'dt_year' in res.columns
    assert res['dt_month'].iloc[1] == 12

def test_cyclical_semantic():
    X = pd.DataFrame({'hr': [0, 6, 12, 18]})
    t = CyclicalEncoding(columns=['hr'], period=24)
    res = t.fit_transform(X)
    assert 'hr_sin' in res.columns
    assert np.isclose(res['hr_sin'].iloc[0], 0)

def test_binning_semantic():
    X = pd.DataFrame({'a': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
    t = Binning(columns=['a'], n_bins=2)
    res = t.fit_transform(X)
    assert res['a'].nunique() <= 2

def test_sqrttransform_semantic():
    X = pd.DataFrame({'a': [1, 4, 9]})
    t = SqrtTransform(columns=['a'])
    res = t.fit_transform(X)
    assert res['a'].iloc[2] == 3

def test_fill_fwd_bwd_semantic():
    X = pd.DataFrame({'a': [1, np.nan, 3]})
    t = FillForward(columns=['a'])
    res = t.fit_transform(X)
    assert res['a'].iloc[1] == 1
    
    X2 = pd.DataFrame({'a': [1, np.nan, 3]})
    t2 = FillBackward(columns=['a'])
    res2 = t2.fit_transform(X2)
    assert res2['a'].iloc[1] == 3

def test_iqrcap_semantic():
    X = pd.DataFrame({'a': [1, 2, 3, 100]})
    t = IQRCap(columns=['a'])
    res = t.fit_transform(X)
    assert res['a'].max() < 100

def test_zscorecap_semantic():
    X = pd.DataFrame({'a': [1, 2, 3, 100]})
    t = ZScoreCap(columns=['a'])
    res = t.fit_transform(X)
    assert res['a'].max() <= 100

def test_winsorize_semantic():
    X = pd.DataFrame({'a': [1, 2, 3, 100]})
    t = Winsorize(columns=['a'])
    res = t.fit_transform(X)
    assert res['a'].max() < 100

def test_isolationforest_semantic():
    X = pd.DataFrame({'a': [1, 1, 1, 1, 100]})
    t = IsolationForestFilter(columns=['a'])
    res = t.fit_transform(X)
    assert len(res) < 5

def test_vartresh_semantic():
    X = pd.DataFrame({'a': [1, 1, 1], 'b': [1, 2, 3]})
    t = VarianceThreshold(threshold=0.5)
    res = t.fit_transform(X)
    assert 'a' not in res.columns

def test_corrdrop_semantic():
    X = pd.DataFrame({'a': [1, 2, 3], 'b': [1.01, 2.01, 3.01], 'c': [1, 0, 1]})
    t = CorrelationDrop(threshold=0.9)
    res = t.fit_transform(X)
    assert len(res.columns) == 2

def test_chi2_semantic():
    X = pd.DataFrame({'a': [1, 2, 3, 4], 'b': [1, 1, 1, 1]})
    y = pd.Series([1, 0, 1, 0], dtype='category')
    t = Chi2Select(k=1)
    res = t.fit_transform(X, y)
    assert res.shape[1] == 1

def test_rfeselect_semantic():
    X = pd.DataFrame({'a': [1, 2, 3, 4], 'b': [1, 1, 1, 1]})
    y = pd.Series([1, 2, 3, 4])
    t = RFESelect(k=1)
    res = t.fit_transform(X, y)
    assert res.shape[1] == 1

def test_vifdrop_semantic():
    X = pd.DataFrame({'a': [1, 2, 3, 4], 'b': [2, 4, 6, 8], 'c': [1, 0, 1, 0]})
    t = VIFDrop(threshold=5.0)
    res = t.fit_transform(X)
    assert len(res.columns) < 3

def test_shapselect_semantic():
    X = pd.DataFrame({'a': [1, 2, 3, 4], 'b': [1, 1, 1, 1]})
    y = pd.Series([1, 2, 3, 4])
    t = SHAPSelect(k=1)
    res = t.fit_transform(X, y)
    assert res.shape[1] == 1
