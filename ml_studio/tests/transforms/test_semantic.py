import pytest
import pandas as pd
import numpy as np

from ml_studio.transforms.feature_eng import Polynomial, Ratios, Differences, DateParts, CyclicalEncoding, Binning, LogTransform, SqrtTransform, InteractionTerms
from ml_studio.transforms.missing import Impute, DropRows, DropColumns, FillForward, FillBackward
from ml_studio.transforms.outliers import IQRCap, ZScoreCap, Winsorize, IsolationForestFilter
from ml_studio.transforms.selection import VarianceThreshold, CorrelationDrop, MISelect, ANOVASelect, Chi2Select, RFESelect, PermutationSelect, SHAPSelect, VIFDrop
from ml_studio.transforms.custom import CustomPython

# ======================== FEATURE ENG ========================
def test_polynomial():
    df = pd.DataFrame({"a": [1, 2, 3]})
    t = Polynomial(degree=2, columns=["a"])
    res = t.fit_transform(df)
    assert "a^2" in res.columns
    assert res["a^2"].iloc[1] == 4

def test_ratios():
    df = pd.DataFrame({"num1": [10, 20], "num2": [2, 5], "num3": [0, 1]})
    t = Ratios(pairs=[("num1", "num2")])
    res = t.fit_transform(df)
    assert "num1_ratio_num2" in res.columns
    assert res["num1_ratio_num2"].iloc[0] == 5.0
    
    # zero division check
    t2 = Ratios(pairs=[("num1", "num3")])
    res2 = t2.fit_transform(df)
    assert np.isnan(res2["num1_ratio_num3"].iloc[0]) or np.isinf(res2["num1_ratio_num3"].iloc[0]) or res2["num1_ratio_num3"].iloc[0] == 0

def test_differences():
    df = pd.DataFrame({"a": [10, 20], "b": [2, 5]})
    t = Differences(pairs=[("a", "b")])
    res = t.fit_transform(df)
    assert "a_diff_b" in res.columns
    assert res["a_diff_b"].iloc[0] == 8

def test_dateparts():
    df = pd.DataFrame({"dt": pd.to_datetime(["2020-01-01", "2020-12-31"])})
    t = DateParts(columns=["dt"], parts=["year", "month", "day", "is_weekend"])
    res = t.fit_transform(df)
    assert res["dt_year"].iloc[0] == 2020
    assert res["dt_month"].iloc[1] == 12
    assert res["dt_day"].iloc[1] == 31
    assert "dt_is_weekend" in res.columns

def test_cyclical_encoding():
    df = pd.DataFrame({"month": [1, 6, 12]})
    t = CyclicalEncoding(columns=["month"], period=12)
    res = t.fit_transform(df)
    assert "month_sin" in res.columns
    assert "month_cos" in res.columns
    assert np.isclose(res["month_sin"].iloc[2], 0.0, atol=1e-5)

def test_binning():
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
    t = Binning(columns=["a"], strategy="equal_width", n_bins=2)
    res = t.fit_transform(df)
    assert res["a"].nunique() <= 2
    assert res["a"].iloc[0] == 0
    assert res["a"].iloc[-1] == 1

def test_logtransform():
    df = pd.DataFrame({"a": [0, 9, 99]})
    t = LogTransform(columns=["a"], offset=0)
    res = t.fit_transform(df)
    assert np.isclose(res["a"].iloc[0], 0.0)

def test_sqrttransform():
    df = pd.DataFrame({"a": [4, 9, 16]})
    t = SqrtTransform(columns=["a"])
    res = t.fit_transform(df)
    assert np.isclose(res["a"].iloc[0], 2.0)
    assert np.isclose(res["a"].iloc[1], 3.0)

def test_interaction_terms():
    df = pd.DataFrame({"a": [2, 3, 4, 5, 6], "b": [4, 5, 6, 7, 8], "target": [1, 0, 1, 0, 1]})
    t = InteractionTerms(columns=["a", "b"], top_k=1)
    res = t.fit_transform(df.drop("target", axis=1), df["target"])
    assert "a_x_b" in res.columns
    assert res["a_x_b"].iloc[0] == 8

# ======================== MISSING ========================
def test_impute_strategies():
    df = pd.DataFrame({"a": [1, np.nan, 3]})
    
    t1 = Impute(columns=["a"], strategy="mean")
    assert t1.fit_transform(df)["a"].iloc[1] == 2.0
    
    t2 = Impute(columns=["a"], strategy="median")
    assert t2.fit_transform(pd.DataFrame({"a": [1, np.nan, 10, 10]}))["a"].iloc[1] == 10.0
    
    df_cat = pd.DataFrame({"c": [20.0, np.nan, 20.0, 30.0]})
    t3 = Impute(columns=["c"], strategy="mode")
    assert t3.fit_transform(df_cat)["c"].iloc[1] == 20.0

def test_impute_knn_uses_neighbors():
    X = pd.DataFrame({
        "a": [1, 2, np.nan, 100, 101],
        "b": [1, 2, 3, 100, 101],
    })
    t = Impute(strategy="knn", knn_neighbors=2)
    res = t.fit_transform(X)
    assert 1.0 < res["a"].iloc[2] < 10.0

def test_drop_rows_cols():
    df = pd.DataFrame({
        "a": [1, 2, 3],
        "b": [np.nan, np.nan, np.nan]
    })
    t1 = DropColumns(threshold=0.5)
    assert "b" not in t1.fit_transform(df).columns
    assert "a" in t1.fit_transform(df).columns

    df2 = pd.DataFrame({
        "a": [1, np.nan, 3],
        "b": [np.nan, np.nan, np.nan]
    })
    t2 = DropRows(threshold=0.0, columns=["a", "b"]) # 0.0 threshold means drop if >0 missing
    assert len(t2.fit_transform(df2)) == 0 # all rows have at least one nan due to b

def test_fill_forward_backward():
    df = pd.DataFrame({"a": [1, np.nan, np.nan, 4]})
    res_f = FillForward().fit_transform(df)
    assert res_f["a"].iloc[1] == 1.0
    assert res_f["a"].iloc[2] == 1.0
    
    res_b = FillBackward().fit_transform(df)
    assert res_b["a"].iloc[1] == 4.0

# ======================== OUTLIERS ========================
def test_iqr_cap():
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5, 100]})
    t = IQRCap(columns=["a"], factor=1.5)
    res = t.fit_transform(df)
    assert res["a"].iloc[5] < 100
    assert res["a"].iloc[5] > 5

def test_zscore_cap():
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5, 1000]})
    t = ZScoreCap(columns=["a"], threshold=1.0)
    res = t.fit_transform(df)
    assert res["a"].iloc[5] < 1000

def test_winsorize():
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 100]})
    t = Winsorize(columns=["a"], limits=(0.1, 0.1))
    res = t.fit_transform(df)
    assert res["a"].max() < 100
    assert res["a"].min() >= 1

def test_isolation_forest():
    df = pd.DataFrame({"a": [1, 1, 1, 1, 1, 100]})
    t = IsolationForestFilter(columns=["a"], contamination=0.1)
    res = t.fit_transform(df)
    assert len(res) < 6
    assert 100 not in res["a"].values

# ======================== SELECTION ========================
def test_variance_threshold():
    df = pd.DataFrame({"a": [1, 1, 1], "b": [1, 2, 3]})
    t = VarianceThreshold(threshold=0.5)
    res = t.fit_transform(df)
    assert "a" not in res.columns
    assert "b" in res.columns

def test_correlation_drop():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [1.1, 2.1, 3.1], "c": [1, 0, 1]})
    t = CorrelationDrop(threshold=0.9)
    res = t.fit_transform(df)
    assert len(res.columns) == 2
    assert "c" in res.columns

def test_mi_select():
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [1, 1, 1, 1, 1]})
    target = pd.Series([1, 2, 3, 4, 5])
    t = MISelect(k=1)
    res = t.fit_transform(df, target)
    assert "a" in res.columns
    assert "b" not in res.columns

def test_anova_select():
    df = pd.DataFrame({"a": [1, 10, 1, 10], "b": [1, 1, 1, 1]})
    target = pd.Series([0, 1, 0, 1])
    t = ANOVASelect(k=1)
    res = t.fit_transform(df, target)
    assert "a" in res.columns

def test_chi2_select():
    df = pd.DataFrame({"a": [0, 10, 0, 10]*5, "b": [1, 1, 1, 1]*5})
    # Target must be strings for classification detection
    target = pd.Series(["A", "B", "A", "B"]*5)
    t = Chi2Select(k=1)
    res = t.fit_transform(df, target)
    assert "a" in res.columns

def test_rfe_select():
    df = pd.DataFrame({"a": [1, 2, 3, 4], "b": [0, 0, 0, 0], "target": [1, 2, 3, 4]})
    t = RFESelect(k=1, estimator="linear")
    res = t.fit_transform(df, df["target"])
    assert "a" in res.columns

def test_permutation_select():
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [1, 0, 1, 0, 1]})
    target = pd.Series([1, 2, 3, 4, 5])
    t = PermutationSelect(k=1)
    res = t.fit_transform(df, target)
    assert "a" in res.columns

def test_vif_drop():
    df = pd.DataFrame({"a": [1, 2, 3, 4], "b": [2, 4, 6, 8], "c": [1, 0, 1, 0]}) # b is exactly 2*a
    t = VIFDrop(threshold=5.0)
    res = t.fit_transform(df)
    assert len(res.columns) < 3

def test_shap_select():
    df = pd.DataFrame({"a": [1, 2, 3, 4], "b": [1, 1, 1, 1]})
    target = pd.Series([1, 2, 3, 4])
    t = SHAPSelect(k=1)
    res = t.fit_transform(df, target)
    assert "a" in res.columns
