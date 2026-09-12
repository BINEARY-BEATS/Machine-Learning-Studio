import pytest
import pandas as pd
import numpy as np
from ml_studio.transforms.encoding import OneHot, Ordinal, Target, WOE, Frequency, Hashing, LeaveOneOut

def test_target_schema():
    schema = Target.get_schema()
    assert "cv" in schema["properties"]

def test_target_leakage():
    # Setup: two categories, distinct target distributions
    X_train = pd.DataFrame({"cat": ["A"]*5 + ["B"]*5})
    # A: [100, 0, 100, 0, 100]  → own-value mean = 60
    # B: [10, 20, 10, 20, 10]   → own-value mean = 14
    y_train = pd.Series([100, 0, 100, 0, 100, 10, 20, 10, 20, 10])
    
    # We use a custom cv scheme to ensure folds are deterministic,
    # but since StratifiedKFold or KFold are used internally, we can't easily hardcode fold indices.
    # However, we DO know the OOF mean must not equal the own target.
    # Actually, we can just use 5-fold, meaning each fold has size 1 for each category.
    # So if A has target 100 in fold 1, its OOF mean is (0 + 100 + 0 + 100) / 4 = 50.
    t = Target(cv=5, smoothing=0, min_samples_leaf=1)
    res_train = t.fit_transform(X_train, y_train)
    
    # Row 0 (cat A, target 100) must NOT see its own target.
    # With 5-fold CV (assuming KFold sequentially or evenly), the mean for A-rows is either 50 or 75.
    # So it should NOT be 100 and NOT be 60.
    val0 = res_train["cat"].iloc[0]
    assert np.isclose(val0, 50.0, atol=0.01) or np.isclose(val0, 75.0, atol=0.01)
    
    # Similarly, test Transform behavior uses global means
    X_test = pd.DataFrame({"cat": ["A", "B", "C"]})
    res_test = t.transform(X_test)
    assert np.isclose(res_test["cat"].iloc[0], 60.0)
    assert np.isclose(res_test["cat"].iloc[1], 14.0)

def test_woe_fit_transform():
    # WOE leakage test
    X_train = pd.DataFrame({"cat": ["A", "A", "B", "B", "C", "C", "A", "B"]})
    y_train = pd.Series([1, 0, 1, 0, 1, 0, 1, 1])
    
    t = WOE(cv=2, smoothing=0)
    res = t.fit_transform(X_train, y_train)
    assert "cat" in res.columns
    # Check that fit_transform does not exactly match transform (which uses global means)
    res_test = t.transform(X_train)
    # The out-of-fold encoded values should generally not exactly equal the global encoded values
    # unless there is no leakage.
    assert not np.allclose(res["cat"], res_test["cat"])
    
def test_onehot_serialize():
    X_train = pd.DataFrame({"cat": ["A", "B", "A"]})
    t = OneHot()
    t.fit(X_train)
    
    d = t.to_dict()
    assert "A" in d["categories_"]["cat"]
    
    t2 = OneHot.from_dict(d)
    assert t2.categories_ == t.categories_
    
def test_ordinal():
    X_train = pd.DataFrame({"cat": ["Low", "High", "Medium"]})
    t = Ordinal(categories={"cat": ["Low", "Medium", "High"]})
    res = t.fit_transform(X_train)
    assert res["cat"].iloc[0] == 0
    assert res["cat"].iloc[1] == 2
    assert res["cat"].iloc[2] == 1

def test_hashing_deterministic():
    X1 = pd.DataFrame({"cat": ["A", "B", "C"]})
    t1 = Hashing(n_features=5)
    r1 = t1.fit_transform(X1)
    
    t2 = Hashing(n_features=5)
    r2 = t2.fit_transform(X1)
    
    pd.testing.assert_frame_equal(r1, r2)
