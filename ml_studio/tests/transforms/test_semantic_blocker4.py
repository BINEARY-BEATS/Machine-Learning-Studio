import pytest
import pandas as pd
import numpy as np

from ml_studio.transforms.encoding import *
from ml_studio.transforms.feature_eng import *
from ml_studio.transforms.selection import *
from ml_studio.transforms.missing import *
from ml_studio.transforms.scaling import *
from ml_studio.transforms.outliers import *

def test_leave_one_out_full():
    df = pd.DataFrame({'a': ['A', 'A', 'B', 'B', 'C']})
    y = pd.Series([1, 1, 0, 0, 1])
    t = LeaveOneOut(columns=['a'], noise_level=0.1)
    # test fit_transform
    res_ft = t.fit_transform(df, y)
    
    # test transform
    res_t = t.transform(df)
    
    # test to_dict / from_dict
    d = t.to_dict()
    t2 = LeaveOneOut.from_dict(d)
    res_t2 = t2.transform(df)

def test_target_noise():
    df = pd.DataFrame({'a': ['A', 'A', 'B', 'B', 'C']})
    y = pd.Series([1, 1, 0, 0, 1])
    t = Target(columns=['a'], noise_level=0.1, min_samples_leaf=1)
    res_ft = t.fit_transform(df, y)
    res_t = t.transform(df)
    d = t.to_dict()
    t2 = Target.from_dict(d)
    t2.transform(df)

def test_woe_full():
    df = pd.DataFrame({'a': ['A', 'A', 'B', 'B', 'C']})
    y = pd.Series([1, 1, 0, 0, 1])
    t = WOE(columns=['a'])
    res_ft = t.fit_transform(df, y)
    res_t = t.transform(df)
    d = t.to_dict()
    t2 = WOE.from_dict(d)
    t2.transform(df)

def test_onehot_unseen_and_dict():
    df = pd.DataFrame({'a': ['A', 'B']})
    t = OneHot(columns=['a'], drop_first=True, handle_unknown='ignore')
    t.fit(df)
    d = t.to_dict()
    t2 = OneHot.from_dict(d)
    res = t2.transform(pd.DataFrame({'a': ['A', 'C']}))

def test_impute_missing_full():
    df = pd.DataFrame({'a': [1, np.nan, 3]})
    
    for strategy in ['mean', 'median', 'most_frequent', 'constant', 'knn']:
        if strategy == 'constant':
            t = Impute(columns=['a'], strategy='constant', constant_value=99)
        else:
            t = Impute(columns=['a'], strategy=strategy)
        t.fit(df)
        t.transform(df)
        d = t.to_dict()
        Impute.from_dict(d).transform(df)

def test_rfe_permutation_full():
    df = pd.DataFrame({'a': [1, 2, 3, 4], 'b': [1, 1, 1, 1], 'c': [1, 0, 1, 0]})
    y = pd.Series([1, 0, 1, 0])
    
    t1 = RFESelect(k=1)
    t1.fit(df, y)
    t1.transform(df)
    RFESelect.from_dict(t1.to_dict()).transform(df)
    
    t2 = PermutationSelect(k=1)
    t2.fit(df, y)
    t2.transform(df)
    PermutationSelect.from_dict(t2.to_dict()).transform(df)

def test_all_base_methods():
    df = pd.DataFrame({'a': [1, 2, 3]})
    t = Standard(columns=['a'])
    t.fit(df)
    
    # call transform
    t.transform(df)
    # call get_schema
    t.get_schema()
    # call to_dict
    d = t.to_dict()
    Standard.from_dict(d).transform(df)
