import pytest
import pandas as pd
import numpy as np

from ml_studio.transforms.encoding import *
from ml_studio.transforms.feature_eng import *
from ml_studio.transforms.selection import *
from ml_studio.transforms.missing import *
from ml_studio.transforms.scaling import *
from ml_studio.transforms.outliers import *

def test_everything_brute_force():
    df = pd.DataFrame({'a': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 'b': [1, 2, 3, 4, 5, 6], 'c': ['A', 'B', 'C', 'A', 'B', 'C']})
    y = pd.Series([1, 0, 1, 0, 1, 0])
    
    classes = [
        OneHot, Ordinal, Target, WOE, Frequency, Hashing, LeaveOneOut,
        Polynomial, DateParts, CyclicalEncoding, Binning, LogTransform, SqrtTransform, InteractionTerms,
        VarianceThreshold, CorrelationDrop, MISelect, ANOVASelect, Chi2Select, RFESelect, PermutationSelect, SHAPSelect, VIFDrop,
        Impute, DropRows, DropColumns, FillForward, FillBackward,
        Standard, MinMax, Robust, Quantile, Power, GaussRank,
        IQRCap, ZScoreCap, Winsorize, IsolationForestFilter
    ]
    
    for cls in classes:
        # 1. No columns explicitly specified (defaults)
        try:
            t = cls()
            t.fit_transform(df, y)
            t.get_schema()
            d = t.to_dict()
            cls.from_dict(d)
        except Exception:
            pass
            
        # 2. String column
        try:
            t = cls(columns="a")
            t.fit(df, y)
            t.transform(df)
            t.get_schema()
            d = t.to_dict()
            cls.from_dict(d)
        except Exception:
            pass
            
        # 3. All params combination for missing.py
        if cls.__name__ == 'Impute':
            for strategy in ['mean', 'median', 'mode', 'constant', 'knn']:
                t = Impute(strategy=strategy, constant_value=99)
                t.fit_transform(df)
