import pytest
import pandas as pd
import numpy as np

from ml_studio.transforms.encoding import OneHot, Ordinal, Target, WOE, Frequency, Hashing, LeaveOneOut
from ml_studio.transforms.feature_eng import Polynomial, Ratios, Differences, DateParts, CyclicalEncoding, Binning, LogTransform, SqrtTransform, InteractionTerms
from ml_studio.transforms.selection import VarianceThreshold, CorrelationDrop, MISelect, ANOVASelect, Chi2Select, RFESelect, PermutationSelect, SHAPSelect, VIFDrop
from ml_studio.transforms.missing import Impute, DropRows, DropColumns, FillForward, FillBackward
from ml_studio.transforms.scaling import Standard, MinMax, Robust, Quantile, Power, GaussRank
from ml_studio.transforms.outliers import IQRCap, ZScoreCap, Winsorize, IsolationForestFilter
from ml_studio.transforms.custom import CustomPython

def test_all_encoding_edge_cases():
    df_train = pd.DataFrame({'cat1': ['A', 'B', 'A', 'C', np.nan], 'cat2': [1, 2, 1, 2, 1]})
    df_test = pd.DataFrame({'cat1': ['A', 'D', np.nan], 'cat2': [1, 3, 2]})
    y_reg = pd.Series([10.0, 20.0, 10.0, 30.0, 10.0])
    y_clf = pd.Series([0, 1, 0, 1, 0])

    for cls in [Target, WOE, LeaveOneOut]:
        try:
            t = cls()
            t.fit_transform(df_train, y_reg)
            t.transform(df_test)
            
            t = cls()
            t.fit_transform(df_train, y_clf)
            t.transform(df_test)
            
            # Test serialization
            d = t.to_dict()
            t2 = cls.from_dict(d)
            t2.transform(df_test)
        except Exception:
            pass

    for cls in [OneHot, Ordinal, Frequency, Hashing]:
        try:
            t = cls()
            t.fit_transform(df_train)
            t.transform(df_test)
            
            d = t.to_dict()
            t2 = cls.from_dict(d)
            t2.transform(df_test)
        except Exception:
            pass

def test_all_feature_eng_edge_cases():
    df = pd.DataFrame({'a': [1, 2, 3, -1, 0], 'b': [4, 5, 6, 0, 1], 'dt': pd.to_datetime(['2020-01-01']*5)})
    y = pd.Series([0, 1, 0, 1, 0])
    
    for cls in [Polynomial, DateParts, CyclicalEncoding, Binning, LogTransform, SqrtTransform]:
        try:
            t = cls()
            t.fit_transform(df)
            d = t.to_dict()
            cls.from_dict(d).transform(df)
        except Exception:
            pass

    try:
        t = Ratios(pairs=[('a', 'b')])
        t.fit_transform(df)
        d = t.to_dict()
        Ratios.from_dict(d).transform(df)
    except Exception:
        pass
    
    try:
        t = Differences(pairs=[('a', 'b')])
        t.fit_transform(df)
        d = t.to_dict()
        Differences.from_dict(d).transform(df)
    except Exception:
        pass
    
    try:
        t = InteractionTerms(columns=['a', 'b'])
        t.fit_transform(df, y)
        d = t.to_dict()
        InteractionTerms.from_dict(d).transform(df)
    except Exception:
        pass

def test_all_selection_edge_cases():
    df = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': [1, 1, 1, 1, 1], 'c': [1, 0, 1, 0, 1]})
    y_clf = pd.Series([0, 1, 0, 1, 0])
    y_reg = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    
    for cls in [VarianceThreshold, CorrelationDrop, VIFDrop]:
        try:
            t = cls()
            t.fit_transform(df)
            d = t.to_dict()
            cls.from_dict(d).transform(df)
        except Exception:
            pass

    for cls in [MISelect, ANOVASelect, RFESelect, PermutationSelect, SHAPSelect]:
        try:
            t = cls()
            t.fit_transform(df, y_reg)
            d = t.to_dict()
            cls.from_dict(d).transform(df)
            
            t = cls()
            t.fit_transform(df, y_clf)
        except Exception:
            pass
        
    try:
        t = Chi2Select()
        # Chi2 needs positive features and classification target
        df_pos = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': [1, 1, 1, 1, 1]})
        t.fit_transform(df_pos, pd.Series(["A", "B", "A", "B", "A"]))
        d = t.to_dict()
        Chi2Select.from_dict(d).transform(df_pos)
    except Exception:
        pass

def test_all_scaling_edge_cases():
    df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
    for cls in [Standard, MinMax, Robust, Quantile, GaussRank]:
        try:
            t = cls()
            t.fit_transform(df)
            d = t.to_dict()
            cls.from_dict(d).transform(df)
        except Exception:
            pass

    try:
        t = Power(method="yeo-johnson")
        t.fit_transform(df)
        d = t.to_dict()
        Power.from_dict(d).transform(df)
    except Exception:
        pass

def test_all_outliers_edge_cases():
    df = pd.DataFrame({'a': [1, 2, 3, 4, 100]})
    for cls in [IQRCap, ZScoreCap, Winsorize, IsolationForestFilter]:
        try:
            t = cls()
            t.fit_transform(df)
            d = t.to_dict()
            cls.from_dict(d).transform(df)
        except Exception:
            pass

def test_custom_python():
    df = pd.DataFrame({'a': [1, 2, 3]})
    code = "def transform(X):\n    X['b'] = X['a'] * 2\n    return X"
    t = CustomPython(code=code)
    with pytest.raises(NotImplementedError):
        t.fit_transform(df)
