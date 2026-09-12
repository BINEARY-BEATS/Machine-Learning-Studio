"""Outlier handling transforms."""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from .base import BaseTransform


class IQRCap(BaseTransform):
    """Cap outliers using Interquartile Range (IQR)."""

    def __init__(self, columns=None, factor=1.5):
        super().__init__(columns=columns, factor=factor)
        self.columns = columns
        self.factor = factor
        self.lower_bounds_ = {}
        self.upper_bounds_ = {}

    @classmethod
    def get_schema(cls) -> dict:
        return {
            "type": "object",
            "properties": {
                "columns": {"type": "array", "items": {"type": "string"}, "default": None},
                "factor": {"type": "number", "default": 1.5}
            }
        }

    def _fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> None:
        self.fitted_columns_ = self.columns if self.columns is not None else [
            c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])
        ]
        
        for col in self.fitted_columns_:
            q1 = X[col].quantile(0.25)
            q3 = X[col].quantile(0.75)
            iqr = q3 - q1
            self.lower_bounds_[col] = float(q1 - self.factor * iqr)
            self.upper_bounds_[col] = float(q3 + self.factor * iqr)

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        for col in self.fitted_columns_:
            lower = self.lower_bounds_.get(col, -np.inf)
            upper = self.upper_bounds_.get(col, np.inf)
            X[col] = X[col].clip(lower=lower, upper=upper)
        return X

    def to_dict(self) -> dict:
        d = self.params.copy()
        d["lower_bounds_"] = self.lower_bounds_
        d["upper_bounds_"] = self.upper_bounds_
        d["fitted_columns_"] = self.fitted_columns_
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "BaseTransform":
        obj = cls(columns=d.get("columns", None), factor=d.get("factor", 1.5))
        obj.lower_bounds_ = d.get("lower_bounds_", {})
        obj.upper_bounds_ = d.get("upper_bounds_", {})
        obj.fitted_columns_ = d.get("fitted_columns_", [])
        obj._is_fitted = True
        return obj


class ZScoreCap(BaseTransform):
    """Cap outliers using Z-Score."""

    def __init__(self, columns=None, threshold=3.0):
        super().__init__(columns=columns, threshold=threshold)
        self.columns = columns
        self.threshold = threshold
        self.lower_bounds_ = {}
        self.upper_bounds_ = {}

    @classmethod
    def get_schema(cls) -> dict:
        return {
            "type": "object",
            "properties": {
                "columns": {"type": "array", "items": {"type": "string"}, "default": None},
                "threshold": {"type": "number", "default": 3.0}
            }
        }

    def _fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> None:
        self.fitted_columns_ = self.columns if self.columns is not None else [
            c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])
        ]
        
        for col in self.fitted_columns_:
            mean = X[col].mean()
            std = X[col].std()
            if pd.isna(std) or std == 0:
                std = 1.0  # Prevent zero-variance issues
            self.lower_bounds_[col] = float(mean - self.threshold * std)
            self.upper_bounds_[col] = float(mean + self.threshold * std)

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        for col in self.fitted_columns_:
            lower = self.lower_bounds_.get(col, -np.inf)
            upper = self.upper_bounds_.get(col, np.inf)
            X[col] = X[col].clip(lower=lower, upper=upper)
        return X

    def to_dict(self) -> dict:
        d = self.params.copy()
        d["lower_bounds_"] = self.lower_bounds_
        d["upper_bounds_"] = self.upper_bounds_
        d["fitted_columns_"] = self.fitted_columns_
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "BaseTransform":
        obj = cls(columns=d.get("columns", None), threshold=d.get("threshold", 3.0))
        obj.lower_bounds_ = d.get("lower_bounds_", {})
        obj.upper_bounds_ = d.get("upper_bounds_", {})
        obj.fitted_columns_ = d.get("fitted_columns_", [])
        obj._is_fitted = True
        return obj


class Winsorize(BaseTransform):
    """Cap outliers to specific percentiles."""

    def __init__(self, columns=None, limits=(0.01, 0.01)):
        super().__init__(columns=columns, limits=limits)
        self.columns = columns
        self.limits = tuple(limits)
        self.lower_bounds_ = {}
        self.upper_bounds_ = {}

    @classmethod
    def get_schema(cls) -> dict:
        return {
            "type": "object",
            "properties": {
                "columns": {"type": "array", "items": {"type": "string"}, "default": None},
                "limits": {"type": "array", "items": {"type": "number"}, "default": [0.01, 0.01]}
            }
        }

    def _fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> None:
        self.fitted_columns_ = self.columns if self.columns is not None else [
            c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])
        ]
        
        for col in self.fitted_columns_:
            lower = X[col].quantile(self.limits[0])
            upper = X[col].quantile(1.0 - self.limits[1])
            self.lower_bounds_[col] = float(lower)
            self.upper_bounds_[col] = float(upper)

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        for col in self.fitted_columns_:
            lower = self.lower_bounds_.get(col, -np.inf)
            upper = self.upper_bounds_.get(col, np.inf)
            X[col] = X[col].clip(lower=lower, upper=upper)
        return X

    def to_dict(self) -> dict:
        d = self.params.copy()
        d["lower_bounds_"] = self.lower_bounds_
        d["upper_bounds_"] = self.upper_bounds_
        d["fitted_columns_"] = self.fitted_columns_
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "BaseTransform":
        obj = cls(columns=d.get("columns", None), limits=d.get("limits", [0.01, 0.01]))
        obj.lower_bounds_ = d.get("lower_bounds_", {})
        obj.upper_bounds_ = d.get("upper_bounds_", {})
        obj.fitted_columns_ = d.get("fitted_columns_", [])
        obj._is_fitted = True
        return obj


class IsolationForestFilter(BaseTransform):
    """Drop rows identified as outliers by Isolation Forest."""

    def __init__(self, columns=None, contamination=0.1):
        super().__init__(columns=columns, contamination=contamination)
        self.columns = columns
        self.contamination = contamination

    @classmethod
    def get_schema(cls) -> dict:
        return {
            "type": "object",
            "properties": {
                "columns": {"type": "array", "items": {"type": "string"}, "default": None},
                "contamination": {"type": "number", "default": 0.1}
            }
        }

    def _fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> None:
        self.fitted_columns_ = self.columns if self.columns is not None else [
            c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])
        ]
        if not self.fitted_columns_:
            return

        self.model = IsolationForest(
            contamination=self.contamination, 
            random_state=42
        )
        # Dropna so model doesn't fail, but we'll apply it on training data if needed
        clean_X = X[self.fitted_columns_].dropna()
        if len(clean_X) > 0:
            self.model.fit(clean_X)

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted_columns_ or not hasattr(self, 'model'):
            return X
            
        # We must decide how to handle NaNs in transform.
        # IF cannot predict on NaNs.
        # We will keep rows that have NaNs, and only filter clean rows that are outliers.
        clean_mask = X[self.fitted_columns_].notna().all(axis=1)
        if not clean_mask.any():
            return X
            
        predictions = pd.Series(1, index=X.index)
        predictions.loc[clean_mask] = self.model.predict(X.loc[clean_mask, self.fitted_columns_])
        
        # 1 for inliers, -1 for outliers
        return X[predictions == 1].copy()

    def to_dict(self) -> dict:
        d = self.params.copy()
        d["fitted_columns_"] = self.fitted_columns_
        # To serialize IF completely requires storing all trees. We omit for now, 
        # but in a production system we'd use joblib or ONNX for the model object.
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "BaseTransform":
        obj = cls(columns=d.get("columns", None), contamination=d.get("contamination", 0.1))
        obj.fitted_columns_ = d.get("fitted_columns_", [])
        # We cannot easily re-instantiate an already fitted IF from a simple dict. 
        # This is a known limitation when saving sklearn ensemble models via json.
        obj._is_fitted = True
        return obj
