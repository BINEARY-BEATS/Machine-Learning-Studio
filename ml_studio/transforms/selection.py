"""Feature selection transforms."""

import numpy as np
import pandas as pd
from sklearn.feature_selection import (
    VarianceThreshold as SklearnVarianceThreshold,
    mutual_info_classif, mutual_info_regression,
    f_classif, f_regression, chi2, RFE
)
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.inspection import permutation_importance
from statsmodels.stats.outliers_influence import variance_inflation_factor

from .base import BaseTransform
from ml_studio.core.schema import detect_task_type


class VarianceThreshold(BaseTransform):
    """Drop features with variance below threshold."""

    def __init__(self, threshold=0.01):
        super().__init__(threshold=threshold)
        self.threshold = threshold
        self.dropped_cols_ = []

    @classmethod
    def get_schema(cls) -> dict:
        return {
            "type": "object",
            "properties": {
                "threshold": {"type": "number", "default": 0.01}
            }
        }

    def _fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> None:
        self.fitted_columns_ = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
        if not self.fitted_columns_:
            return

        sel = SklearnVarianceThreshold(threshold=self.threshold)
        # Handle NaNs temporarily for fitting
        sel.fit(X[self.fitted_columns_].fillna(0))
        
        support = sel.get_support()
        self.dropped_cols_ = [col for i, col in enumerate(self.fitted_columns_) if not support[i]]

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        cols_to_drop = [c for c in self.dropped_cols_ if c in X.columns]
        if cols_to_drop:
            return X.drop(columns=cols_to_drop)
        return X

    def get_output_columns(self, input_columns: list[str]) -> list[str]:
        return [c for c in input_columns if c not in getattr(self, "dropped_cols_", [])]

    def to_dict(self) -> dict:
        d = self.params.copy()
        d["dropped_cols_"] = self.dropped_cols_
        d["fitted_columns_"] = self.fitted_columns_
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "BaseTransform":
        obj = cls(threshold=d.get("threshold", 0.01))
        obj.dropped_cols_ = d.get("dropped_cols_", [])
        obj.fitted_columns_ = d.get("fitted_columns_", [])
        obj._is_fitted = True
        return obj


class CorrelationDrop(BaseTransform):
    """Drop highly correlated features."""

    def __init__(self, threshold=0.95, keep="first"):
        super().__init__(threshold=threshold, keep=keep)
        self.threshold = threshold
        self.keep = keep
        self.dropped_cols_ = []

    @classmethod
    def get_schema(cls) -> dict:
        return {
            "type": "object",
            "properties": {
                "threshold": {"type": "number", "default": 0.95},
                "keep": {"type": "string", "enum": ["first"], "default": "first"}
            }
        }

    def _fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> None:
        self.fitted_columns_ = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
        if not self.fitted_columns_:
            return

        corr_matrix = X[self.fitted_columns_].corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        self.dropped_cols_ = [column for column in upper.columns if any(upper[column] >= self.threshold)]

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        cols_to_drop = [c for c in self.dropped_cols_ if c in X.columns]
        if cols_to_drop:
            return X.drop(columns=cols_to_drop)
        return X

    def get_output_columns(self, input_columns: list[str]) -> list[str]:
        return [c for c in input_columns if c not in getattr(self, "dropped_cols_", [])]

    def to_dict(self) -> dict:
        d = self.params.copy()
        d["dropped_cols_"] = self.dropped_cols_
        d["fitted_columns_"] = self.fitted_columns_
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "BaseTransform":
        obj = cls(threshold=d.get("threshold", 0.95), keep=d.get("keep", "first"))
        obj.dropped_cols_ = d.get("dropped_cols_", [])
        obj.fitted_columns_ = d.get("fitted_columns_", [])
        obj._is_fitted = True
        return obj


class MISelect(BaseTransform):
    """Top-K features by mutual information."""

    def __init__(self, k=20):
        super().__init__(k=k)
        self.k = k
        self.dropped_cols_ = []

    @classmethod
    def get_schema(cls) -> dict:
        return {
            "type": "object",
            "properties": {
                "k": {"type": "integer", "default": 20}
            }
        }

    def _fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> None:
        if y is None:
            raise ValueError("MISelect requires a target (y).")
            
        self.fitted_columns_ = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
        if not self.fitted_columns_ or len(self.fitted_columns_) <= self.k:
            return

        task = detect_task_type(y)
        mi_func = mutual_info_classif if task == "CLASSIFICATION" else mutual_info_regression
        
        scores = mi_func(X[self.fitted_columns_].fillna(0), y, random_state=42)
        top_k_indices = np.argsort(scores)[-self.k:]
        keep_cols = set([self.fitted_columns_[i] for i in top_k_indices])
        
        self.dropped_cols_ = [c for c in self.fitted_columns_ if c not in keep_cols]

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        cols_to_drop = [c for c in self.dropped_cols_ if c in X.columns]
        if cols_to_drop:
            return X.drop(columns=cols_to_drop)
        return X

    def get_output_columns(self, input_columns: list[str]) -> list[str]:
        return [c for c in input_columns if c not in getattr(self, "dropped_cols_", [])]

    def to_dict(self) -> dict:
        d = self.params.copy()
        d["dropped_cols_"] = self.dropped_cols_
        d["fitted_columns_"] = self.fitted_columns_
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "BaseTransform":
        obj = cls(k=d.get("k", 20))
        obj.dropped_cols_ = d.get("dropped_cols_", [])
        obj.fitted_columns_ = d.get("fitted_columns_", [])
        obj._is_fitted = True
        return obj


class ANOVASelect(MISelect):
    """Top-K features by ANOVA F-value."""

    def _fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> None:
        if y is None:
            raise ValueError("ANOVASelect requires a target (y).")
            
        self.fitted_columns_ = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
        if not self.fitted_columns_ or len(self.fitted_columns_) <= self.k:
            return

        task = detect_task_type(y)
        f_func = f_classif if task == "CLASSIFICATION" else f_regression
        
        scores, _ = f_func(X[self.fitted_columns_].fillna(0), y)
        scores = np.nan_to_num(scores)
        top_k_indices = np.argsort(scores)[-self.k:]
        keep_cols = set([self.fitted_columns_[i] for i in top_k_indices])
        
        self.dropped_cols_ = [c for c in self.fitted_columns_ if c not in keep_cols]


class Chi2Select(MISelect):
    """Top-K features by Chi-Squared (Classification only, non-negative features)."""

    def _fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> None:
        if y is None:
            raise ValueError("Chi2Select requires a target (y).")
            
        task = detect_task_type(y)
        if task != "CLASSIFICATION":
            raise ValueError("Chi2Select is for classification tasks only.")
            
        self.fitted_columns_ = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
        if not self.fitted_columns_ or len(self.fitted_columns_) <= self.k:
            return

        # Ensure non-negative
        clean_X = X[self.fitted_columns_].fillna(0)
        if (clean_X < 0).any().any():
            clean_X = clean_X - clean_X.min()
            
        scores, _ = chi2(clean_X, y)
        scores = np.nan_to_num(scores)
        top_k_indices = np.argsort(scores)[-self.k:]
        keep_cols = set([self.fitted_columns_[i] for i in top_k_indices])
        
        self.dropped_cols_ = [c for c in self.fitted_columns_ if c not in keep_cols]


class RFESelect(BaseTransform):
    """Recursive Feature Elimination."""

    def __init__(self, k=20, estimator="linear"):
        super().__init__(k=k, estimator=estimator)
        self.k = k
        self.estimator = estimator
        self.dropped_cols_ = []

    @classmethod
    def get_schema(cls) -> dict:
        return {
            "type": "object",
            "properties": {
                "k": {"type": "integer", "default": 20},
                "estimator": {"type": "string", "enum": ["linear"], "default": "linear"}
            }
        }

    def _fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> None:
        if y is None:
            raise ValueError("RFESelect requires a target (y).")
            
        self.fitted_columns_ = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
        if not self.fitted_columns_ or len(self.fitted_columns_) <= self.k:
            return

        task = detect_task_type(y)
        if task == "CLASSIFICATION":
            est = LogisticRegression(random_state=42, max_iter=1000)
        else:
            est = Ridge(random_state=42)
            
        rfe = RFE(estimator=est, n_features_to_select=self.k)
        rfe.fit(X[self.fitted_columns_].fillna(0), y)
        
        support = rfe.support_
        self.dropped_cols_ = [col for i, col in enumerate(self.fitted_columns_) if not support[i]]

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        cols_to_drop = [c for c in self.dropped_cols_ if c in X.columns]
        if cols_to_drop:
            return X.drop(columns=cols_to_drop)
        return X

    def get_output_columns(self, input_columns: list[str]) -> list[str]:
        return [c for c in input_columns if c not in getattr(self, "dropped_cols_", [])]

    def to_dict(self) -> dict:
        d = self.params.copy()
        d["dropped_cols_"] = self.dropped_cols_
        d["fitted_columns_"] = self.fitted_columns_
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "BaseTransform":
        obj = cls(k=d.get("k", 20), estimator=d.get("estimator", "linear"))
        obj.dropped_cols_ = d.get("dropped_cols_", [])
        obj.fitted_columns_ = d.get("fitted_columns_", [])
        obj._is_fitted = True
        return obj


class PermutationSelect(BaseTransform):
    """Feature selection via permutation importance on a linear model."""

    def __init__(self, k=20):
        super().__init__(k=k)
        self.k = k
        self.dropped_cols_ = []

    @classmethod
    def get_schema(cls) -> dict:
        return {
            "type": "object",
            "properties": {
                "k": {"type": "integer", "default": 20}
            }
        }

    def _fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> None:
        if y is None:
            raise ValueError("PermutationSelect requires a target (y).")
            
        self.fitted_columns_ = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
        if not self.fitted_columns_ or len(self.fitted_columns_) <= self.k:
            return

        task = detect_task_type(y)
        if task == "CLASSIFICATION":
            est = LogisticRegression(random_state=42, max_iter=1000)
        else:
            est = Ridge(random_state=42)
            
        clean_X = X[self.fitted_columns_].fillna(0)
        est.fit(clean_X, y)
        
        result = permutation_importance(est, clean_X, y, n_repeats=5, random_state=42)
        top_k_indices = np.argsort(result.importances_mean)[-self.k:]
        keep_cols = set([self.fitted_columns_[i] for i in top_k_indices])
        
        self.dropped_cols_ = [c for c in self.fitted_columns_ if c not in keep_cols]

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        cols_to_drop = [c for c in self.dropped_cols_ if c in X.columns]
        if cols_to_drop:
            return X.drop(columns=cols_to_drop)
        return X
        
    def get_output_columns(self, input_columns: list[str]) -> list[str]:
        return [c for c in input_columns if c not in getattr(self, "dropped_cols_", [])]

    def to_dict(self) -> dict:
        d = self.params.copy()
        d["dropped_cols_"] = self.dropped_cols_
        d["fitted_columns_"] = self.fitted_columns_
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "BaseTransform":
        obj = cls(k=d.get("k", 20))
        obj.dropped_cols_ = d.get("dropped_cols_", [])
        obj.fitted_columns_ = d.get("fitted_columns_", [])
        obj._is_fitted = True
        return obj


class SHAPSelect(PermutationSelect):
    """SHAP Feature Selection. Currently aliases to PermutationSelect. Real SHAP in Phase 4."""

    @classmethod
    def get_schema(cls) -> dict:
        return {
            "type": "object",
            "properties": {
                "k": {"type": "integer", "default": 20},
                "backend": {
                    "type": "string", 
                    "default": "permutation",
                    "description": "SHAP backend available in Phase 4."
                }
            }
        }


class VIFDrop(BaseTransform):
    """Drop columns with high Variance Inflation Factor."""

    def __init__(self, threshold=10.0):
        super().__init__(threshold=threshold)
        self.threshold = threshold
        self.dropped_cols_ = []

    @classmethod
    def get_schema(cls) -> dict:
        return {
            "type": "object",
            "properties": {
                "threshold": {"type": "number", "default": 10.0}
            }
        }

    def _fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> None:
        self.fitted_columns_ = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
        if not self.fitted_columns_:
            return

        clean_X = X[self.fitted_columns_].fillna(0)
        cols_to_check = list(self.fitted_columns_)
        
        # Iterative drop: drop the highest VIF column until all < threshold
        # Max 50 iterations to prevent infinite loop on large datasets
        for _ in range(50):
            if len(cols_to_check) < 2:
                break
                
            vals = clean_X[cols_to_check].values
            vifs = []
            for i in range(len(cols_to_check)):
                try:
                    vif = variance_inflation_factor(vals, i)
                except Exception:
                    vif = np.inf
                vifs.append(vif)
                
            max_vif = max(vifs)
            if max_vif > self.threshold:
                max_idx = vifs.index(max_vif)
                self.dropped_cols_.append(cols_to_check[max_idx])
                cols_to_check.pop(max_idx)
            else:
                break

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        cols_to_drop = [c for c in self.dropped_cols_ if c in X.columns]
        if cols_to_drop:
            return X.drop(columns=cols_to_drop)
        return X

    def get_output_columns(self, input_columns: list[str]) -> list[str]:
        return [c for c in input_columns if c not in getattr(self, "dropped_cols_", [])]

    def to_dict(self) -> dict:
        d = self.params.copy()
        d["dropped_cols_"] = self.dropped_cols_
        d["fitted_columns_"] = self.fitted_columns_
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "BaseTransform":
        obj = cls(threshold=d.get("threshold", 10.0))
        obj.dropped_cols_ = d.get("dropped_cols_", [])
        obj.fitted_columns_ = d.get("fitted_columns_", [])
        obj._is_fitted = True
        return obj
