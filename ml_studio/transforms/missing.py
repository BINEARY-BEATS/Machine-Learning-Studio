"""Missing value imputation transforms."""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer

from .base import BaseTransform


class Impute(BaseTransform):
    """Impute missing values using various strategies."""

    def __init__(self, strategy="median", columns=None, constant_value=0, knn_neighbors=5):
        super().__init__(strategy=strategy, columns=columns, constant_value=constant_value, knn_neighbors=knn_neighbors)
        self.strategy = strategy
        self.columns = columns
        self.constant_value = constant_value
        self.knn_neighbors = knn_neighbors
        self.imputers_ = {}  # {col_name: learned_value or sklearn_imputer}

    @classmethod
    def get_schema(cls) -> dict:
        return {
            "type": "object",
            "properties": {
                "strategy": {"type": "string", "enum": ["mean", "median", "mode", "constant", "knn"], "default": "median"},
                "columns": {"type": "array", "items": {"type": "string"}, "default": None},
                "constant_value": {"type": ["number", "string"], "default": 0},
                "knn_neighbors": {"type": "integer", "default": 5}
            }
        }

    def _fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> None:
        if self.columns is None:
            self.fitted_columns_ = list(X.columns)
        else:
            self.fitted_columns_ = self.columns

        if self.strategy == "knn":
            # knn imputer works on all fitted_columns together
            # but requires numeric data
            numeric_cols = [c for c in self.fitted_columns_ if pd.api.types.is_numeric_dtype(X[c])]
            self.fitted_columns_ = numeric_cols
            if numeric_cols:
                self.imputer = KNNImputer(n_neighbors=self.knn_neighbors)
                self.imputer.fit(X[numeric_cols])
            return

        for col in self.fitted_columns_:
            series = X[col].dropna()
            if len(series) == 0:
                self.imputers_[col] = self.constant_value
                continue

            if self.strategy == "mean":
                if pd.api.types.is_numeric_dtype(series):
                    self.imputers_[col] = series.mean()
                else:
                    self.imputers_[col] = series.mode().iloc[0] if not series.mode().empty else self.constant_value
            elif self.strategy == "median":
                if pd.api.types.is_numeric_dtype(series):
                    self.imputers_[col] = series.median()
                else:
                    self.imputers_[col] = series.mode().iloc[0] if not series.mode().empty else self.constant_value
            elif self.strategy == "mode":
                self.imputers_[col] = series.mode().iloc[0] if not series.mode().empty else self.constant_value
            elif self.strategy == "constant":
                self.imputers_[col] = self.constant_value

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.strategy == "knn":
            if hasattr(self, 'imputer') and self.fitted_columns_:
                X[self.fitted_columns_] = self.imputer.transform(X[self.fitted_columns_])
            return X
            
        for col in self.fitted_columns_:
            fill_val = self.imputers_.get(col, self.constant_value)
            X[col] = X[col].fillna(fill_val)
        return X

    def to_dict(self) -> dict:
        from ml_studio.core.serialization import to_jsonable

        d = self.params.copy()
        if self.strategy == "knn" and hasattr(self, "imputer"):
            d["knn_fit_X_"] = to_jsonable(getattr(self.imputer, "_fit_X", None))
            d["knn_mask_fit_X_"] = to_jsonable(getattr(self.imputer, "_mask_fit_X", None))
            d["knn_valid_mask_"] = to_jsonable(getattr(self.imputer, "_valid_mask", None))
            d["knn_n_features_in_"] = int(getattr(self.imputer, "n_features_in_", 0))
        d["imputers_"] = to_jsonable(self.imputers_)
        d["fitted_columns_"] = self.fitted_columns_
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "BaseTransform":
        from ml_studio.transforms.base import mark_fitted

        obj = cls(
            strategy=d.get("strategy", "median"), 
            columns=d.get("columns", None), 
            constant_value=d.get("constant_value", 0),
            knn_neighbors=d.get("knn_neighbors", 5)
        )
        obj.imputers_ = d.get("imputers_", {})
        obj.fitted_columns_ = d.get("fitted_columns_", [])
        if obj.strategy == "knn":
            ready = d.get("knn_fit_X_") is not None and bool(obj.fitted_columns_)
            if ready:
                obj.imputer = KNNImputer(n_neighbors=obj.knn_neighbors)
                fit_x = np.asarray(d["knn_fit_X_"], dtype=float)
                obj.imputer._fit_X = fit_x
                mask = d.get("knn_mask_fit_X_")
                obj.imputer._mask_fit_X = (
                    np.asarray(mask, dtype=bool) if mask is not None else np.isnan(fit_x)
                )
                valid = d.get("knn_valid_mask_")
                obj.imputer._valid_mask = (
                    np.asarray(valid, dtype=bool)
                    if valid is not None
                    else np.ones(fit_x.shape[1], dtype=bool)
                )
                obj.imputer.n_features_in_ = int(
                    d.get("knn_n_features_in_", len(obj.fitted_columns_))
                )
            mark_fitted(obj, ready)
        else:
            mark_fitted(obj, "imputers_" in d or bool(obj.fitted_columns_))
        return obj


class DropRows(BaseTransform):
    """Drop rows where % missing > threshold."""

    def __init__(self, threshold=0.5, columns=None):
        super().__init__(threshold=threshold, columns=columns)
        self.threshold = threshold
        self.columns = columns

    @classmethod
    def get_schema(cls) -> dict:
        return {
            "type": "object",
            "properties": {
                "threshold": {"type": "number", "default": 0.5},
                "columns": {"type": "array", "items": {"type": "string"}, "default": None}
            }
        }

    def _fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> None:
        self.fitted_columns_ = self.columns if self.columns is not None else list(X.columns)

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # threshold is the max percentage of missing allowed?
        # "drop rows where % missing > threshold"
        # Wait, % missing across the `fitted_columns_` for that row.
        missing_pct = X[self.fitted_columns_].isnull().mean(axis=1)
        return X[missing_pct <= self.threshold].copy()

    def to_dict(self) -> dict:
        d = self.params.copy()
        d["fitted_columns_"] = self.fitted_columns_
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "BaseTransform":
        obj = cls(threshold=d.get("threshold", 0.5), columns=d.get("columns", None))
        obj.fitted_columns_ = d.get("fitted_columns_", [])
        obj._is_fitted = True
        return obj


class DropColumns(BaseTransform):
    """Drop columns where % missing > threshold."""

    def __init__(self, threshold=0.5):
        super().__init__(threshold=threshold)
        self.threshold = threshold
        self.dropped_cols_ = []

    @classmethod
    def get_schema(cls) -> dict:
        return {
            "type": "object",
            "properties": {
                "threshold": {"type": "number", "default": 0.5}
            }
        }

    def _fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> None:
        self.fitted_columns_ = list(X.columns)
        missing_pct = X.isnull().mean()
        self.dropped_cols_ = missing_pct[missing_pct > self.threshold].index.tolist()

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
        obj = cls(threshold=d.get("threshold", 0.5))
        obj.dropped_cols_ = d.get("dropped_cols_", [])
        obj.fitted_columns_ = d.get("fitted_columns_", [])
        obj._is_fitted = True
        return obj


class FillForward(BaseTransform):
    """Forward fill missing values."""

    def __init__(self, columns=None):
        super().__init__(columns=columns)
        self.columns = columns

    @classmethod
    def get_schema(cls) -> dict:
        return {
            "type": "object",
            "properties": {
                "columns": {"type": "array", "items": {"type": "string"}, "default": None}
            }
        }

    def _fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> None:
        self.fitted_columns_ = self.columns if self.columns is not None else list(X.columns)

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X[self.fitted_columns_] = X[self.fitted_columns_].ffill()
        return X

    def to_dict(self) -> dict:
        d = self.params.copy()
        d["fitted_columns_"] = self.fitted_columns_
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "BaseTransform":
        obj = cls(columns=d.get("columns", None))
        obj.fitted_columns_ = d.get("fitted_columns_", [])
        obj._is_fitted = True
        return obj


class FillBackward(BaseTransform):
    """Backward fill missing values."""

    def __init__(self, columns=None):
        super().__init__(columns=columns)
        self.columns = columns

    @classmethod
    def get_schema(cls) -> dict:
        return {
            "type": "object",
            "properties": {
                "columns": {"type": "array", "items": {"type": "string"}, "default": None}
            }
        }

    def _fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> None:
        self.fitted_columns_ = self.columns if self.columns is not None else list(X.columns)

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X[self.fitted_columns_] = X[self.fitted_columns_].bfill()
        return X

    def to_dict(self) -> dict:
        d = self.params.copy()
        d["fitted_columns_"] = self.fitted_columns_
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "BaseTransform":
        obj = cls(columns=d.get("columns", None))
        obj.fitted_columns_ = d.get("fitted_columns_", [])
        obj._is_fitted = True
        return obj
