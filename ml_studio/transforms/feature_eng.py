"""Feature Engineering transforms."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, KBinsDiscretizer
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif

from .base import BaseTransform
from ml_studio.core.schema import detect_task_type


class Polynomial(BaseTransform):
    """Generate polynomial and interaction features."""

    def __init__(self, columns=None, degree=2, interaction_only=False, include_bias=False):
        super().__init__(columns=columns, degree=degree, interaction_only=interaction_only, include_bias=include_bias)
        self.columns = columns
        self.degree = degree
        self.interaction_only = interaction_only
        self.include_bias = include_bias
        
    @classmethod
    def get_schema(cls) -> dict:
        return {
            "type": "object",
            "properties": {
                "columns": {"type": "array", "items": {"type": "string"}, "default": None},
                "degree": {"type": "integer", "default": 2},
                "interaction_only": {"type": "boolean", "default": False},
                "include_bias": {"type": "boolean", "default": False}
            }
        }

    def _fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> None:
        self.fitted_columns_ = self.columns if self.columns is not None else [
            c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])
        ]
        if not self.fitted_columns_:
            return

        self.poly = PolynomialFeatures(
            degree=self.degree, 
            interaction_only=self.interaction_only, 
            include_bias=self.include_bias
        )
        self.poly.fit(X[self.fitted_columns_].fillna(0))
        self.feature_names_out_ = self.poly.get_feature_names_out(self.fitted_columns_).tolist()

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted_columns_ or not hasattr(self, 'poly'):
            return X
            
        poly_features = self.poly.transform(X[self.fitted_columns_].fillna(0))
        poly_df = pd.DataFrame(poly_features, columns=self.feature_names_out_, index=X.index)
        
        # Drop original fitted columns to replace with poly features (which includes originals if degree >= 1)
        # Note: PolynomialFeatures degree=2 includes original features by default.
        X = X.drop(columns=self.fitted_columns_)
        return pd.concat([X, poly_df], axis=1)

    def get_output_columns(self, input_columns: list[str]) -> list[str]:
        out = [c for c in input_columns if c not in getattr(self, "fitted_columns_", [])]
        out.extend(getattr(self, "feature_names_out_", []))
        return out

    def to_dict(self) -> dict:
        d = self.params.copy()
        d["fitted_columns_"] = self.fitted_columns_
        d["feature_names_out_"] = getattr(self, "feature_names_out_", [])
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "BaseTransform":
        obj = cls(
            columns=d.get("columns", None),
            degree=d.get("degree", 2),
            interaction_only=d.get("interaction_only", False),
            include_bias=d.get("include_bias", False)
        )
        obj.fitted_columns_ = d.get("fitted_columns_", [])
        obj.feature_names_out_ = d.get("feature_names_out_", [])
        if obj.fitted_columns_:
            obj.poly = PolynomialFeatures(degree=obj.degree, interaction_only=obj.interaction_only, include_bias=obj.include_bias)
            # Reconstruct is hard without data, so we don't fully support transforming from dict 
            # for sklearn estimators that require state.
        obj._is_fitted = True
        return obj


class Ratios(BaseTransform):
    """Compute ratio of pairs of columns (col_a / col_b)."""

    def __init__(self, pairs=None):
        super().__init__(pairs=pairs)
        self.pairs = pairs if pairs else []

    @classmethod
    def get_schema(cls) -> dict:
        return {
            "type": "object",
            "properties": {
                "pairs": {
                    "type": "array", 
                    "items": {
                        "type": "array", 
                        "items": {"type": "string"},
                        "minItems": 2,
                        "maxItems": 2
                    },
                    "default": []
                }
            }
        }

    def _fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> None:
        self.fitted_columns_ = list(set([col for pair in self.pairs for col in pair]))

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        for a, b in self.pairs:
            if a in X.columns and b in X.columns:
                # Add small epsilon to avoid div by zero?
                # or just let it be np.inf
                X[f"{a}_ratio_{b}"] = X[a] / X[b].replace(0, np.nan)
        return X

    def get_output_columns(self, input_columns: list[str]) -> list[str]:
        out = list(input_columns)
        for a, b in self.pairs:
            out.append(f"{a}_ratio_{b}")
        return out

    def to_dict(self) -> dict:
        d = self.params.copy()
        d["fitted_columns_"] = self.fitted_columns_
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "BaseTransform":
        obj = cls(pairs=d.get("pairs", []))
        obj.fitted_columns_ = d.get("fitted_columns_", [])
        obj._is_fitted = True
        return obj


class Differences(BaseTransform):
    """Compute difference of pairs of columns (col_a - col_b)."""

    def __init__(self, pairs=None):
        super().__init__(pairs=pairs)
        self.pairs = pairs if pairs else []

    @classmethod
    def get_schema(cls) -> dict:
        return {
            "type": "object",
            "properties": {
                "pairs": {
                    "type": "array", 
                    "items": {
                        "type": "array", 
                        "items": {"type": "string"},
                        "minItems": 2,
                        "maxItems": 2
                    },
                    "default": []
                }
            }
        }

    def _fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> None:
        self.fitted_columns_ = list(set([col for pair in self.pairs for col in pair]))

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        for a, b in self.pairs:
            if a in X.columns and b in X.columns:
                X[f"{a}_diff_{b}"] = X[a] - X[b]
        return X

    def get_output_columns(self, input_columns: list[str]) -> list[str]:
        out = list(input_columns)
        for a, b in self.pairs:
            out.append(f"{a}_diff_{b}")
        return out

    def to_dict(self) -> dict:
        d = self.params.copy()
        d["fitted_columns_"] = self.fitted_columns_
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "BaseTransform":
        obj = cls(pairs=d.get("pairs", []))
        obj.fitted_columns_ = d.get("fitted_columns_", [])
        obj._is_fitted = True
        return obj


class DateParts(BaseTransform):
    """Extract parts from datetime columns."""

    def __init__(self, columns=None, parts=None):
        if parts is None:
            parts = ["year", "month", "day", "dow", "is_weekend", "hour"]
        super().__init__(columns=columns, parts=parts)
        self.columns = columns
        self.parts = parts

    @classmethod
    def get_schema(cls) -> dict:
        return {
            "type": "object",
            "properties": {
                "columns": {"type": "array", "items": {"type": "string"}, "default": None},
                "parts": {
                    "type": "array", 
                    "items": {"type": "string", "enum": ["year", "month", "day", "dow", "is_weekend", "hour"]},
                    "default": ["year", "month", "day", "dow", "is_weekend", "hour"]
                }
            }
        }

    def _fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> None:
        self.fitted_columns_ = self.columns if self.columns is not None else [
            c for c in X.columns if pd.api.types.is_datetime64_any_dtype(X[c])
        ]

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        for col in self.fitted_columns_:
            s = pd.to_datetime(X[col])
            if "year" in self.parts: X[f"{col}_year"] = s.dt.year
            if "month" in self.parts: X[f"{col}_month"] = s.dt.month
            if "day" in self.parts: X[f"{col}_day"] = s.dt.day
            if "dow" in self.parts: X[f"{col}_dow"] = s.dt.dayofweek
            if "is_weekend" in self.parts: X[f"{col}_is_weekend"] = (s.dt.dayofweek >= 5).astype(int)
            if "hour" in self.parts: X[f"{col}_hour"] = s.dt.hour
            # We don't drop original column just extract parts.
        return X

    def get_output_columns(self, input_columns: list[str]) -> list[str]:
        out = list(input_columns)
        for col in getattr(self, "fitted_columns_", []):
            for part in self.parts:
                out.append(f"{col}_{part}")
        return out

    def to_dict(self) -> dict:
        d = self.params.copy()
        d["fitted_columns_"] = self.fitted_columns_
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "BaseTransform":
        obj = cls(columns=d.get("columns", None), parts=d.get("parts", None))
        obj.fitted_columns_ = d.get("fitted_columns_", [])
        obj._is_fitted = True
        return obj


class CyclicalEncoding(BaseTransform):
    """Sin/Cos encoding for cyclical features."""

    def __init__(self, columns=None, period=None):
        super().__init__(columns=columns, period=period)
        self.columns = columns
        self.period = period
        self.periods_ = {}

    @classmethod
    def get_schema(cls) -> dict:
        return {
            "type": "object",
            "properties": {
                "columns": {"type": "array", "items": {"type": "string"}, "default": None},
                "period": {"type": "number", "default": None}
            }
        }

    def _fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> None:
        self.fitted_columns_ = self.columns if self.columns is not None else [
            c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])
        ]
        for col in self.fitted_columns_:
            self.periods_[col] = self.period if self.period is not None else float(X[col].max())

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        for col in self.fitted_columns_:
            period = self.periods_.get(col, 1.0)
            if period == 0:
                period = 1.0
            X[f"{col}_sin"] = np.sin(2 * np.pi * X[col] / period)
            X[f"{col}_cos"] = np.cos(2 * np.pi * X[col] / period)
            X = X.drop(columns=[col])
        return X

    def get_output_columns(self, input_columns: list[str]) -> list[str]:
        out = [c for c in input_columns if c not in getattr(self, "fitted_columns_", [])]
        for col in getattr(self, "fitted_columns_", []):
            out.append(f"{col}_sin")
            out.append(f"{col}_cos")
        return out

    def to_dict(self) -> dict:
        d = self.params.copy()
        d["fitted_columns_"] = self.fitted_columns_
        d["periods_"] = self.periods_
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "BaseTransform":
        obj = cls(columns=d.get("columns", None), period=d.get("period", None))
        obj.fitted_columns_ = d.get("fitted_columns_", [])
        obj.periods_ = d.get("periods_", {})
        obj._is_fitted = True
        return obj


class Binning(BaseTransform):
    """Bin continuous variables into discrete intervals."""

    def __init__(self, columns=None, strategy="equal_width", n_bins=10):
        super().__init__(columns=columns, strategy=strategy, n_bins=n_bins)
        self.columns = columns
        self.strategy = strategy
        self.n_bins = n_bins

    @classmethod
    def get_schema(cls) -> dict:
        return {
            "type": "object",
            "properties": {
                "columns": {"type": "array", "items": {"type": "string"}, "default": None},
                "strategy": {"type": "string", "enum": ["equal_width", "equal_freq", "kmeans"], "default": "equal_width"},
                "n_bins": {"type": "integer", "default": 10}
            }
        }

    def _fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> None:
        self.fitted_columns_ = self.columns if self.columns is not None else [
            c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])
        ]
        if not self.fitted_columns_:
            return

        # map strategy name to sklearn strategy
        strat_map = {"equal_width": "uniform", "equal_freq": "quantile", "kmeans": "kmeans"}
        self.est = KBinsDiscretizer(n_bins=self.n_bins, encode='ordinal', strategy=strat_map[self.strategy])
        self.est.fit(X[self.fitted_columns_].fillna(0))

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted_columns_ or not hasattr(self, 'est'):
            return X
        X[self.fitted_columns_] = self.est.transform(X[self.fitted_columns_].fillna(0))
        return X

    def to_dict(self) -> dict:
        d = self.params.copy()
        d["fitted_columns_"] = self.fitted_columns_
        if hasattr(self, 'est'):
            d["bin_edges_"] = [e.tolist() for e in self.est.bin_edges_]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "BaseTransform":
        obj = cls(columns=d.get("columns", None), strategy=d.get("strategy", "equal_width"), n_bins=d.get("n_bins", 10))
        obj.fitted_columns_ = d.get("fitted_columns_", [])
        if "bin_edges_" in d:
            strat_map = {"equal_width": "uniform", "equal_freq": "quantile", "kmeans": "kmeans"}
            obj.est = KBinsDiscretizer(n_bins=obj.n_bins, encode='ordinal', strategy=strat_map[obj.strategy])
            obj.est.bin_edges_ = np.array([np.array(e) for e in d["bin_edges_"]], dtype=object)
        obj._is_fitted = True
        return obj


class LogTransform(BaseTransform):
    """Log transform (log1p)."""

    def __init__(self, columns=None, offset=0.0):
        super().__init__(columns=columns, offset=offset)
        self.columns = columns
        self.offset = offset

    @classmethod
    def get_schema(cls) -> dict:
        return {
            "type": "object",
            "properties": {
                "columns": {"type": "array", "items": {"type": "string"}, "default": None},
                "offset": {"type": "number", "default": 0.0}
            }
        }

    def _fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> None:
        self.fitted_columns_ = self.columns if self.columns is not None else [
            c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])
        ]

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        for col in self.fitted_columns_:
            # use np.log1p (log(1+x)) which requires x >= -1
            # we apply offset first
            X[col] = np.log1p(np.maximum(X[col] + self.offset, -1.0))
        return X

    def to_dict(self) -> dict:
        d = self.params.copy()
        d["fitted_columns_"] = self.fitted_columns_
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "BaseTransform":
        obj = cls(columns=d.get("columns", None), offset=d.get("offset", 0.0))
        obj.fitted_columns_ = d.get("fitted_columns_", [])
        obj._is_fitted = True
        return obj


class SqrtTransform(BaseTransform):
    """Square root transform."""

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
        self.fitted_columns_ = self.columns if self.columns is not None else [
            c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])
        ]

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        for col in self.fitted_columns_:
            X[col] = np.sqrt(np.maximum(X[col], 0.0))
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


class InteractionTerms(BaseTransform):
    """Generate interaction terms selected by mutual info."""

    def __init__(self, columns=None, top_k=5, method="mutual_info"):
        super().__init__(columns=columns, top_k=top_k, method=method)
        self.columns = columns
        self.top_k = top_k
        self.method = method
        self.selected_pairs_ = []

    @classmethod
    def get_schema(cls) -> dict:
        return {
            "type": "object",
            "properties": {
                "columns": {"type": "array", "items": {"type": "string"}, "default": None},
                "top_k": {"type": "integer", "default": 5},
                "method": {"type": "string", "enum": ["mutual_info"], "default": "mutual_info"}
            }
        }

    def _fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> None:
        if y is None:
            raise ValueError("InteractionTerms requires a target (y) for mutual info.")
            
        self.fitted_columns_ = self.columns if self.columns is not None else [
            c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])
        ]
        
        if len(self.fitted_columns_) < 2:
            return
            
        task = detect_task_type(y)
        mi_func = mutual_info_classif if task == "CLASSIFICATION" else mutual_info_regression
        
        # We need to evaluate pairs. For large datasets this is very slow. 
        # We'll just generate pairs and compute MI.
        pairs = []
        import itertools
        for c1, c2 in itertools.combinations(self.fitted_columns_, 2):
            pairs.append((c1, c2))
            
        if not pairs:
            return
            
        # Create a temp df with interactions (multiplication)
        interact_df = pd.DataFrame(index=X.index)
        for c1, c2 in pairs:
            interact_df[f"{c1}_x_{c2}"] = X[c1] * X[c2]
            
        interact_df = interact_df.fillna(0)
        mi_scores = mi_func(interact_df, y, random_state=42)
        
        # Sort and get top K
        sorted_indices = np.argsort(mi_scores)[::-1][:self.top_k]
        self.selected_pairs_ = [pairs[i] for i in sorted_indices]

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        for c1, c2 in self.selected_pairs_:
            if c1 in X.columns and c2 in X.columns:
                X[f"{c1}_x_{c2}"] = X[c1] * X[c2]
        return X

    def get_output_columns(self, input_columns: list[str]) -> list[str]:
        out = list(input_columns)
        for c1, c2 in getattr(self, "selected_pairs_", []):
            out.append(f"{c1}_x_{c2}")
        return out

    def to_dict(self) -> dict:
        d = self.params.copy()
        d["fitted_columns_"] = self.fitted_columns_
        d["selected_pairs_"] = self.selected_pairs_
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "BaseTransform":
        obj = cls(columns=d.get("columns", None), top_k=d.get("top_k", 5), method=d.get("method", "mutual_info"))
        obj.fitted_columns_ = d.get("fitted_columns_", [])
        obj.selected_pairs_ = d.get("selected_pairs_", [])
        obj._is_fitted = True
        return obj
