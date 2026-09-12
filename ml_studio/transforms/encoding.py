"""Categorical encoding transforms."""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
import hashlib

from .base import BaseTransform
from ml_studio.core.schema import detect_task_type


class OneHot(BaseTransform):
    """One-hot encoding for categorical features."""

    def __init__(self, columns=None, max_categories=50, drop_first=False, handle_unknown="ignore"):
        super().__init__(columns=columns, max_categories=max_categories, drop_first=drop_first, handle_unknown=handle_unknown)
        self.columns = columns
        self.max_categories = max_categories
        self.drop_first = drop_first
        self.handle_unknown = handle_unknown
        self.categories_ = {}  # {col: [cat1, cat2, ...]}

    @classmethod
    def get_schema(cls) -> dict:
        return {
            "type": "object",
            "properties": {
                "columns": {"type": "array", "items": {"type": "string"}, "default": None},
                "max_categories": {"type": "integer", "default": 50},
                "drop_first": {"type": "boolean", "default": False},
                "handle_unknown": {"type": "string", "enum": ["ignore", "error"], "default": "ignore"}
            }
        }

    def _fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> None:
        self.fitted_columns_ = self.columns if self.columns is not None else [
            c for c in X.columns if X[c].dtype == object or pd.api.types.is_categorical_dtype(X[c]) or pd.api.types.is_string_dtype(X[c]) or pd.api.types.is_string_dtype(X[c])
        ]
        
        for col in self.fitted_columns_:
            val_counts = X[col].value_counts(dropna=False)
            cats = val_counts.nlargest(self.max_categories).index.tolist()
            if self.drop_first and len(cats) > 0:
                cats = cats[1:]
            self.categories_[col] = cats

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        for col in self.fitted_columns_:
            cats = self.categories_.get(col, [])
            for cat in cats:
                # Handle NA as category if it was in top cats
                if pd.isna(cat):
                    new_col_name = f"{col}_nan"
                    X[new_col_name] = X[col].isna().astype(int)
                else:
                    new_col_name = f"{col}_{cat}"
                    X[new_col_name] = (X[col] == cat).astype(int)
            X = X.drop(columns=[col])
        return X

    def get_output_columns(self, input_columns: list[str]) -> list[str]:
        out = [c for c in input_columns if c not in getattr(self, "fitted_columns_", [])]
        for col in getattr(self, "fitted_columns_", []):
            cats = self.categories_.get(col, [])
            for cat in cats:
                if pd.isna(cat):
                    out.append(f"{col}_nan")
                else:
                    out.append(f"{col}_{cat}")
        return out

    def to_dict(self) -> dict:
        d = self.params.copy()
        d["categories_"] = self.categories_
        d["fitted_columns_"] = self.fitted_columns_
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "BaseTransform":
        obj = cls(
            columns=d.get("columns", None),
            max_categories=d.get("max_categories", 50),
            drop_first=d.get("drop_first", False),
            handle_unknown=d.get("handle_unknown", "ignore")
        )
        obj.categories_ = d.get("categories_", {})
        obj.fitted_columns_ = d.get("fitted_columns_", [])
        obj._is_fitted = True
        return obj


class Ordinal(BaseTransform):
    """Ordinal encoding."""
    
    def __init__(self, columns=None, categories=None):
        super().__init__(columns=columns, categories=categories)
        self.columns = columns
        self.categories = categories
        self.mapping_ = {}
        
    @classmethod
    def get_schema(cls) -> dict:
        return {
            "type": "object",
            "properties": {
                "columns": {"type": "array", "items": {"type": "string"}, "default": None},
                "categories": {"type": "object", "default": None}  # {col: [ordered_cats]}
            }
        }

    def _fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> None:
        self.fitted_columns_ = self.columns if self.columns is not None else [
            c for c in X.columns if X[c].dtype == object or pd.api.types.is_categorical_dtype(X[c]) or pd.api.types.is_string_dtype(X[c])
        ]
        
        for col in self.fitted_columns_:
            if self.categories and col in self.categories:
                cats = self.categories[col]
            else:
                cats = sorted(X[col].dropna().unique().tolist())
            self.mapping_[col] = {cat: i for i, cat in enumerate(cats)}

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        for col in self.fitted_columns_:
            mapping = self.mapping_.get(col, {})
            X[col] = X[col].map(mapping).fillna(-1)
        return X

    def to_dict(self) -> dict:
        d = self.params.copy()
        d["mapping_"] = self.mapping_
        d["fitted_columns_"] = self.fitted_columns_
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "BaseTransform":
        obj = cls(columns=d.get("columns", None), categories=d.get("categories", None))
        obj.mapping_ = d.get("mapping_", {})
        obj.fitted_columns_ = d.get("fitted_columns_", [])
        obj._is_fitted = True
        return obj


class Frequency(BaseTransform):
    """Frequency encoding."""
    def __init__(self, columns=None):
        super().__init__(columns=columns)
        self.columns = columns
        self.frequencies_ = {}

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
            c for c in X.columns if X[c].dtype == object or pd.api.types.is_categorical_dtype(X[c]) or pd.api.types.is_string_dtype(X[c])
        ]
        for col in self.fitted_columns_:
            self.frequencies_[col] = X[col].value_counts(normalize=True, dropna=False).to_dict()

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        for col in self.fitted_columns_:
            freqs = self.frequencies_.get(col, {})
            X[col] = X[col].map(freqs).fillna(0.0)
        return X

    def to_dict(self) -> dict:
        d = self.params.copy()
        d["frequencies_"] = self.frequencies_
        d["fitted_columns_"] = self.fitted_columns_
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "BaseTransform":
        obj = cls(columns=d.get("columns", None))
        obj.frequencies_ = d.get("frequencies_", {})
        obj.fitted_columns_ = d.get("fitted_columns_", [])
        obj._is_fitted = True
        return obj


class Hashing(BaseTransform):
    """Hashing encoder (Hash Trick)."""

    def __init__(self, columns=None, n_features=64):
        super().__init__(columns=columns, n_features=n_features)
        self.columns = columns
        self.n_features = n_features

    @classmethod
    def get_schema(cls) -> dict:
        return {
            "type": "object",
            "properties": {
                "columns": {"type": "array", "items": {"type": "string"}, "default": None},
                "n_features": {"type": "integer", "default": 64}
            }
        }

    def _fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> None:
        self.fitted_columns_ = self.columns if self.columns is not None else [
            c for c in X.columns if X[c].dtype == object or pd.api.types.is_categorical_dtype(X[c]) or pd.api.types.is_string_dtype(X[c])
        ]

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        for col in self.fitted_columns_:
            hashed = X[col].astype(str).apply(
                lambda x: int(hashlib.md5(x.encode('utf-8')).hexdigest(), 16) % self.n_features
            )
            # Create one-hot like columns for the hash bins
            for i in range(self.n_features):
                X[f"{col}_hash_{i}"] = (hashed == i).astype(int)
            X = X.drop(columns=[col])
        return X

    def get_output_columns(self, input_columns: list[str]) -> list[str]:
        out = [c for c in input_columns if c not in getattr(self, "fitted_columns_", [])]
        for col in getattr(self, "fitted_columns_", []):
            for i in range(self.n_features):
                out.append(f"{col}_hash_{i}")
        return out

    def to_dict(self) -> dict:
        d = self.params.copy()
        d["fitted_columns_"] = self.fitted_columns_
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "BaseTransform":
        obj = cls(columns=d.get("columns", None), n_features=d.get("n_features", 64))
        obj.fitted_columns_ = d.get("fitted_columns_", [])
        obj._is_fitted = True
        return obj


# K-Fold Encoders Base Class
class BaseKFoldEncoder(BaseTransform):
    def __init__(self, columns=None, cv=5, smoothing=10, min_samples_leaf=1, noise_level=0.0):
        super().__init__(columns=columns, cv=cv, smoothing=smoothing, min_samples_leaf=min_samples_leaf, noise_level=noise_level)
        self.columns = columns
        self.cv = cv
        self.smoothing = smoothing
        self.min_samples_leaf = min_samples_leaf
        self.noise_level = noise_level
        
        self.category_means_ = {}
        self.global_mean_ = 0.0
        # fold_statistics_ is used internally to verify leakage tests if needed, but primarily we store it 
        # to ensure we follow the instruction. Actually, during transform we only use category_means_.
        self.fold_statistics_ = {}

    def _fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> None:
        if y is None:
            raise ValueError(f"{self.__class__.__name__} requires a target (y) for fitting.")
            
        self.fitted_columns_ = self.columns if self.columns is not None else [
            c for c in X.columns if X[c].dtype == object or pd.api.types.is_categorical_dtype(X[c]) or pd.api.types.is_string_dtype(X[c])
        ]
        
        self.global_mean_ = float(y.mean())
        
        task = detect_task_type(y)
        if task == "CLASSIFICATION":
            kf = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=42)
            splits = list(kf.split(X, y))
        else:
            kf = KFold(n_splits=self.cv, shuffle=True, random_state=42)
            splits = list(kf.split(X))

        # We compute out-of-fold means for the training set so if fit_transform is called, it's leak-free
        # However, for pure transform (on test data), we just use category_means_.
        self.category_means_ = {}
        self.fold_statistics_ = {col: [] for col in self.fitted_columns_}
        
        # Calculate global category means for transform()
        for col in self.fitted_columns_:
            self.category_means_[col] = self._compute_smoothed_means(X[col], y, self.global_mean_)
            
            # compute per-fold stats for leak-free fit_transform (we will store them here for fit_transform to use)
            self.fold_statistics_[col] = []
            for train_idx, val_idx in splits:
                X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
                fold_means = self._compute_smoothed_means(X_tr[col], y_tr, self.global_mean_)
                self.fold_statistics_[col].append((val_idx, fold_means))
                
    def _compute_smoothed_means(self, series: pd.Series, y: pd.Series, global_mean: float) -> dict:
        stats = y.groupby(series).agg(['count', 'mean'])
        
        # Smoothing: weight = 1 / (1 + exp(-(count - min_samples_leaf) / smoothing))
        # if smoothing is 0, weight is 1 if count >= min_samples else 0.
        if self.smoothing > 0:
            smoothing_factor = (stats['count'] - self.min_samples_leaf) / self.smoothing
            # prevent overflow
            smoothing_factor = np.clip(smoothing_factor, -20, 20)
            weight = 1.0 / (1.0 + np.exp(-smoothing_factor))
        else:
            weight = (stats['count'] >= self.min_samples_leaf).astype(float)
            
        smoothed = weight * stats['mean'] + (1.0 - weight) * global_mean
        return smoothed.to_dict()

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        for col in self.fitted_columns_:
            means = self.category_means_.get(col, {})
            encoded = X[col].map(means).fillna(self.global_mean_)
            if self.noise_level > 0:
                np.random.seed(42)
                encoded += np.random.normal(0, self.noise_level, size=len(encoded))
            X[col] = encoded
        return X

    def fit_transform(self, X: pd.DataFrame, y: pd.Series | None = None) -> pd.DataFrame:
        """Override fit_transform to apply out-of-fold encoding."""
        self.fit(X, y)
        X_out = X.copy()
        
        for col in self.fitted_columns_:
            encoded = pd.Series(index=X.index, dtype=float)
            for val_idx, fold_means in self.fold_statistics_[col]:
                mapped = X.iloc[val_idx][col].map(fold_means).fillna(self.global_mean_)
                encoded.iloc[val_idx] = mapped
            
            # Any rows not in folds? (Shouldn't happen with kfold, but just in case)
            missing = encoded.isna()
            if missing.any():
                encoded[missing] = X.loc[missing, col].map(self.category_means_[col]).fillna(self.global_mean_)
                
            if self.noise_level > 0:
                np.random.seed(42)
                encoded += np.random.normal(0, self.noise_level, size=len(encoded))
                
            X_out[col] = encoded
        return X_out
        
    def to_dict(self) -> dict:
        d = self.params.copy()
        d["category_means_"] = self.category_means_
        d["global_mean_"] = self.global_mean_
        d["fitted_columns_"] = self.fitted_columns_
        return d
        
    @classmethod
    def from_dict(cls, d: dict) -> "BaseTransform":
        obj = cls(
            columns=d.get("columns", None),
            cv=d.get("cv", 5),
            smoothing=d.get("smoothing", 10),
            min_samples_leaf=d.get("min_samples_leaf", 1),
            noise_level=d.get("noise_level", 0.0)
        )
        obj.category_means_ = d.get("category_means_", {})
        obj.global_mean_ = d.get("global_mean_", 0.0)
        obj.fitted_columns_ = d.get("fitted_columns_", [])
        obj._is_fitted = True
        return obj


class Target(BaseKFoldEncoder):
    """Target encoding (smoothed K-fold mean)."""

    @classmethod
    def get_schema(cls) -> dict:
        return {
            "type": "object",
            "properties": {
                "columns": {"type": "array", "items": {"type": "string"}, "default": None},
                "cv": {"type": "integer", "default": 5},
                "smoothing": {"type": "number", "default": 10},
                "min_samples_leaf": {"type": "integer", "default": 1},
                "noise_level": {"type": "number", "default": 0.0}
            }
        }


class WOE(BaseKFoldEncoder):
    """Weight of Evidence Encoding (Classification Only)."""

    @classmethod
    def get_schema(cls) -> dict:
        return {
            "type": "object",
            "properties": {
                "columns": {"type": "array", "items": {"type": "string"}, "default": None},
                "cv": {"type": "integer", "default": 5},
                "smoothing": {"type": "number", "default": 10},
                "min_samples_leaf": {"type": "integer", "default": 1},
                "noise_level": {"type": "number", "default": 0.0}
            }
        }

    def _compute_smoothed_means(self, series: pd.Series, y: pd.Series, global_mean: float) -> dict:
        # WOE: ln(P(1|cat) / P(0|cat)) - ln(P(1) / P(0))
        # We must add smoothing to avoid log(0)
        
        pos_total = max(y.sum(), 0.5)
        neg_total = max(len(y) - pos_total, 0.5)
        
        counts = y.groupby(series).agg(['count', 'sum'])
        counts['pos'] = counts['sum']
        counts['neg'] = counts['count'] - counts['pos']
        
        # Apply smoothing
        smooth = self.smoothing
        pos_prob = (counts['pos'] + smooth) / (pos_total + smooth * 2)
        neg_prob = (counts['neg'] + smooth) / (neg_total + smooth * 2)
        
        woe = np.log(pos_prob / neg_prob)
        return woe.to_dict()


class LeaveOneOut(BaseTransform):
    """Leave-one-out target encoding."""

    def __init__(self, columns=None, noise_level=0.0):
        super().__init__(columns=columns, noise_level=noise_level)
        self.columns = columns
        self.noise_level = noise_level
        self.category_means_ = {}
        self.global_mean_ = 0.0
        
    @classmethod
    def get_schema(cls) -> dict:
        return {
            "type": "object",
            "properties": {
                "columns": {"type": "array", "items": {"type": "string"}, "default": None},
                "noise_level": {"type": "number", "default": 0.0}
            }
        }

    def _fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> None:
        if y is None:
            raise ValueError("LeaveOneOut requires a target (y) for fitting.")
        self.fitted_columns_ = self.columns if self.columns is not None else [
            c for c in X.columns if X[c].dtype == object or pd.api.types.is_categorical_dtype(X[c]) or pd.api.types.is_string_dtype(X[c])
        ]
        self.global_mean_ = float(y.mean())
        
        for col in self.fitted_columns_:
            stats = y.groupby(X[col]).agg(['sum', 'count'])
            self.category_means_[col] = {
                'sum': stats['sum'].to_dict(),
                'count': stats['count'].to_dict()
            }

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        for col in self.fitted_columns_:
            stats = self.category_means_.get(col, {})
            sum_dict = stats.get('sum', {})
            count_dict = stats.get('count', {})
            
            encoded = X[col].map(lambda x: sum_dict.get(x, self.global_mean_ * count_dict.get(x, 1)) / max(count_dict.get(x, 1), 1))
            encoded = encoded.fillna(self.global_mean_)
            
            if self.noise_level > 0:
                np.random.seed(42)
                encoded += np.random.normal(0, self.noise_level, size=len(encoded))
            X[col] = encoded
        return X
        
    def fit_transform(self, X: pd.DataFrame, y: pd.Series | None = None) -> pd.DataFrame:
        self.fit(X, y)
        X_out = X.copy()
        for col in self.fitted_columns_:
            stats = self.category_means_[col]
            sum_dict = stats['sum']
            count_dict = stats['count']
            
            # Leave one out: (sum - y) / (count - 1)
            cat_sum = X[col].map(sum_dict)
            cat_count = X[col].map(count_dict)
            
            # For count == 1, fallback to global mean
            encoded = (cat_sum - y) / (cat_count - 1).replace(0, np.nan)
            encoded = encoded.fillna(self.global_mean_)
            
            if self.noise_level > 0:
                np.random.seed(42)
                encoded += np.random.normal(0, self.noise_level, size=len(encoded))
            X_out[col] = encoded
            
        return X_out

    def to_dict(self) -> dict:
        d = self.params.copy()
        d["category_means_"] = self.category_means_
        d["global_mean_"] = self.global_mean_
        d["fitted_columns_"] = self.fitted_columns_
        return d
        
    @classmethod
    def from_dict(cls, d: dict) -> "BaseTransform":
        obj = cls(
            columns=d.get("columns", None),
            noise_level=d.get("noise_level", 0.0)
        )
        obj.category_means_ = d.get("category_means_", {})
        obj.global_mean_ = d.get("global_mean_", 0.0)
        obj.fitted_columns_ = d.get("fitted_columns_", [])
        obj._is_fitted = True
        return obj
