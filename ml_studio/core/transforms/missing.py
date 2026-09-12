"""Missing value imputation transforms."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.impute import KNNImputer, SimpleImputer

from ml_studio.core.pipeline import TransformStep
from ml_studio.core.transforms.base import register_transform


@register_transform("missing_mean")
class MissingMeanImputer(TransformStep):
    description = "Impute numeric missing values with column mean"

    def __init__(self, strategy: str = "mean") -> None:
        self.strategy = strategy
        self._imputer: SimpleImputer | None = None
        self._columns: list[str] = []

    def fit(self, X: pd.DataFrame, y=None) -> MissingMeanImputer:
        self._columns = X.select_dtypes(include="number").columns.tolist()
        self._imputer = SimpleImputer(strategy=self.strategy)
        if self._columns:
            self._imputer.fit(X[self._columns])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        if self._imputer and self._columns:
            out[self._columns] = self._imputer.transform(out[self._columns])
        for col in out.select_dtypes(include=["object", "category"]).columns:
            if out[col].isnull().any():
                mode = out[col].mode()
                out[col] = out[col].fillna(mode.iloc[0] if len(mode) else "Unknown")
        return out

    def to_sklearn(self):
        return self._imputer or SimpleImputer(strategy=self.strategy)

    def get_params(self) -> dict[str, Any]:
        return {"strategy": self.strategy}


@register_transform("missing_median")
class MissingMedianImputer(MissingMeanImputer):
    description = "Impute numeric missing values with column median"

    def __init__(self) -> None:
        super().__init__(strategy="median")


@register_transform("missing_knn")
class MissingKNNImputer(TransformStep):
    description = "KNN imputation for numeric columns"

    def __init__(self, n_neighbors: int = 5) -> None:
        self.n_neighbors = n_neighbors
        self._imputer: KNNImputer | None = None
        self._columns: list[str] = []

    def fit(self, X: pd.DataFrame, y=None) -> MissingKNNImputer:
        self._columns = X.select_dtypes(include="number").columns.tolist()
        self._imputer = KNNImputer(n_neighbors=self.n_neighbors)
        if self._columns:
            self._imputer.fit(X[self._columns])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        if self._imputer and self._columns:
            out[self._columns] = self._imputer.transform(out[self._columns])
        return out

    def to_sklearn(self):
        return self._imputer or KNNImputer(n_neighbors=self.n_neighbors)

    def get_params(self) -> dict[str, Any]:
        return {"n_neighbors": self.n_neighbors}


@register_transform("missing_ffill")
class ForwardFillImputer(TransformStep):
    description = "Forward-fill missing values"

    def fit(self, X: pd.DataFrame, y=None) -> ForwardFillImputer:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.ffill().bfill()

    def get_params(self) -> dict[str, Any]:
        return {}
