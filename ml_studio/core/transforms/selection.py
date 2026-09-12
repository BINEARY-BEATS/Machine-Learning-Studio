"""Feature selection transforms."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.feature_selection import RFE, SelectKBest, VarianceThreshold, f_classif, f_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from ml_studio.core.pipeline import TransformStep
from ml_studio.core.transforms.base import register_transform


@register_transform("select_variance")
class VarianceThresholdStep(TransformStep):
    description = "Remove low-variance features"

    def __init__(self, threshold: float = 0.0) -> None:
        self.threshold = threshold
        self._selector: VarianceThreshold | None = None
        self._columns: list[str] = []

    def fit(self, X: pd.DataFrame, y=None) -> VarianceThresholdStep:
        self._columns = X.select_dtypes(include="number").columns.tolist()
        self._selector = VarianceThreshold(threshold=self.threshold)
        if self._columns:
            self._selector.fit(X[self._columns])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self._selector or not self._columns:
            return X.copy()
        selected = self._selector.transform(X[self._columns])
        mask = self._selector.get_support()
        kept = [c for c, m in zip(self._columns, mask) if m]
        num_df = pd.DataFrame(selected, columns=kept, index=X.index)
        other = X.drop(columns=self._columns)
        return pd.concat([other, num_df], axis=1)

    def to_sklearn(self):
        return self._selector or VarianceThreshold(threshold=self.threshold)

    def get_params(self) -> dict[str, Any]:
        return {"threshold": self.threshold}


@register_transform("select_univariate")
class UnivariateSelectionStep(TransformStep):
    description = "Univariate feature selection (k best)"

    def __init__(self, k: int = 10, task: str = "regression") -> None:
        self.k = k
        self.task = task
        self._selector: SelectKBest | None = None
        self._columns: list[str] = []
        self._selected: list[str] = []

    def fit(self, X: pd.DataFrame, y=None) -> UnivariateSelectionStep:
        self._columns = X.select_dtypes(include="number").columns.tolist()
        score_func = f_regression if self.task == "regression" else f_classif
        self._selector = SelectKBest(score_func=score_func, k=min(self.k, len(self._columns)))
        if self._columns and y is not None:
            self._selector.fit(X[self._columns], y)
            mask = self._selector.get_support()
            self._selected = [c for c, m in zip(self._columns, mask) if m]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        other = X.drop(columns=self._columns, errors="ignore")
        if self._selected:
            return pd.concat([other, X[self._selected]], axis=1)
        return X.copy()

    def get_params(self) -> dict[str, Any]:
        return {"k": self.k, "task": self.task}
