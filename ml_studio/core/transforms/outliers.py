"""Outlier detection and treatment."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from ml_studio.core.pipeline import TransformStep
from ml_studio.core.transforms.base import register_transform


@register_transform("outlier_iqr")
class IQROutlierStep(TransformStep):
    description = "Cap outliers using IQR method"

    def __init__(self, threshold: float = 1.5, method: str = "cap") -> None:
        self.threshold = threshold
        self.method = method
        self._bounds: dict[str, tuple[float, float]] = {}

    def fit(self, X: pd.DataFrame, y=None) -> IQROutlierStep:
        for col in X.select_dtypes(include="number").columns:
            q1, q3 = X[col].quantile(0.25), X[col].quantile(0.75)
            iqr = q3 - q1
            self._bounds[col] = (q1 - self.threshold * iqr, q3 + self.threshold * iqr)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        if self.method == "cap":
            for col, (lo, hi) in self._bounds.items():
                if col in out.columns:
                    out[col] = out[col].clip(lo, hi)
        elif self.method == "remove":
            mask = pd.Series(True, index=out.index)
            for col, (lo, hi) in self._bounds.items():
                if col in out.columns:
                    mask &= out[col].between(lo, hi)
            out = out[mask]
        return out

    def get_params(self) -> dict[str, Any]:
        return {"threshold": self.threshold, "method": self.method}


@register_transform("outlier_isolation_forest")
class IsolationForestOutlierStep(TransformStep):
    description = "Remove outliers detected by Isolation Forest"

    def __init__(self, contamination: float = 0.05) -> None:
        self.contamination = contamination
        self._model: IsolationForest | None = None
        self._columns: list[str] = []

    def fit(self, X: pd.DataFrame, y=None) -> IsolationForestOutlierStep:
        self._columns = X.select_dtypes(include="number").columns.tolist()
        self._model = IsolationForest(contamination=self.contamination, random_state=42)
        if self._columns:
            self._model.fit(X[self._columns])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self._model or not self._columns:
            return X.copy()
        preds = self._model.predict(X[self._columns])
        return X[preds == 1].copy()

    def to_sklearn(self):
        return self._model or IsolationForest(contamination=self.contamination)

    def get_params(self) -> dict[str, Any]:
        return {"contamination": self.contamination}
