"""Class imbalance handling."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ml_studio.core.pipeline import TransformStep
from ml_studio.core.transforms.base import register_transform


@register_transform("balance_smote")
class SMOTEStep(TransformStep):
    description = "SMOTE oversampling for classification"

    def __init__(self, k_neighbors: int = 5) -> None:
        self.k_neighbors = k_neighbors
        self._smote = None

    def fit(self, X: pd.DataFrame, y=None) -> SMOTEStep:
        try:
            from imblearn.over_sampling import SMOTE

            self._smote = SMOTE(k_neighbors=self.k_neighbors, random_state=42)
        except ImportError as e:
            raise ImportError(
                "imbalanced-learn is required for SMOTE. pip install imbalanced-learn"
            ) from e
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.copy()

    def fit_resample(self, X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
        if self._smote is None:
            raise RuntimeError("SMOTE not fitted")
        X_res, y_res = self._smote.fit_resample(X, y)
        return pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res, name=y.name)

    def get_params(self) -> dict[str, Any]:
        return {"k_neighbors": self.k_neighbors}


@register_transform("balance_class_weight")
class ClassWeightStep(TransformStep):
    description = "Marker step — class weights applied at model level"

    def fit(self, X: pd.DataFrame, y=None) -> ClassWeightStep:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.copy()

    def get_params(self) -> dict[str, Any]:
        return {}
