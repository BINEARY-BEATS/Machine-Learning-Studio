"""Feature scaling transforms."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.preprocessing import (
    MinMaxScaler,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)

from ml_studio.core.pipeline import TransformStep
from ml_studio.core.transforms.base import register_transform


def _numeric_cols(X: pd.DataFrame) -> list[str]:
    return X.select_dtypes(include="number").columns.tolist()


class _ScalerStep(TransformStep):
    scaler_cls = StandardScaler

    def __init__(self) -> None:
        self._scaler = None
        self._columns: list[str] = []

    def fit(self, X: pd.DataFrame, y=None):
        self._columns = _numeric_cols(X)
        self._scaler = self.scaler_cls()
        if self._columns:
            self._scaler.fit(X[self._columns])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        if self._scaler and self._columns:
            out[self._columns] = self._scaler.transform(out[self._columns])
        return out

    def to_sklearn(self):
        return self._scaler or self.scaler_cls()

    def get_params(self) -> dict[str, Any]:
        return {}


@register_transform("scale_standard")
class StandardScalerStep(_ScalerStep):
    description = "StandardScaler (z-score normalization)"
    scaler_cls = StandardScaler


@register_transform("scale_minmax")
class MinMaxScalerStep(_ScalerStep):
    description = "MinMaxScaler (0-1 range)"
    scaler_cls = MinMaxScaler


@register_transform("scale_robust")
class RobustScalerStep(_ScalerStep):
    description = "RobustScaler (median/IQR based)"
    scaler_cls = RobustScaler


@register_transform("scale_power")
class PowerTransformerStep(_ScalerStep):
    description = "PowerTransformer (Yeo-Johnson)"
    scaler_cls = PowerTransformer


@register_transform("scale_quantile")
class QuantileTransformerStep(_ScalerStep):
    description = "QuantileTransformer (uniform/normal output)"
    scaler_cls = QuantileTransformer
