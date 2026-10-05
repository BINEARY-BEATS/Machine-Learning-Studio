"""Automatic train-only feature encoding for mixed-type columns."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ml_studio.transforms.base import BaseTransform

_MAX_ONEHOT = 15


class AutoEncode(BaseTransform):
    """Encode mixed columns to numeric. Fit on TRAIN only; unknown → safe defaults."""

    def __init__(self, max_onehot: int = _MAX_ONEHOT, **kwargs: Any):
        super().__init__(max_onehot=max_onehot, **kwargs)
        self.max_onehot = max_onehot
        self.col_kinds_: dict[str, str] = {}
        self.medians_: dict[str, float] = {}
        self.onehot_cats_: dict[str, list[str]] = {}
        self.freq_maps_: dict[str, dict[str, float]] = {}
        self.output_columns_: list[str] = []

    @classmethod
    def get_schema(cls) -> dict:
        return {
            "type": "object",
            "properties": {
                "max_onehot": {"type": "integer", "default": _MAX_ONEHOT},
            },
        }

    def _fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> None:
        self.fitted_columns_ = list(X.columns)
        self.col_kinds_.clear()
        self.medians_.clear()
        self.onehot_cats_.clear()
        self.freq_maps_.clear()
        out_cols: list[str] = []
        for col in X.columns:
            kind = self._detect_kind(X[col])
            self.col_kinds_[col] = kind
            out_cols.extend(self._fit_column(col, X[col], kind))
        self.output_columns_ = out_cols

    def _detect_kind(self, s: pd.Series) -> str:
        if pd.api.types.is_bool_dtype(s):
            return "bool"
        if pd.api.types.is_datetime64_any_dtype(s):
            return "datetime"
        if pd.api.types.is_numeric_dtype(s):
            return "numeric"
        nunique = int(s.nunique(dropna=True))
        return "onehot" if nunique <= self.max_onehot else "freq"

    def _fit_column(self, col: str, s: pd.Series, kind: str) -> list[str]:
        if kind == "numeric":
            self.medians_[col] = float(s.median()) if s.notna().any() else 0.0
            return [col]
        if kind == "bool":
            return [col]
        if kind == "datetime":
            return [f"{col}_year", f"{col}_month", f"{col}_day", f"{col}_dow"]
        if kind == "onehot":
            cats = sorted(s.dropna().astype(str).unique().tolist())
            self.onehot_cats_[col] = cats
            return [f"{col}__{c}" for c in cats]
        # freq
        vc = s.dropna().astype(str).value_counts(normalize=True)
        self.freq_maps_[col] = {str(k): float(v) for k, v in vc.items()}
        return [col]

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        parts: list[pd.Series] = []
        for col in self.fitted_columns_:
            kind = self.col_kinds_[col]
            series = X[col] if col in X.columns else pd.Series(np.nan, index=X.index)
            parts.extend(self._transform_column(col, series, kind, X.index))
        out = pd.concat(parts, axis=1)
        out.index = X.index
        return self._impute(out)

    def _transform_column(
        self, col: str, s: pd.Series, kind: str, index: pd.Index
    ) -> list[pd.Series]:
        if kind == "numeric":
            return [pd.to_numeric(s, errors="coerce").rename(col)]
        if kind == "bool":
            return [s.map(self._to_bool_int).astype(float).rename(col)]
        if kind == "datetime":
            return self._datetime_parts(col, s)
        if kind == "onehot":
            return self._onehot_parts(col, s, index)
        mapped = s.astype(str).map(lambda v, m=self.freq_maps_[col]: m.get(v, 0.0))
        return [mapped.astype(float).rename(col)]

    @staticmethod
    def _to_bool_int(v: Any) -> float:
        if pd.isna(v):
            return np.nan
        if isinstance(v, (bool, np.bool_)):
            return float(v)
        text = str(v).strip().lower()
        if text in ("1", "true", "yes", "y"):
            return 1.0
        if text in ("0", "false", "no", "n"):
            return 0.0
        return np.nan

    def _datetime_parts(self, col: str, s: pd.Series) -> list[pd.Series]:
        dt = pd.to_datetime(s, errors="coerce")
        return [
            dt.dt.year.astype(float).rename(f"{col}_year"),
            dt.dt.month.astype(float).rename(f"{col}_month"),
            dt.dt.day.astype(float).rename(f"{col}_day"),
            dt.dt.dayofweek.astype(float).rename(f"{col}_dow"),
        ]

    def _onehot_parts(self, col: str, s: pd.Series, index: pd.Index) -> list[pd.Series]:
        raw = s.astype(str)
        parts = []
        for cat in self.onehot_cats_[col]:
            name = f"{col}__{cat}"
            # Unknown / NaN → 0 for all one-hots
            mask = s.notna() & (raw == cat)
            parts.append(mask.astype(float).rename(name))
        return parts

    def _impute(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        for col in out.columns:
            if out[col].isna().any():
                fill = self.medians_.get(col, 0.0)
                # onehot/freq/bool columns use 0; numeric uses median when known
                if col in self.medians_:
                    fill = self.medians_[col]
                else:
                    fill = 0.0
                out[col] = out[col].fillna(fill)
        return out

    def get_output_columns(self, input_columns: list[str]) -> list[str]:
        return list(self.output_columns_) if self.output_columns_ else input_columns

    def to_dict(self) -> dict:
        return {
            "params": dict(self.params),
            "fitted_columns_": self.fitted_columns_,
            "col_kinds_": self.col_kinds_,
            "medians_": self.medians_,
            "onehot_cats_": self.onehot_cats_,
            "freq_maps_": self.freq_maps_,
            "output_columns_": self.output_columns_,
            "_is_fitted": self._is_fitted,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AutoEncode":
        params = d.get("params", d)
        obj = cls(max_onehot=params.get("max_onehot", _MAX_ONEHOT))
        obj.fitted_columns_ = d.get("fitted_columns_", [])
        obj.col_kinds_ = d.get("col_kinds_", {})
        obj.medians_ = {k: float(v) for k, v in d.get("medians_", {}).items()}
        obj.onehot_cats_ = d.get("onehot_cats_", {})
        obj.freq_maps_ = d.get("freq_maps_", {})
        obj.output_columns_ = d.get("output_columns_", [])
        # Only mark fitted when transform state is present
        obj._is_fitted = bool(obj.col_kinds_) and bool(d.get("_is_fitted", True))
        return obj
