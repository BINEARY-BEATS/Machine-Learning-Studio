"""Categorical encoding transforms."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from ml_studio.core.pipeline import TransformStep
from ml_studio.core.transforms.base import register_transform


@register_transform("encode_onehot")
class OneHotEncoderStep(TransformStep):
    description = "One-hot encode categorical columns"

    def __init__(self, max_categories: int = 25) -> None:
        self.max_categories = max_categories
        self._encoder: OneHotEncoder | None = None
        self._cat_cols: list[str] = []

    def fit(self, X: pd.DataFrame, y=None) -> OneHotEncoderStep:
        self._cat_cols = [
            c for c in X.select_dtypes(include=["object", "category"]).columns
            if X[c].nunique() <= self.max_categories
        ]
        if self._cat_cols:
            self._encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            self._encoder.fit(X[self._cat_cols].astype(str))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self._encoder or not self._cat_cols:
            return X.copy()
        encoded = self._encoder.transform(X[self._cat_cols].astype(str))
        names = self._encoder.get_feature_names_out(self._cat_cols)
        enc_df = pd.DataFrame(encoded, columns=names, index=X.index)
        return pd.concat([X.drop(columns=self._cat_cols), enc_df], axis=1)

    def to_sklearn(self):
        return self._encoder or OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    def get_params(self) -> dict[str, Any]:
        return {"max_categories": self.max_categories}


@register_transform("encode_ordinal")
class OrdinalEncoderStep(TransformStep):
    description = "Ordinal encode categorical columns"

    def __init__(self) -> None:
        self._encoder: OrdinalEncoder | None = None
        self._cat_cols: list[str] = []

    def fit(self, X: pd.DataFrame, y=None) -> OrdinalEncoderStep:
        self._cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
        if self._cat_cols:
            self._encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            self._encoder.fit(X[self._cat_cols].astype(str))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        if self._encoder and self._cat_cols:
            out[self._cat_cols] = self._encoder.transform(out[self._cat_cols].astype(str))
        return out

    def to_sklearn(self):
        return self._encoder or OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)


@register_transform("encode_frequency")
class FrequencyEncoderStep(TransformStep):
    description = "Replace categories with frequency counts"

    def __init__(self) -> None:
        self._freq_maps: dict[str, dict] = {}

    def fit(self, X: pd.DataFrame, y=None) -> FrequencyEncoderStep:
        for col in X.select_dtypes(include=["object", "category"]).columns:
            self._freq_maps[col] = X[col].value_counts(normalize=True).to_dict()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        for col, freq in self._freq_maps.items():
            if col in out.columns:
                out[col] = out[col].map(freq).fillna(0)
        return out

    def get_params(self) -> dict[str, Any]:
        return {}
