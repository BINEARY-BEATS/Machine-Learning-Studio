"""Feature engineering transforms."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import PolynomialFeatures

from ml_studio.core.pipeline import TransformStep
from ml_studio.core.transforms.base import register_transform


@register_transform("poly_features")
class PolynomialFeaturesStep(TransformStep):
    description = "Generate polynomial interaction features"

    def __init__(self, degree: int = 2) -> None:
        self.degree = degree
        self._poly: PolynomialFeatures | None = None
        self._columns: list[str] = []

    def fit(self, X: pd.DataFrame, y=None) -> PolynomialFeaturesStep:
        self._columns = X.select_dtypes(include="number").columns.tolist()
        self._poly = PolynomialFeatures(degree=self.degree, include_bias=False)
        if self._columns:
            self._poly.fit(X[self._columns])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self._poly or not self._columns:
            return X.copy()
        poly = self._poly.transform(X[self._columns])
        names = self._poly.get_feature_names_out(self._columns)
        poly_df = pd.DataFrame(poly, columns=names, index=X.index)
        non_num = X.drop(columns=self._columns)
        return pd.concat([non_num, poly_df], axis=1)

    def to_sklearn(self):
        return self._poly or PolynomialFeatures(degree=self.degree)

    def get_params(self) -> dict[str, Any]:
        return {"degree": self.degree}


@register_transform("date_components")
class DateComponentsStep(TransformStep):
    description = "Extract year/month/day from datetime columns"

    def __init__(self) -> None:
        self._date_cols: list[str] = []

    def fit(self, X: pd.DataFrame, y=None) -> DateComponentsStep:
        for col in X.columns:
            if pd.api.types.is_datetime64_any_dtype(X[col]):
                self._date_cols.append(col)
            else:
                try:
                    pd.to_datetime(X[col].head(100), errors="raise")
                    self._date_cols.append(col)
                except (ValueError, TypeError):
                    pass
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        for col in self._date_cols:
            if col not in out.columns:
                continue
            dt = pd.to_datetime(out[col], errors="coerce")
            out[f"{col}_year"] = dt.dt.year
            out[f"{col}_month"] = dt.dt.month
            out[f"{col}_day"] = dt.dt.day
            out[f"{col}_dow"] = dt.dt.dayofweek
            out = out.drop(columns=[col])
        return out

    def get_params(self) -> dict[str, Any]:
        return {}


@register_transform("text_tfidf")
class TfidfStep(TransformStep):
    description = "TF-IDF vectorization for text columns"

    def __init__(self, max_features: int = 100) -> None:
        self.max_features = max_features
        self._vectorizers: dict[str, TfidfVectorizer] = {}

    def fit(self, X: pd.DataFrame, y=None) -> TfidfStep:
        for col in X.select_dtypes(include=["object", "string"]).columns:
            avg_len = X[col].dropna().astype(str).str.len().mean()
            if avg_len and avg_len > 20:
                vec = TfidfVectorizer(max_features=self.max_features)
                vec.fit(X[col].fillna("").astype(str))
                self._vectorizers[col] = vec
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        for col, vec in self._vectorizers.items():
            if col not in out.columns:
                continue
            matrix = vec.transform(out[col].fillna("").astype(str)).toarray()
            names = [f"{col}_tfidf_{i}" for i in range(matrix.shape[1])]
            tfidf_df = pd.DataFrame(matrix, columns=names, index=out.index)
            out = pd.concat([out.drop(columns=[col]), tfidf_df], axis=1)
        return out

    def get_params(self) -> dict[str, Any]:
        return {"max_features": self.max_features}
