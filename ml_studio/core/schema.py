"""Schema inference and column metadata."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd


class ColumnRole(str, Enum):
    FEATURE = "feature"
    TARGET = "target"
    ID = "id"
    GROUP = "group"
    TIME_INDEX = "time_index"
    WEIGHT = "weight"
    DROP = "drop"

class ColumnKind(str, Enum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"
    BOOLEAN = "boolean"
    TEXT = "text"
    UNKNOWN = "unknown"


@dataclass
class ColumnSchema:
    name: str
    dtype: str
    kind: ColumnKind
    role: ColumnRole = ColumnRole.FEATURE
    nullable: bool = True
    unique_count: int = 0
    sample_values: list[Any] = field(default_factory=list)


@dataclass
class DatasetSchema:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    columns: list[ColumnSchema] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "columns": [
                {
                    "name": c.name,
                    "dtype": c.dtype,
                    "kind": c.kind.value,
                    "role": c.role.value,
                    "nullable": c.nullable,
                    "unique_count": c.unique_count,
                }
                for c in self.columns
            ],
        }


def infer_column_kind(series: pd.Series) -> ColumnKind:
    if pd.api.types.is_bool_dtype(series):
        return ColumnKind.BOOLEAN
    if pd.api.types.is_datetime64_any_dtype(series):
        return ColumnKind.DATETIME
    if pd.api.types.is_numeric_dtype(series):
        return ColumnKind.NUMERIC
    if series.dtype == object or pd.api.types.is_string_dtype(series):
        nunique = series.nunique(dropna=True)
        avg_len = series.dropna().astype(str).str.len().mean() if len(series) else 0
        if avg_len and avg_len > 50:
            return ColumnKind.TEXT
        if nunique <= min(50, max(1, len(series) // 10)):
            return ColumnKind.CATEGORICAL
        return ColumnKind.TEXT
    if pd.api.types.is_categorical_dtype(series):
        return ColumnKind.CATEGORICAL
    return ColumnKind.UNKNOWN


def infer_schema(df: pd.DataFrame) -> DatasetSchema:
    schema = DatasetSchema()
    for col in df.columns:
        series = df[col]
        kind = infer_column_kind(series)
        schema.columns.append(
            ColumnSchema(
                name=str(col),
                dtype=str(series.dtype),
                kind=kind,
                role=ColumnRole.FEATURE,
                nullable=bool(series.isnull().any()),
                unique_count=int(series.nunique(dropna=True)),
                sample_values=series.dropna().head(3).tolist(),
            )
        )
    return schema


def detect_task_type(target: pd.Series) -> str:
    """Heuristic task detection from target column."""
    if pd.api.types.is_numeric_dtype(target):
        nunique = target.nunique()
        if nunique <= 20 and nunique / max(len(target), 1) < 0.05:
            return "CLASSIFICATION"
        return "REGRESSION"
    return "CLASSIFICATION"
