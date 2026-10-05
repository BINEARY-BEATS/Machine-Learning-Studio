"""Prepare tabular data for sklearn training — leakage-safe encoding."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from ml_studio.core.training.task import TaskType


@dataclass
class EncodingBundle:
    """Fitted feature/target LabelEncoders — fit on train only, reused at predict."""

    feature_encoders: dict[str, LabelEncoder] = field(default_factory=dict)
    target_encoder: LabelEncoder | None = None
    feature_columns: list[str] = field(default_factory=list)
    target_column: str = ""

    def transform_features(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        for col, le in self.feature_encoders.items():
            if col not in out.columns:
                continue
            raw = out[col].astype(str)
            known = set(le.classes_)
            # Unseen categories → most frequent class index (0) to avoid crash
            mapped = raw.map(lambda v, _le=le, _known=known: (
                int(_le.transform([v])[0]) if v in _known else 0
            ))
            out[col] = mapped.astype(float)
        return out

    def transform_target(self, y: pd.Series) -> pd.Series:
        if self.target_encoder is None:
            return y
        return pd.Series(
            self.target_encoder.transform(y.astype(str)),
            index=y.index,
            name=y.name,
        )

    def inverse_target(self, y_encoded) -> Any:
        if self.target_encoder is None:
            return y_encoded
        import numpy as np

        arr = np.asarray(y_encoded)
        flat = arr.ravel()
        decoded = self.target_encoder.inverse_transform(flat.astype(int))
        if arr.ndim == 0 or (arr.ndim == 1 and arr.shape == ()):
            return decoded[0]
        if len(decoded) == 1:
            return decoded[0]
        return decoded

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | None,
        task: TaskType,
        target_column: str,
    ) -> EncodingBundle:
        self.feature_columns = list(X.columns)
        self.target_column = target_column
        self.feature_encoders = {}
        for col in X.columns:
            if not pd.api.types.is_numeric_dtype(X[col]):
                le = LabelEncoder()
                le.fit(X[col].astype(str))
                self.feature_encoders[col] = le
        self.target_encoder = None
        if (
            y is not None
            and task == TaskType.CLASSIFICATION
            and not pd.api.types.is_numeric_dtype(y)
        ):
            le_y = LabelEncoder()
            le_y.fit(y.astype(str))
            self.target_encoder = le_y
        return self


def select_training_frame(
    df: pd.DataFrame,
    task: TaskType,
    target_column: str | None = None,
    feature_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, str, list[str]]:
    """
    Column selection and NA cleanup only — no encoding (avoids pre-split leakage).

    Returns (frame, target_column, feature_columns).
    """
    work = df.copy()

    if task in (TaskType.CLUSTERING, TaskType.ANOMALY_DETECTION):
        if feature_columns:
            cols = [c for c in feature_columns if c in work.columns]
        else:
            cols = work.select_dtypes(include="number").columns.tolist()
        if not cols:
            raise ValueError("No numeric feature columns available for this task.")
        work = work[cols].dropna()
        if len(work) < 10:
            raise ValueError(f"Need at least 10 rows after cleaning; got {len(work)}.")
        return work, "", cols

    if not target_column:
        numeric = work.select_dtypes(include="number").columns.tolist()
        if not numeric:
            raise ValueError("No numeric columns found for target selection.")
        target_column = numeric[-1]

    if target_column not in work.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset.")

    if feature_columns:
        features = [c for c in feature_columns if c in work.columns and c != target_column]
    else:
        features = [c for c in work.columns if c != target_column]

    if not features:
        raise ValueError("No feature columns selected.")

    work = work[[target_column] + features].dropna(subset=[target_column])
    if len(work) < 10:
        raise ValueError(
            f"Need at least 10 rows after removing missing target values; got {len(work)}."
        )
    return work, target_column, features


def prepare_for_training(
    df: pd.DataFrame,
    task: TaskType,
    target_column: str | None = None,
    feature_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, str, list[str], dict]:
    """
    Backward-compatible helper: select columns only (no full-frame encoding).

    Encoding is performed inside Trainer on the train split via EncodingBundle.
    Returns (frame, target_column, feature_columns, empty_meta).
    """
    frame, target, features = select_training_frame(
        df, task, target_column=target_column, feature_columns=feature_columns
    )
    return frame, target, features, {"label_encoders": {}}


def suggest_training_columns(df: pd.DataFrame) -> tuple[str, list[str], TaskType]:
    """Suggest target, features, and task type from dataset."""
    from ml_studio.core.schema import detect_task_type

    numeric = df.select_dtypes(include="number").columns.tolist()
    categorical = df.select_dtypes(include=["object", "category", "string"]).columns.tolist()

    if numeric:
        target = numeric[-1]
        features = numeric[:-1] + [c for c in categorical if c != target]
        task = TaskType(detect_task_type(df[target]))
        return target, features, task

    if categorical:
        target = categorical[0]
        features = [c for c in df.columns if c != target]
        return target, features, TaskType.CLASSIFICATION

    raise ValueError("Dataset has no usable columns for training.")
