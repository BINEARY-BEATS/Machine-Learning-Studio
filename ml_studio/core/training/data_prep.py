"""Prepare tabular data for sklearn training — leakage-safe encoding."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from ml_studio.core.training.task import TaskType


@dataclass
class EncodingBundle:
    """Target LabelEncoder only (features use AutoEncode). Fit on train only."""

    feature_encoders: dict[str, LabelEncoder] = field(default_factory=dict)
    target_encoder: LabelEncoder | None = None
    feature_columns: list[str] = field(default_factory=list)
    target_column: str = ""
    target_classes: list[str] = field(default_factory=list)

    def transform_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """No-op for features — AutoEncode handles them. Kept for API compat."""
        return X.copy()

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
        self.feature_encoders = {}  # features → AutoEncode, never LabelEncode
        self.target_encoder = None
        self.target_classes = []
        if (
            y is not None
            and task == TaskType.CLASSIFICATION
            and not pd.api.types.is_numeric_dtype(y)
        ):
            le_y = LabelEncoder()
            le_y.fit(y.astype(str))
            self.target_encoder = le_y
            self.target_classes = [str(c) for c in le_y.classes_]
        return self


def select_training_frame(
    df: pd.DataFrame,
    task: TaskType,
    target_column: str | None = None,
    feature_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, str, list[str]]:
    """Column selection and NA cleanup only — no encoding."""
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
    Select columns, drop missing target, enforce min rows.
    Never encodes features. For classification with a non-numeric target, fit a
    LabelEncoder and return classes in meta['target_classes'] (frame target stays raw).
    """
    frame, target, features = select_training_frame(
        df, task, target_column=target_column, feature_columns=feature_columns
    )
    meta: dict[str, Any] = {"label_encoders": {}, "target_classes": None}
    if (
        task == TaskType.CLASSIFICATION
        and target
        and target in frame.columns
        and not pd.api.types.is_numeric_dtype(frame[target])
    ):
        le = LabelEncoder()
        le.fit(frame[target].astype(str))
        meta["target_classes"] = [str(c) for c in le.classes_]
    return frame, target, features, meta


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
