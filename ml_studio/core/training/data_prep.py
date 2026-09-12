"""Prepare tabular data for sklearn training."""

from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from ml_studio.core.training.task import TaskType


def prepare_for_training(
    df: pd.DataFrame,
    task: TaskType,
    target_column: str | None = None,
    feature_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, str, list[str], dict]:
    """
    Clean and encode data for training.

    Returns (prepared_df, target_column, feature_columns, encoders_meta).
    """
    work = df.copy()
    meta: dict = {"label_encoders": {}}

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
        return work, "", cols, meta

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
        raise ValueError(f"Need at least 10 rows after removing missing target values; got {len(work)}.")

    X_cols: list[str] = []
    for col in features:
        if pd.api.types.is_numeric_dtype(work[col]):
            X_cols.append(col)
        else:
            le = LabelEncoder()
            work[col] = le.fit_transform(work[col].astype(str))
            meta["label_encoders"][col] = le
            X_cols.append(col)

    if task == TaskType.CLASSIFICATION and not pd.api.types.is_numeric_dtype(work[target_column]):
        le_y = LabelEncoder()
        work[target_column] = le_y.fit_transform(work[target_column].astype(str))
        meta["label_encoders"][target_column] = le_y

    work = work.dropna()
    if len(work) < 10:
        raise ValueError(f"Need at least 10 complete rows after encoding; got {len(work)}.")

    return work, target_column, X_cols, meta


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
