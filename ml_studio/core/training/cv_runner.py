"""Leak-free train/test split, preprocessing, and cross-validation (no Qt)."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.metrics import get_scorer
from sklearn.model_selection import train_test_split

from ml_studio.core.inference.schema_coerce import needs_auto_encode
from ml_studio.core.pipeline import Pipeline
from ml_studio.core.training.data_prep import EncodingBundle
from ml_studio.core.training.task import TaskType
from ml_studio.transforms.auto_encode import AutoEncode


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    config: Any,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split RAW (un-preprocessed) data. Time series → chronological, no stratify."""
    is_ts = bool(getattr(config, "is_time_series", False))
    task = getattr(config, "task", None)
    if is_ts or task == TaskType.TIME_SERIES:
        return train_test_split(
            X,
            y,
            test_size=config.test_size,
            random_state=config.random_state,
            shuffle=False,
        )

    stratify = None
    if task == TaskType.CLASSIFICATION and y is not None:
        counts = y.value_counts()
        if y.nunique() <= 20 and int(counts.min()) >= 2:
            stratify = y
    return train_test_split(
        X,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=stratify,
    )


def fit_preprocessing(
    pipeline: Pipeline | None,
    X_train: pd.DataFrame,
    y_train: pd.Series | None,
) -> tuple[Pipeline | None, pd.DataFrame, pd.Series | None]:
    """Clone, fit_transform on TRAIN only, align y to surviving row index."""
    if pipeline is None or not getattr(pipeline, "steps", None):
        y_out = y_train.copy() if y_train is not None else None
        return None, X_train.copy(), y_out

    fitted = pipeline.clone_unfitted()
    if y_train is not None:
        X_t = fitted.fit_transform(X_train, y_train)
        y_aligned = y_train.loc[X_t.index]
    else:
        X_t = fitted.fit_transform(X_train)
        y_aligned = None
    return fitted, X_t, y_aligned


def apply_preprocessing(
    fitted: Pipeline | None,
    X: pd.DataFrame,
    y: pd.Series | None,
) -> tuple[pd.DataFrame, pd.Series | None]:
    """Transform only; align y to surviving row index."""
    if fitted is None or not getattr(fitted, "steps", None):
        X_t = X.copy()
    else:
        X_t = fitted.transform(X)
    y_aligned = y.loc[X_t.index] if y is not None else None
    return X_t, y_aligned


def append_auto_encode(
    fitted: Pipeline | None,
    X_train: pd.DataFrame,
    y_train: pd.Series | None,
    X_test: pd.DataFrame | None = None,
    y_test: pd.Series | None = None,
) -> tuple[Pipeline, pd.DataFrame, pd.Series | None, pd.DataFrame | None, pd.Series | None]:
    """If non-numeric/NaN remain, fit AutoEncode on train and append to pipeline."""
    pipe = fitted if fitted is not None else Pipeline(steps=[])
    if not needs_auto_encode(X_train):
        return fitted if fitted is not None else pipe, X_train, y_train, X_test, y_test
    if pipe.steps and type(pipe.steps[-1]).__name__ == "AutoEncode":
        ae = pipe.steps[-1]
        X_tr = ae.transform(X_train)
    else:
        ae = AutoEncode()
        X_tr = ae.fit_transform(X_train, y_train)
        pipe.add(ae)
    y_tr = y_train.loc[X_tr.index] if y_train is not None else None
    X_te, y_te = X_test, y_test
    if X_test is not None and len(X_test):
        X_te = ae.transform(X_test)
        y_te = y_test.loc[X_te.index] if y_test is not None else None
    return pipe, X_tr, y_tr, X_te, y_te


def _align_dropna(
    X: pd.DataFrame,
    y: pd.Series | None,
) -> tuple[pd.DataFrame, pd.Series | None]:
    if y is None:
        return X.dropna(), None
    mask = X.notna().all(axis=1) & y.notna()
    return X.loc[mask], y.loc[mask]


def _encode_fold(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_va: pd.DataFrame,
    y_va: pd.Series,
    task: TaskType | None,
    target_column: str,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    pipe, X_tr, y_tr, X_va, y_va = append_auto_encode(None, X_tr, y_tr, X_va, y_va)
    del pipe  # fold-local AutoEncode only
    if task is None:
        return X_tr, y_tr, X_va, y_va
    enc = EncodingBundle().fit(X_tr, y_tr, task, target_column)
    y_tr = enc.transform_target(y_tr)
    y_va = enc.transform_target(y_va)
    X_tr, y_tr = _align_dropna(X_tr, y_tr)
    X_va, y_va = _align_dropna(X_va, y_va)
    return X_tr, y_tr, X_va, y_va


def cross_val_score_leakfree(
    model_factory: Callable[[], Any],
    preprocessing: Pipeline | None,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: Any,
    scoring: str,
    *,
    task: TaskType | None = None,
    target_column: str = "",
) -> np.ndarray:
    """Per fold: clone prep → fit fold-train → AutoEncode → transform fold-val → score."""
    scorer = get_scorer(scoring)
    scores: list[float] = []
    for tr_idx, va_idx in cv.split(X_train, y_train):
        X_tr = X_train.iloc[tr_idx]
        y_tr = y_train.iloc[tr_idx]
        X_va = X_train.iloc[va_idx]
        y_va = y_train.iloc[va_idx]
        fitted, X_tr_t, y_tr_a = fit_preprocessing(preprocessing, X_tr, y_tr)
        X_va_t, y_va_a = apply_preprocessing(fitted, X_va, y_va)
        assert y_tr_a is not None and y_va_a is not None
        X_tr_t, y_tr_a, X_va_t, y_va_a = _encode_fold(
            X_tr_t, y_tr_a, X_va_t, y_va_a, task, target_column
        )
        model = model_factory()
        model.fit(X_tr_t, y_tr_a)
        scores.append(float(scorer(model, X_va_t, y_va_a)))
    return np.asarray(scores, dtype=float)
