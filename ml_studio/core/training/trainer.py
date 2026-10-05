"""Model training engine — split RAW first; fit prep/encoders on train only."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

from ml_studio.core.inference.schema_coerce import build_feature_schema
from ml_studio.core.pipeline import Pipeline
from ml_studio.core.training.cv import recommend_cv_strategy
from ml_studio.core.training.cv_runner import (
    append_auto_encode,
    apply_preprocessing,
    cross_val_score_leakfree,
    fit_preprocessing,
    split_data,
)
from ml_studio.core.training.data_prep import EncodingBundle
from ml_studio.core.training.registry import MODEL_REGISTRY, get_model
from ml_studio.core.training.task import TaskType
from ml_studio.core.training.tuning import maybe_tune, scoring_for_task

_UNSUPERVISED = (TaskType.CLUSTERING, TaskType.ANOMALY_DETECTION)
ProgressCb = Callable[[int, str], None] | None


@dataclass
class TrainingConfig:
    task: TaskType
    model_id: str
    target_column: str
    feature_columns: list[str]
    test_size: float = 0.2
    random_state: int = 42
    cv_splits: int = 5
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    is_time_series: bool = False
    tune_method: str = "none"  # none | grid | optuna
    tune_trials: int = 20


@dataclass
class TrainingResult:
    experiment_id: str
    model_id: str
    task: TaskType
    metrics: dict[str, Any]
    cv_scores: dict[str, float]
    training_duration: float
    estimator: Any
    preprocessing: Pipeline | None
    feature_columns: list[str]
    target_column: str
    train_size: int
    test_size: int
    encoding: EncodingBundle | None = None
    input_feature_columns: list[str] = field(default_factory=list)
    target_classes: list[str] | None = None
    feature_schema: dict[str, Any] = field(default_factory=dict)
    y_true_holdout: Any = None
    y_pred_holdout: Any = None


class Trainer:
    def __init__(self) -> None:
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def train(
        self,
        df: pd.DataFrame,
        config: TrainingConfig,
        preprocessing: Pipeline | None = None,
        progress_callback: ProgressCb = None,
    ) -> TrainingResult:
        start = time.time()
        experiment_id = str(uuid.uuid4())
        self._check_cancel()
        self._emit(
            progress_callback,
            5,
            f"Loaded {len(df):,} rows, {len(config.feature_columns)} features, "
            f"target '{config.target_column}'",
        )
        input_cols = list(config.feature_columns)
        X_raw = df[input_cols].copy()
        y_raw = None if config.task in _UNSUPERVISED else df[config.target_column].copy()
        bundles = self._prepare_splits(X_raw, y_raw, config, preprocessing, progress_callback)
        params = self._resolve_params(config, bundles, preprocessing, progress_callback)
        return self._finish(
            experiment_id, start, config, preprocessing, params, bundles, input_cols,
            progress_callback,
        )

    def _prepare_splits(self, X_raw, y_raw, config, preprocessing, cb):
        X_tr_r, X_te_r, y_tr_r, y_te_r = self._split(X_raw, y_raw, config, cb)
        schema = build_feature_schema(X_tr_r) if len(X_tr_r) else {}
        fitted, X_tr, y_tr, X_te, y_te = self._prepare(
            preprocessing, X_tr_r, y_tr_r, X_te_r, y_te_r, cb
        )
        fitted, X_tr, y_tr, X_te, y_te = append_auto_encode(
            fitted, X_tr, y_tr, X_te if len(X_te) else None, y_te
        )
        if X_te is None:
            X_te, y_te = pd.DataFrame(), y_te_r
        encoding, X_tr, y_tr, X_te, y_te = self._encode_target(
            X_tr, y_tr, X_te, y_te, config, cb
        )
        return {
            "fitted_prep": fitted,
            "encoding": encoding,
            "feature_schema": schema,
            "target_classes": list(encoding.target_classes) if encoding.target_classes else None,
            "X_train_raw": X_tr_r,
            "y_train_raw": y_tr_r,
            "X_train": X_tr,
            "y_train": y_tr,
            "X_test": X_te,
            "y_test": y_te,
            "feature_columns": list(X_tr.columns),
        }

    def _resolve_params(self, config, bundles, preprocessing, cb) -> dict[str, Any]:
        if (
            config.tune_method in ("optuna", "grid")
            and config.task not in _UNSUPERVISED
            and bundles["y_train_raw"] is not None
        ):
            return maybe_tune(
                config,
                bundles["X_train_raw"],
                bundles["y_train_raw"],
                preprocessing,
                lambda: self._cancelled,
                cb,
            )
        return dict(config.hyperparameters)

    def _finish(self, experiment_id, start, config, preprocessing, params, b, input_cols, cb):
        meta = MODEL_REGISTRY.get(config.model_id)
        name = meta.name if meta else config.model_id
        self._emit(cb, 35, f"Building {name}…")
        model = get_model(config.model_id, **params)
        self._check_cancel()
        cv_arr, preds, y_eval = self._fit_and_eval(
            model, name, config, preprocessing, params, b, cb
        )
        metrics = self._compute_metrics(
            config.task, model, b["X_train"], preds, y_eval, b["y_train"]
        )
        self._emit_done(cb, metrics)
        cv_scores = {}
        if len(cv_arr):
            cv_scores = {"mean": float(np.mean(cv_arr)), "std": float(np.std(cv_arr))}
        return TrainingResult(
            experiment_id=experiment_id,
            model_id=config.model_id,
            task=config.task,
            metrics=metrics,
            cv_scores=cv_scores,
            training_duration=time.time() - start,
            estimator=model,
            preprocessing=b["fitted_prep"],
            feature_columns=b["feature_columns"],
            target_column=config.target_column,
            train_size=len(b["X_train"]),
            test_size=len(b["X_test"]) if len(b["X_test"]) else 0,
            encoding=b["encoding"],
            input_feature_columns=input_cols,
            target_classes=b.get("target_classes"),
            feature_schema=b.get("feature_schema") or {},
            y_true_holdout=y_eval,
            y_pred_holdout=preds if config.task not in _UNSUPERVISED else None,
        )

    def _split(self, X_raw, y_raw, config, cb):
        if config.task in _UNSUPERVISED:
            return X_raw, pd.DataFrame(), None, None
        self._emit(
            cb,
            15,
            f"Splitting data ({int((1 - config.test_size) * 100)}% train / "
            f"{int(config.test_size * 100)}% test)…",
        )
        return split_data(X_raw, y_raw, config)

    def _prepare(self, preprocessing, X_tr_r, y_tr_r, X_te_r, y_te_r, cb):
        n_steps = 0
        if preprocessing is not None:
            n_steps = len([s for s in preprocessing.steps if getattr(s, "enabled", True)])
        msg = (
            f"Fitting preprocessing on train only ({n_steps} step(s))…"
            if n_steps
            else "No preprocessing pipeline — using raw features"
        )
        self._emit(cb, 22, msg)
        fitted, X_tr, y_tr = fit_preprocessing(preprocessing, X_tr_r, y_tr_r)
        if len(X_te_r):
            X_te, y_te = apply_preprocessing(fitted, X_te_r, y_te_r)
        else:
            X_te, y_te = X_te_r, y_te_r
        return fitted, X_tr, y_tr, X_te, y_te

    def _encode_target(self, X_train, y_train, X_test, y_test, config, cb):
        self._emit(cb, 28, "Fitting target encoder on train only…")
        encoding = EncodingBundle()
        encoding.fit(X_train, y_train, config.task, config.target_column or "")
        y_train = encoding.transform_target(y_train) if y_train is not None else None
        y_test = encoding.transform_target(y_test) if y_test is not None else None
        X_train, y_train = _dropna_xy(X_train, y_train)
        if y_test is not None and len(X_test):
            X_test, y_test = _dropna_xy(X_test, y_test)
        return encoding, X_train, y_train, X_test, y_test

    def _fit_and_eval(self, model, name, config, preprocessing, params, b, cb):
        if config.task in _UNSUPERVISED:
            from ml_studio.core.training.unsupervised import fit_predict_labels

            self._check_cancel()
            self._emit(cb, 55, f"Fitting {name} on {len(b['X_train']):,} samples…")
            labels = fit_predict_labels(model, b["X_train"])
            self._check_cancel()
            return np.array([]), labels, None
        cv_arr = self._run_cv(
            config, params, preprocessing, b["X_train_raw"], b["y_train_raw"], cb
        )
        self._emit(
            cb,
            68,
            f"CV complete — mean={float(np.mean(cv_arr)):.4f} "
            f"(±{float(np.std(cv_arr)):.4f}). Fitting final model…",
        )
        self._check_cancel()
        model.fit(b["X_train"], b["y_train"])
        self._emit(cb, 80, f"Evaluating on {len(b['X_test']):,} held-out test rows…")
        self._emit(cb, 92, "Computing evaluation metrics…")
        return cv_arr, model.predict(b["X_test"]), b["y_test"]

    def _run_cv(self, config, params, preprocessing, X_raw, y_raw, cb):
        n_classes = y_raw.nunique() if config.task == TaskType.CLASSIFICATION else None
        cv = recommend_cv_strategy(
            config.task,
            len(X_raw),
            n_classes=n_classes,
            is_time_series=config.is_time_series,
            n_splits=config.cv_splits,
        )
        scoring = scoring_for_task(config.task)
        n_folds = getattr(cv, "n_splits", config.cv_splits)
        self._emit(
            cb,
            40,
            f"Cross-validation ({type(cv).__name__}, {n_folds} folds, scoring={scoring}) "
            f"on {len(X_raw):,} training rows…",
        )
        self._check_cancel()

        def factory():
            return get_model(config.model_id, **params)

        return cross_val_score_leakfree(
            factory,
            preprocessing,
            X_raw,
            y_raw,
            cv,
            scoring,
            task=config.task,
            target_column=config.target_column or "",
        )

    def _compute_metrics(self, task, model, X_train, preds, y_test, y_train):
        from ml_studio.core.evaluation.metrics import compute_metrics

        if task in _UNSUPERVISED:
            return compute_metrics(task, y_true=None, y_pred=preds, X=X_train, model=model)
        return compute_metrics(task, y_true=y_test, y_pred=preds)

    def _check_cancel(self) -> None:
        if self._cancelled:
            raise InterruptedError("Training cancelled by user")

    @staticmethod
    def _emit(cb, pct: int, msg: str) -> None:
        if cb:
            cb(pct, msg)

    @staticmethod
    def _emit_done(cb, metrics: dict) -> None:
        if not cb:
            return
        primary = (
            metrics.get("r2")
            or metrics.get("f1")
            or metrics.get("accuracy")
            or metrics.get("silhouette")
        )
        summary = f"{primary:.4f}" if isinstance(primary, float) else "done"
        cb(100, f"Training complete — primary score: {summary}")


def _dropna_xy(X, y):
    if y is None:
        return X.dropna(), None
    mask = X.notna().all(axis=1) & y.notna()
    return X.loc[mask], y.loc[mask]
