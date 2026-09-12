"""Model training engine."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline as SklearnPipeline

from ml_studio.app.logger import get_logger
from ml_studio.core.pipeline import PreprocessingPipeline
from ml_studio.core.training.cv import recommend_cv_strategy
from ml_studio.core.training.registry import get_model
from ml_studio.core.training.task import TaskType

logger = get_logger("trainer")


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


@dataclass
class TrainingResult:
    experiment_id: str
    model_id: str
    task: TaskType
    metrics: dict[str, Any]
    cv_scores: dict[str, float]
    training_duration: float
    estimator: Any
    preprocessing: PreprocessingPipeline | None
    feature_columns: list[str]
    target_column: str
    train_size: int
    test_size: int


class Trainer:
    def __init__(self) -> None:
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def train(
        self,
        df: pd.DataFrame,
        config: TrainingConfig,
        preprocessing: PreprocessingPipeline | None = None,
        progress_callback: Callable[[int, str], None] | None = None,
    ) -> TrainingResult:
        start = time.time()
        experiment_id = str(uuid.uuid4())

        if self._cancelled:
            raise InterruptedError("Training cancelled by user")

        if progress_callback:
            progress_callback(5, f"Loaded {len(df):,} rows, {len(config.feature_columns)} features, target '{config.target_column}'")

        X = df[config.feature_columns]
        y = df[config.target_column] if config.task not in (
            TaskType.CLUSTERING,
            TaskType.ANOMALY_DETECTION,
        ) else None

        if preprocessing:
            if progress_callback:
                progress_callback(12, f"Applying preprocessing pipeline ({len(preprocessing.nodes)} step(s))…")
            if y is not None:
                X_processed = preprocessing.fit_transform(X, y)
            else:
                X_processed = preprocessing.fit_transform(X)
        else:
            if progress_callback:
                progress_callback(12, "No preprocessing pipeline — using prepared features as-is")
            X_processed = X.copy()

        if config.task in (TaskType.CLUSTERING, TaskType.ANOMALY_DETECTION):
            X_train = X_processed
            X_test = pd.DataFrame()
            y_train = y_test = None
        else:
            if progress_callback:
                progress_callback(20, f"Splitting data ({int((1-config.test_size)*100)}% train / {int(config.test_size*100)}% test)…")
            stratify = None
            if config.task == TaskType.CLASSIFICATION:
                class_counts = y.value_counts()
                if y.nunique() <= 20 and class_counts.min() >= 2:
                    stratify = y
            X_train, X_test, y_train, y_test = train_test_split(
                X_processed,
                y,
                test_size=config.test_size,
                random_state=config.random_state,
                stratify=stratify,
            )

        from ml_studio.core.training.registry import MODEL_REGISTRY
        model_label = MODEL_REGISTRY.get(config.model_id)
        model_name = model_label.name if model_label else config.model_id

        if progress_callback:
            progress_callback(30, f"Building {model_name}…")

        model = get_model(config.model_id, **config.hyperparameters)

        if self._cancelled:
            raise RuntimeError("Training cancelled")

        cv_scores_arr = np.array([])
        y_test_eval = None
        if config.task in (TaskType.CLUSTERING, TaskType.ANOMALY_DETECTION):
            if self._cancelled:
                raise InterruptedError("Training cancelled by user")
            if progress_callback:
                progress_callback(55, f"Fitting {model_name} on {len(X_train):,} samples…")
            model.fit(X_train)
            if self._cancelled:
                raise InterruptedError("Training cancelled by user")
            preds = model.predict(X_train)
        else:
            n_classes = y_train.nunique() if config.task == TaskType.CLASSIFICATION else None
            if self._cancelled:
                raise InterruptedError("Training cancelled by user")
            cv = recommend_cv_strategy(
                config.task,
                len(X_train),
                n_classes=n_classes,
                is_time_series=config.is_time_series,
                n_splits=config.cv_splits,
            )
            scoring = self._scoring_for_task(config.task)
            cv_name = type(cv).__name__
            if progress_callback:
                progress_callback(
                    45,
                    f"Cross-validation ({cv_name}, {config.cv_splits} folds, scoring={scoring}) on {len(X_train):,} training rows…",
                )
            if self._cancelled:
                raise InterruptedError("Training cancelled by user")
            cv_scores_arr = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1)
            if progress_callback:
                progress_callback(
                    65,
                    f"CV complete — mean {scoring}={float(np.mean(cv_scores_arr)):.4f}. Fitting final model…",
                )
            if self._cancelled:
                raise InterruptedError("Training cancelled by user")
            model.fit(X_train, y_train)
            if progress_callback:
                progress_callback(80, f"Evaluating on {len(X_test):,} held-out test rows…")
            preds = model.predict(X_test)
            y_test_eval = y_test

        duration = time.time() - start

        if progress_callback:
            progress_callback(92, "Computing evaluation metrics…")

        metrics = self._compute_metrics(
            config.task,
            model,
            X_train,
            preds,
            y_test_eval,
            y_train,
        )

        cv_scores = {}
        if len(cv_scores_arr):
            cv_scores = {"mean": float(np.mean(cv_scores_arr)), "std": float(np.std(cv_scores_arr))}

        if progress_callback:
            primary = metrics.get("r2") or metrics.get("f1") or metrics.get("accuracy") or metrics.get("silhouette")
            summary = f"{primary:.4f}" if isinstance(primary, float) else "done"
            progress_callback(100, f"Training complete — primary score: {summary}")

        return TrainingResult(
            experiment_id=experiment_id,
            model_id=config.model_id,
            task=config.task,
            metrics=metrics,
            cv_scores=cv_scores,
            training_duration=duration,
            estimator=model,
            preprocessing=preprocessing,
            feature_columns=config.feature_columns,
            target_column=config.target_column,
            train_size=len(X_train),
            test_size=len(X_test) if len(X_test) else 0,
        )

    def _scoring_for_task(self, task: TaskType) -> str:
        return {
            TaskType.REGRESSION: "r2",
            TaskType.CLASSIFICATION: "f1_weighted",
            TaskType.TIME_SERIES: "r2",
        }.get(task, "r2")

    def _compute_metrics(
        self,
        task: TaskType,
        model,
        X_train,
        preds,
        y_test,
        y_train,
    ) -> dict[str, Any]:
        from ml_studio.core.evaluation.metrics import compute_metrics

        if task in (TaskType.CLUSTERING, TaskType.ANOMALY_DETECTION):
            return compute_metrics(task, y_true=None, y_pred=preds, X=X_train, model=model)
        return compute_metrics(task, y_true=y_test, y_pred=preds)
