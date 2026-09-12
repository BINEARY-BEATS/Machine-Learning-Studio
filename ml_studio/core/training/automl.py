"""Controlled AutoML workflow."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score

from ml_studio.core.training.cv import recommend_cv_strategy
from ml_studio.core.training.registry import get_model, get_models_for_task
from ml_studio.core.training.task import TaskType


@dataclass
class LeaderboardEntry:
    rank: int
    model_id: str
    model_name: str
    cv_score: float
    validation_score: float | None
    training_time: float
    inference_time: float
    parameters: dict[str, Any]


@dataclass
class AutoMLResult:
    entries: list[LeaderboardEntry] = field(default_factory=list)
    best_model_id: str = ""
    best_estimator: Any = None


class AutoMLRunner:
    def __init__(
        self,
        task: TaskType,
        max_models: int = 5,
        max_trials_per_model: int = 1,
        max_runtime_seconds: float = 300,
    ) -> None:
        self.task = task
        self.max_models = max_models
        self.max_trials = max_trials_per_model
        self.max_runtime = max_runtime_seconds
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
        progress_callback: Callable[[str], None] | None = None,
    ) -> AutoMLResult:
        models = get_models_for_task(self.task)[: self.max_models]
        scoring = "r2" if self.task == TaskType.REGRESSION else "f1_weighted"
        if self.task == TaskType.CLUSTERING:
            scoring = "silhouette"
        cv = recommend_cv_strategy(self.task, len(X_train))

        entries: list[LeaderboardEntry] = []
        start_total = time.time()
        best_score = float("-inf")
        best_model_id = ""
        best_estimator = None

        for i, meta in enumerate(models):
            if self._cancelled or time.time() - start_total > self.max_runtime:
                break
            model_id = next(k for k, v in __import__(
                "ml_studio.core.training.registry", fromlist=["MODEL_REGISTRY"]
            ).MODEL_REGISTRY.items() if v.name == meta.name)

            if progress_callback:
                progress_callback(f"Training {meta.name} ({i+1}/{len(models)})")

            model = get_model(model_id)
            t0 = time.time()
            try:
                if self.task in (TaskType.CLUSTERING, TaskType.ANOMALY_DETECTION):
                    model.fit(X_train)
                    train_time = time.time() - t0
                    cv_score = 0.0
                else:
                    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
                    cv_score = float(np.mean(scores))
                    model.fit(X_train, y_train)
                    train_time = time.time() - t0
            except Exception:
                continue

            t1 = time.time()
            if self.task not in (TaskType.CLUSTERING, TaskType.ANOMALY_DETECTION):
                _ = model.predict(X_train.head(min(100, len(X_train))))
            else:
                if hasattr(model, "predict"):
                    _ = model.predict(X_train.head(min(100, len(X_train))))
            infer_time = time.time() - t1

            val_score = None
            if X_val is not None and y_val is not None and self.task not in (
                TaskType.CLUSTERING,
                TaskType.ANOMALY_DETECTION,
            ):
                from sklearn.metrics import f1_score, r2_score

                preds = model.predict(X_val)
                val_score = float(
                    r2_score(y_val, preds)
                    if self.task == TaskType.REGRESSION
                    else f1_score(y_val, preds, average="weighted")
                )

            entries.append(
                LeaderboardEntry(
                    rank=0,
                    model_id=model_id,
                    model_name=meta.name,
                    cv_score=cv_score,
                    validation_score=val_score,
                    training_time=train_time,
                    inference_time=infer_time,
                    parameters=meta.default_params,
                )
            )
            if cv_score > best_score:
                best_score = cv_score
                best_model_id = model_id
                best_estimator = model

        entries.sort(key=lambda e: e.cv_score, reverse=True)
        for rank, entry in enumerate(entries, 1):
            entry.rank = rank

        return AutoMLResult(entries=entries, best_model_id=best_model_id, best_estimator=best_estimator)
