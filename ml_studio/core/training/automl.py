"""Controlled AutoML workflow with cancellable per-model progress."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

from ml_studio.core.training.cv import recommend_cv_strategy
from ml_studio.core.training.interruptible import fit_score_cancellable
from ml_studio.core.training.registry import MODEL_REGISTRY, get_model, get_models_for_task
from ml_studio.core.training.task import TaskType

# Kernel / distance models dominate wall time on medium datasets.
_AUTOML_SKIP = frozenset(
    {
        "svr",
        "svc",
        "knn_classifier",
        "knn_regressor",
    }
)

# Lighter defaults so leaderboard folds finish quickly and Cancel can land sooner.
_AUTOML_FAST_PARAMS: dict[str, dict[str, Any]] = {
    "random_forest_classifier": {"n_estimators": 40, "n_jobs": 2},
    "random_forest_regressor": {"n_estimators": 40, "n_jobs": 2},
    "extra_trees_classifier": {"n_estimators": 40, "n_jobs": 2},
    "extra_trees_regressor": {"n_estimators": 40, "n_jobs": 2},
    "hist_gb_classifier": {"max_iter": 40},
    "hist_gb_regressor": {"max_iter": 40},
    "xgboost_classifier": {"n_estimators": 50, "n_jobs": 2},
    "xgboost_regressor": {"n_estimators": 50, "n_jobs": 2},
    "lightgbm_classifier": {"n_estimators": 50, "n_jobs": 2},
    "lightgbm_regressor": {"n_estimators": 50, "n_jobs": 2},
}


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


ProgressCb = Callable[[int, str], None]


class AutoMLRunner:
    def __init__(
        self,
        task: TaskType,
        max_models: int = 5,
        max_trials_per_model: int = 1,
        max_runtime_seconds: float = 300,
        n_cv_splits: int = 3,
    ) -> None:
        self.task = task
        self.max_models = max_models
        self.max_trials = max_trials_per_model
        self.max_runtime = max_runtime_seconds
        self.n_cv_splits = n_cv_splits
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
        progress_callback: ProgressCb | Callable[[str], None] | None = None,
    ) -> AutoMLResult:
        candidates = [
            m
            for m in get_models_for_task(self.task)
            if next(
                (k for k, v in MODEL_REGISTRY.items() if v.name == m.name),
                "",
            )
            not in _AUTOML_SKIP
        ]
        models = candidates[: self.max_models]
        scoring = "r2" if self.task == TaskType.REGRESSION else "f1_weighted"
        if self.task == TaskType.CLUSTERING:
            scoring = "silhouette"
        cv = recommend_cv_strategy(self.task, len(X_train), n_splits=self.n_cv_splits)

        entries: list[LeaderboardEntry] = []
        start_total = time.time()
        best_score = float("-inf")
        best_model_id = ""
        best_estimator = None
        n = max(len(models), 1)

        def _emit(pct: int, msg: str) -> None:
            if progress_callback is None:
                return
            try:
                progress_callback(pct, msg)  # type: ignore[call-arg]
            except TypeError:
                progress_callback(msg)  # type: ignore[misc]

        def _cancelled() -> bool:
            return self._cancelled

        for i, meta in enumerate(models):
            if self._cancelled or time.time() - start_total > self.max_runtime:
                _emit(
                    int(100 * i / n),
                    "AutoML stopped (cancelled or time limit).",
                )
                break

            model_id = next(
                (k for k, v in MODEL_REGISTRY.items() if v.name == meta.name),
                None,
            )
            if not model_id or model_id in _AUTOML_SKIP:
                continue

            base_pct = int(100 * i / n)
            _emit(
                base_pct,
                f"[{i + 1}/{len(models)}] {meta.name} — starting "
                f"{self.n_cv_splits}-fold CV on {len(X_train):,} rows…",
            )

            fast = dict(_AUTOML_FAST_PARAMS.get(model_id, {}))
            params = {**(getattr(meta, "default_params", {}) or {}), **fast}
            t0 = time.time()
            try:
                if self.task in (TaskType.CLUSTERING, TaskType.ANOMALY_DETECTION):
                    model = get_model(model_id, **fast)
                    model.fit(X_train)
                    train_time = time.time() - t0
                    cv_score = 0.0
                else:
                    fold_scores: list[float] = []
                    splits = list(cv.split(X_train, y_train))
                    for fi, (tr_idx, te_idx) in enumerate(splits):
                        if self._cancelled:
                            raise InterruptedError("cancelled")
                        fold_pct = base_pct + int((100 / n) * (fi / max(len(splits), 1)))
                        _emit(
                            fold_pct,
                            f"[{i + 1}/{len(models)}] {meta.name} — "
                            f"CV fold {fi + 1}/{len(splits)}…",
                        )
                        X_tr = X_train.iloc[tr_idx]
                        y_tr = y_train.iloc[tr_idx]
                        X_te = X_train.iloc[te_idx]
                        y_te = y_train.iloc[te_idx]
                        score = fit_score_cancellable(
                            model_id,
                            fast,
                            X_tr,
                            y_tr,
                            X_te,
                            y_te,
                            scoring,
                            _cancelled,
                        )
                        fold_scores.append(score)
                    cv_score = float(np.mean(fold_scores)) if fold_scores else 0.0
                    if self._cancelled:
                        raise InterruptedError("cancelled")
                    _emit(
                        base_pct + int(80 / n),
                        f"[{i + 1}/{len(models)}] {meta.name} — "
                        f"CV={cv_score:.4f}, fitting full train…",
                    )
                    model = get_model(model_id, **fast)
                    model.fit(X_train, y_train)
                    if self._cancelled:
                        raise InterruptedError("cancelled")
                    train_time = time.time() - t0
            except InterruptedError:
                break
            except Exception as exc:
                _emit(base_pct, f"[{i + 1}/{len(models)}] {meta.name} skipped ({exc})")
                continue

            t1 = time.time()
            try:
                _ = model.predict(X_train.head(min(100, len(X_train))))
            except Exception:
                pass
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
                    parameters=params,
                )
            )
            _emit(
                int(100 * (i + 1) / n),
                f"[{i + 1}/{len(models)}] {meta.name} done — "
                f"CV={cv_score:.4f}"
                + (f" val={val_score:.4f}" if val_score is not None else "")
                + f" ({train_time:.1f}s)",
            )
            if cv_score > best_score:
                best_score = cv_score
                best_model_id = model_id
                best_estimator = model

        entries.sort(key=lambda e: e.cv_score, reverse=True)
        for rank, entry in enumerate(entries, 1):
            entry.rank = rank

        _emit(100, f"Leaderboard ready — {len(entries)} model(s)")
        return AutoMLResult(
            entries=entries, best_model_id=best_model_id, best_estimator=best_estimator
        )
