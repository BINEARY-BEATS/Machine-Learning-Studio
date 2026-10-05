"""Hyperparameter optimization — leak-free CV on RAW train data."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from itertools import product
from typing import Any, Callable

import numpy as np
import pandas as pd

from ml_studio.app.logger import get_logger
from ml_studio.core.pipeline import Pipeline
from ml_studio.core.training.cv import recommend_cv_strategy
from ml_studio.core.training.cv_runner import cross_val_score_leakfree
from ml_studio.core.training.registry import MODEL_REGISTRY, get_model
from ml_studio.core.training.task import TaskType

logger = get_logger("tuning")
ProgressCb = Callable[[int, str], None] | None


@dataclass
class TuningResult:
    best_params: dict[str, Any]
    best_score: float
    trials: list[dict[str, Any]] = field(default_factory=list)
    duration_seconds: float = 0.0


def clean_param_space(space: dict) -> dict[str, list[Any]]:
    clean: dict[str, list[Any]] = {}
    for key, values in space.items():
        if isinstance(values, (list, tuple)) and values:
            vals = [v for v in values if v is not None]
            if vals:
                clean[key] = list(vals)
    return clean


def scoring_for_task(task: TaskType) -> str:
    return {
        TaskType.REGRESSION: "r2",
        TaskType.CLASSIFICATION: "f1_weighted",
        TaskType.TIME_SERIES: "r2",
    }.get(task, "r2")


def maybe_tune(
    config: Any,
    X_raw: pd.DataFrame,
    y_raw: pd.Series,
    preprocessing: Pipeline | None,
    is_cancelled: Callable[[], bool],
    progress_callback: ProgressCb = None,
) -> dict[str, Any]:
    """Run Optuna or grid search on RAW X with leak-free CV; else return defaults."""
    model_label = MODEL_REGISTRY.get(config.model_id)
    clean = clean_param_space(dict(getattr(model_label, "hyperparameters", None) or {}))
    if not clean:
        _emit(progress_callback, 28, "No tunable hyperparameters — using defaults")
        return dict(config.hyperparameters)
    n_classes = y_raw.nunique() if config.task == TaskType.CLASSIFICATION else None
    cv = recommend_cv_strategy(
        config.task,
        len(X_raw),
        n_classes=n_classes,
        is_time_series=config.is_time_series,
        n_splits=min(config.cv_splits, 3),
    )
    scoring = scoring_for_task(config.task)
    if config.tune_method == "optuna":
        return _tune_optuna(
            config, X_raw, y_raw, preprocessing, clean, cv, scoring, is_cancelled,
            progress_callback,
        )
    return _tune_grid(
        config, X_raw, y_raw, preprocessing, clean, cv, scoring, is_cancelled,
        progress_callback,
    )


def _emit(cb: ProgressCb, pct: int, msg: str) -> None:
    if cb:
        cb(pct, msg)


def _tune_optuna(config, X_raw, y_raw, preprocessing, space, cv, scoring, is_cancelled, cb):
    try:
        import optuna  # noqa: F401
    except Exception as exc:
        logger.warning("Optuna unavailable: %s", exc)
        _emit(cb, 28, "Optuna not installed — skipping tuning")
        return dict(config.hyperparameters)
    _emit(cb, 25, f"Optuna tuning ({config.tune_trials} trials)…")
    tuner = OptunaTuner(
        config.model_id, space, scoring=scoring, n_trials=max(1, int(config.tune_trials))
    )
    if is_cancelled():
        tuner.cancel()

    def tune_progress(trial_n: int, msg: str) -> None:
        if is_cancelled():
            tuner.cancel()
        _emit(cb, 25 + min(10, int(10 * trial_n / max(config.tune_trials, 1))), msg)

    result = tuner.tune(
        X_raw,
        y_raw,
        cv,
        progress_callback=tune_progress,
        preprocessing=preprocessing,
        task=config.task,
        target_column=config.target_column or "",
    )
    _emit(cb, 35, f"Best tune score={result.best_score:.4f} params={result.best_params}")
    return result.best_params or dict(config.hyperparameters)


def _tune_grid(config, X_raw, y_raw, preprocessing, space, cv, scoring, is_cancelled, cb):
    keys = list(space.keys())
    combos = list(product(*(space[k] for k in keys)))[:40]
    _emit(cb, 25, f"Grid search over {len(combos)} combinations…")
    best_score, best_params = float("-inf"), {}
    for i, combo in enumerate(combos):
        if is_cancelled():
            raise InterruptedError("Training cancelled by user")
        params = dict(zip(keys, combo))

        def factory(p=params):
            return get_model(config.model_id, **p)

        score = float(
            np.mean(
                cross_val_score_leakfree(
                    factory,
                    preprocessing,
                    X_raw,
                    y_raw,
                    cv,
                    scoring,
                    task=config.task,
                    target_column=config.target_column or "",
                )
            )
        )
        if score > best_score:
            best_score, best_params = score, params
        if cb and i % max(1, len(combos) // 5) == 0:
            _emit(cb, 25 + int(10 * (i + 1) / len(combos)), f"Grid {i + 1}/{len(combos)}: {score:.4f}")
    _emit(cb, 35, f"Best grid score={best_score:.4f}")
    return best_params or dict(config.hyperparameters)


class OptunaTuner:
    def __init__(
        self,
        model_id: str,
        param_space: dict[str, list[Any]],
        scoring: str = "auto",
        n_trials: int = 20,
        timeout: float | None = None,
    ) -> None:
        self.model_id = model_id
        self.param_space = param_space
        self.n_trials = n_trials
        self.timeout = timeout
        self._cancelled = False
        meta = MODEL_REGISTRY[model_id]
        if scoring == "auto":
            tasks = [t.value for t in meta.task_types]
            self.scoring = "r2" if "REGRESSION" in tasks else "f1_weighted"
        else:
            self.scoring = scoring

    def cancel(self) -> None:
        self._cancelled = True

    def tune(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv,
        progress_callback: ProgressCb = None,
        preprocessing: Pipeline | None = None,
        task: TaskType | None = None,
        target_column: str = "",
    ) -> TuningResult:
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        trials_log: list[dict[str, Any]] = []
        start = time.time()
        study = optuna.create_study(direction="maximize")
        objective = self._make_objective(
            study, X, y, cv, preprocessing, task, target_column, trials_log, progress_callback
        )
        study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout)
        return self._to_result(study, trials_log, start)

    def _make_objective(
        self, study, X, y, cv, preprocessing, task, target_column, trials_log, progress_callback
    ):
        import optuna

        def objective(trial: optuna.Trial) -> float:
            if self._cancelled:
                study.stop()
                raise optuna.TrialPruned()
            params = {
                name: trial.suggest_categorical(name, list(values))
                for name, values in self.param_space.items()
            }

            def model_factory():
                return get_model(self.model_id, **params)

            score = float(
                np.mean(
                    cross_val_score_leakfree(
                        model_factory,
                        preprocessing,
                        X,
                        y,
                        cv,
                        self.scoring,
                        task=task,
                        target_column=target_column,
                    )
                )
            )
            trials_log.append({"params": params, "score": score})
            if progress_callback:
                progress_callback(len(trials_log), f"Trial {len(trials_log)}: {score:.4f}")
            return score

        return objective

    @staticmethod
    def _to_result(study, trials_log, start: float) -> TuningResult:
        if not study.best_trials:
            return TuningResult(
                best_params={},
                best_score=float("-inf"),
                trials=trials_log,
                duration_seconds=time.time() - start,
            )
        return TuningResult(
            best_params=study.best_params,
            best_score=study.best_value,
            trials=trials_log,
            duration_seconds=time.time() - start,
        )
