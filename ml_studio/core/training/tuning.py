"""Hyperparameter optimization with Optuna."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score

from ml_studio.app.logger import get_logger
from ml_studio.core.training.registry import MODEL_REGISTRY, get_model

logger = get_logger("tuning")


@dataclass
class TuningResult:
    best_params: dict[str, Any]
    best_score: float
    trials: list[dict[str, Any]] = field(default_factory=list)
    duration_seconds: float = 0.0


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
            self.scoring = "r2" if "REGRESSION" in [t.value for t in meta.task_types] else "f1_weighted"
        else:
            self.scoring = scoring

    def cancel(self) -> None:
        self._cancelled = True

    def tune(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv,
        progress_callback: Callable[[int, str], None] | None = None,
    ) -> TuningResult:
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        trials_log: list[dict[str, Any]] = []
        start = time.time()

        def objective(trial):
            if self._cancelled:
                raise optuna.exceptions.OptunaError("Cancelled")
            params = {}
            for name, values in self.param_space.items():
                if all(isinstance(v, int) for v in values):
                    params[name] = trial.suggest_categorical(name, values)
                elif all(isinstance(v, float) for v in values):
                    params[name] = trial.suggest_categorical(name, values)
                else:
                    params[name] = trial.suggest_categorical(name, values)
            model = get_model(self.model_id, **params)
            scores = cross_val_score(model, X, y, cv=cv, scoring=self.scoring, n_jobs=-1)
            score = float(np.mean(scores))
            trials_log.append({"params": params, "score": score})
            if progress_callback:
                progress_callback(len(trials_log), f"Trial {len(trials_log)}: {score:.4f}")
            return score

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout)

        return TuningResult(
            best_params=study.best_params,
            best_score=study.best_value,
            trials=trials_log,
            duration_seconds=time.time() - start,
        )
