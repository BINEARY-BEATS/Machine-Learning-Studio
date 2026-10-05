"""Model training engine — split first, fit prep/encoders on train only."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split

from ml_studio.app.logger import get_logger
from ml_studio.core.pipeline import Pipeline
from ml_studio.core.training.cv import recommend_cv_strategy
from ml_studio.core.training.data_prep import EncodingBundle
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
        progress_callback: Callable[[int, str], None] | None = None,
    ) -> TrainingResult:
        start = time.time()
        experiment_id = str(uuid.uuid4())

        if self._cancelled:
            raise InterruptedError("Training cancelled by user")

        if progress_callback:
            progress_callback(
                5,
                f"Loaded {len(df):,} rows, {len(config.feature_columns)} features, "
                f"target '{config.target_column}'",
            )

        input_feature_columns = list(config.feature_columns)
        X_raw = df[input_feature_columns].copy()
        y_raw = (
            df[config.target_column].copy()
            if config.task
            not in (
                TaskType.CLUSTERING,
                TaskType.ANOMALY_DETECTION,
            )
            else None
        )

        # --- Split FIRST (before any fit) to avoid leakage ---
        encoding = EncodingBundle()
        fitted_preprocessing: Pipeline | None = None

        if config.task in (TaskType.CLUSTERING, TaskType.ANOMALY_DETECTION):
            X_train_raw, X_test_raw = X_raw, pd.DataFrame()
            y_train_raw = y_test_raw = None
        else:
            if progress_callback:
                progress_callback(
                    15,
                    f"Splitting data ({int((1 - config.test_size) * 100)}% train / "
                    f"{int(config.test_size * 100)}% test)…",
                )
            stratify = None
            if config.task == TaskType.CLASSIFICATION and y_raw is not None:
                class_counts = y_raw.value_counts()
                if y_raw.nunique() <= 20 and class_counts.min() >= 2:
                    stratify = y_raw
            X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
                X_raw,
                y_raw,
                test_size=config.test_size,
                random_state=config.random_state,
                stratify=stratify,
            )

        # --- Fit Prepare pipeline on TRAIN only ---
        if preprocessing is not None and getattr(preprocessing, "steps", None):
            n_steps = len([s for s in preprocessing.steps if getattr(s, "enabled", True)])
            if progress_callback:
                progress_callback(22, f"Fitting preprocessing on train only ({n_steps} step(s))…")
            fitted_preprocessing = preprocessing
            if y_train_raw is not None:
                fitted_preprocessing.fit(X_train_raw, y_train_raw)
            else:
                fitted_preprocessing.fit(X_train_raw)
            X_train = fitted_preprocessing.transform(X_train_raw)
            X_test = (
                fitted_preprocessing.transform(X_test_raw)
                if len(X_test_raw)
                else X_test_raw
            )
        else:
            if progress_callback:
                progress_callback(22, "No preprocessing pipeline — using raw features")
            X_train = X_train_raw.copy()
            X_test = X_test_raw.copy() if len(X_test_raw) else X_test_raw

        # Align feature columns after prep (columns may change)
        feature_columns = list(X_train.columns)

        # --- Fit encoders on TRAIN only ---
        if progress_callback:
            progress_callback(28, "Fitting feature/target encoders on train only…")
        encoding.fit(
            X_train,
            y_train_raw,
            config.task,
            config.target_column or "",
        )
        X_train = encoding.transform_features(X_train)
        if len(X_test):
            # Ensure test has same columns as train after prep
            for col in feature_columns:
                if col not in X_test.columns:
                    X_test[col] = np.nan
            X_test = X_test[feature_columns]
            X_test = encoding.transform_features(X_test)

        y_train = encoding.transform_target(y_train_raw) if y_train_raw is not None else None
        y_test = encoding.transform_target(y_test_raw) if y_test_raw is not None else None

        # Drop rows with NaN after prep/encode
        if y_train is not None:
            train_mask = X_train.notna().all(axis=1) & y_train.notna()
            X_train = X_train.loc[train_mask]
            y_train = y_train.loc[train_mask]
        else:
            X_train = X_train.dropna()
        if y_test is not None and len(X_test):
            test_mask = X_test.notna().all(axis=1) & y_test.notna()
            X_test = X_test.loc[test_mask]
            y_test = y_test.loc[test_mask]

        from ml_studio.core.training.registry import MODEL_REGISTRY

        model_label = MODEL_REGISTRY.get(config.model_id)
        model_name = model_label.name if model_label else config.model_id
        params = dict(config.hyperparameters)

        if (
            config.tune_method in ("optuna", "grid")
            and config.task not in (TaskType.CLUSTERING, TaskType.ANOMALY_DETECTION)
            and y_train is not None
        ):
            params = self._maybe_tune(
                config,
                X_train,
                y_train,
                model_label,
                progress_callback,
            )

        if progress_callback:
            progress_callback(35, f"Building {model_name}…")

        model = get_model(config.model_id, **params)

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
            n_folds = getattr(cv, "n_splits", config.cv_splits)
            if progress_callback:
                progress_callback(
                    40,
                    f"Cross-validation ({cv_name}, {n_folds} folds, scoring={scoring}) "
                    f"on {len(X_train):,} training rows — fold 1/{n_folds}…",
                )
            if self._cancelled:
                raise InterruptedError("Training cancelled by user")
            # Fold-by-fold so the UI updates (SVR/RF on large data can take minutes per fold)
            from sklearn.base import clone
            from sklearn.metrics import get_scorer

            scorer = get_scorer(scoring)
            fold_scores: list[float] = []
            splits = list(cv.split(X_train, y_train))
            for fi, (tr_idx, te_idx) in enumerate(splits):
                if self._cancelled:
                    raise InterruptedError("Training cancelled by user")
                pct = 40 + int(25 * fi / max(len(splits), 1))
                if progress_callback:
                    progress_callback(
                        pct,
                        f"CV fold {fi + 1}/{len(splits)} "
                        f"({cv_name}, scoring={scoring}, "
                        f"{len(tr_idx):,} train / {len(te_idx):,} val)…",
                    )
                est = clone(model)
                est.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
                fold_scores.append(
                    float(scorer(est, X_train.iloc[te_idx], y_train.iloc[te_idx]))
                )
            cv_scores_arr = np.asarray(fold_scores, dtype=float)
            if progress_callback:
                progress_callback(
                    68,
                    f"CV complete — mean {scoring}={float(np.mean(cv_scores_arr)):.4f} "
                    f"(±{float(np.std(cv_scores_arr)):.4f}). Fitting final model…",
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
            cv_scores = {
                "mean": float(np.mean(cv_scores_arr)),
                "std": float(np.std(cv_scores_arr)),
            }

        if progress_callback:
            primary = (
                metrics.get("r2")
                or metrics.get("f1")
                or metrics.get("accuracy")
                or metrics.get("silhouette")
            )
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
            preprocessing=fitted_preprocessing,
            feature_columns=feature_columns,
            target_column=config.target_column,
            train_size=len(X_train),
            test_size=len(X_test) if len(X_test) else 0,
            encoding=encoding,
            input_feature_columns=input_feature_columns,
            y_true_holdout=y_test_eval,
            y_pred_holdout=preds if config.task not in (
                TaskType.CLUSTERING,
                TaskType.ANOMALY_DETECTION,
            ) else None,
        )

    def _maybe_tune(
        self,
        config: TrainingConfig,
        X_train,
        y_train,
        model_label,
        progress_callback: Callable[[int, str], None] | None,
    ) -> dict[str, Any]:
        """Run Optuna or grid search when a param space exists; else keep defaults."""
        space = dict(getattr(model_label, "hyperparameters", None) or {})
        clean_space: dict[str, list[Any]] = {}
        for key, values in space.items():
            if not isinstance(values, (list, tuple)) or not values:
                continue
            vals = [v for v in values if v is not None]
            if vals:
                clean_space[key] = list(vals)
        if not clean_space:
            if progress_callback:
                progress_callback(28, "No tunable hyperparameters for this model — using defaults")
            return dict(config.hyperparameters)

        n_classes = y_train.nunique() if config.task == TaskType.CLASSIFICATION else None
        cv = recommend_cv_strategy(
            config.task,
            len(X_train),
            n_classes=n_classes,
            is_time_series=config.is_time_series,
            n_splits=min(config.cv_splits, 3),
        )
        scoring = self._scoring_for_task(config.task)

        if config.tune_method == "optuna":
            try:
                from ml_studio.core.training.tuning import OptunaTuner
            except Exception as exc:
                logger.warning("Optuna unavailable: %s", exc)
                if progress_callback:
                    progress_callback(28, "Optuna not installed — skipping tuning")
                return dict(config.hyperparameters)

            if progress_callback:
                progress_callback(25, f"Optuna tuning ({config.tune_trials} trials)…")
            tuner = OptunaTuner(
                config.model_id,
                clean_space,
                scoring=scoring,
                n_trials=max(1, int(config.tune_trials)),
            )
            if self._cancelled:
                tuner.cancel()

            def tune_progress(trial_n: int, msg: str) -> None:
                pct = 25 + min(10, int(10 * trial_n / max(config.tune_trials, 1)))
                if progress_callback:
                    progress_callback(pct, msg)

            result = tuner.tune(X_train, y_train, cv, progress_callback=tune_progress)
            if progress_callback:
                progress_callback(
                    35,
                    f"Best tune score={result.best_score:.4f} params={result.best_params}",
                )
            return result.best_params

        from itertools import product

        keys = list(clean_space.keys())
        combos = list(product(*(clean_space[k] for k in keys)))
        if len(combos) > 40:
            combos = combos[:40]
        if progress_callback:
            progress_callback(25, f"Grid search over {len(combos)} combinations…")

        best_score = float("-inf")
        best_params: dict[str, Any] = {}
        for i, combo in enumerate(combos):
            if self._cancelled:
                raise InterruptedError("Training cancelled by user")
            params = dict(zip(keys, combo))
            model = get_model(config.model_id, **params)
            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1)
            score = float(np.mean(scores))
            if score > best_score:
                best_score = score
                best_params = params
            if progress_callback and i % max(1, len(combos) // 5) == 0:
                progress_callback(
                    25 + int(10 * (i + 1) / len(combos)),
                    f"Grid {i + 1}/{len(combos)}: {score:.4f}",
                )
        if progress_callback:
            progress_callback(35, f"Best grid score={best_score:.4f}")
        return best_params or dict(config.hyperparameters)

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
