"""Cross-validation strategies."""

from __future__ import annotations

from sklearn.model_selection import (
    GroupKFold,
    KFold,
    StratifiedKFold,
    TimeSeriesSplit,
)

from ml_studio.core.training.task import TaskType


def recommend_cv_strategy(
    task: TaskType,
    n_samples: int,
    n_classes: int | None = None,
    has_groups: bool = False,
    is_time_series: bool = False,
    n_splits: int = 5,
):
    if is_time_series or task == TaskType.TIME_SERIES:
        return TimeSeriesSplit(n_splits=n_splits)
    if has_groups:
        return GroupKFold(n_splits=n_splits)
    if task == TaskType.CLASSIFICATION and n_classes and n_classes >= 2:
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    return KFold(n_splits=n_splits, shuffle=True, random_state=42)


def get_cv_strategy(name: str, n_splits: int = 5, **kwargs):
    strategies = {
        "kfold": KFold(n_splits=n_splits, shuffle=True, random_state=42),
        "stratified": StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42),
        "group": GroupKFold(n_splits=n_splits),
        "timeseries": TimeSeriesSplit(n_splits=n_splits),
    }
    if name not in strategies:
        raise ValueError(f"Unknown CV strategy: {name}")
    return strategies[name]
