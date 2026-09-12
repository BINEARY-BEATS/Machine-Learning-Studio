"""Metric semantic coloring rules."""

from __future__ import annotations

import math
from typing import Any

from ml_studio.app.theme_tokens import ThemeMode, color_token


def metric_color(mode: ThemeMode, metric: str, value: Any, task: str = "") -> str:
    """Return semantic hex color for a metric value."""
    if value is None:
        return color_token(mode, "metric_na")
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return color_token(mode, "metric_na")

    name = metric.lower()

    if name in ("r2", "r_squared", "rsq"):
        return _r2_color(mode, float(value))
    if name in ("f1", "accuracy", "precision", "recall"):
        return _score_color(mode, float(value))
    if name == "silhouette":
        return _silhouette_color(mode, float(value))
    if name in ("mae", "mse", "rmse", "mape"):
        return color_token(mode, "text")
    return color_token(mode, "text")


def metric_na_reason(metric: str, task: str = "") -> str:
    name = metric.lower()
    if name == "silhouette":
        return "Silhouette requires at least 2 clusters with valid labels."
    if name in ("r2", "f1", "accuracy"):
        return "Metric could not be computed for this run."
    return "Not available for this task or model output."


def _r2_color(mode: ThemeMode, value: float) -> str:
    if value < 0:
        return color_token(mode, "danger")
    if value <= 0.5:
        return color_token(mode, "warning")
    return color_token(mode, "success")


def _score_color(mode: ThemeMode, value: float) -> str:
    if value < 0.5:
        return color_token(mode, "warning")
    return color_token(mode, "success")


def _silhouette_color(mode: ThemeMode, value: float) -> str:
    if value < 0.25:
        return color_token(mode, "danger")
    if value <= 0.5:
        return color_token(mode, "warning")
    return color_token(mode, "success")
