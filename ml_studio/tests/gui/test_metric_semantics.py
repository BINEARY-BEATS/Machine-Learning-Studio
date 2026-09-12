"""Metric semantic color tests."""

from __future__ import annotations

import math

from ml_studio.app.metric_color import metric_color, metric_na_reason
from ml_studio.app.theme import ThemeMode, color_token


def test_r2_negative_is_danger():
    color = metric_color(ThemeMode.LIGHT, "r2", -0.0002)
    assert color == color_token(ThemeMode.LIGHT, "danger")


def test_r2_good_is_success():
    color = metric_color(ThemeMode.LIGHT, "r2", 0.82)
    assert color == color_token(ThemeMode.LIGHT, "success")


def test_r2_mid_is_warning():
    color = metric_color(ThemeMode.LIGHT, "r2", 0.3)
    assert color == color_token(ThemeMode.LIGHT, "warning")


def test_none_uses_metric_na():
    color = metric_color(ThemeMode.DARK, "f1", None)
    assert color == color_token(ThemeMode.DARK, "metric_na")


def test_nan_uses_metric_na():
    color = metric_color(ThemeMode.LIGHT, "accuracy", float("nan"))
    assert color == color_token(ThemeMode.LIGHT, "metric_na")


def test_silhouette_bands():
    assert metric_color(ThemeMode.LIGHT, "silhouette", 0.1) == color_token(
        ThemeMode.LIGHT, "danger"
    )
    assert metric_color(ThemeMode.LIGHT, "silhouette", 0.4) == color_token(
        ThemeMode.LIGHT, "warning"
    )
    assert metric_color(ThemeMode.LIGHT, "silhouette", 0.7) == color_token(
        ThemeMode.LIGHT, "success"
    )


def test_na_reason_for_silhouette():
    assert "cluster" in metric_na_reason("silhouette").lower()
