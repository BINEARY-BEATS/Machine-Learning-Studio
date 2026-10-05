"""Theme API — tokens, QSS, and application helpers."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QApplication

from ml_studio.app.metric_color import metric_color, metric_na_reason
from ml_studio.app.theme_qss import build_stylesheet
from ml_studio.app.theme_tokens import (
    FONT_SANS,
    MOTION,
    RADIUS,
    SPACE,
    TYPE,
    ColorPalette,
    ThemeMode,
    color_token,
    palette_for,
)

THEMES_DIR = Path(__file__).resolve().parent.parent / "assets" / "themes"

__all__ = [
    "FONT_SANS",
    "MOTION",
    "RADIUS",
    "SPACE",
    "TYPE",
    "ColorPalette",
    "ThemeMode",
    "THEMES_DIR",
    "apply_theme",
    "build_stylesheet",
    "color_token",
    "metric_color",
    "metric_na_reason",
    "palette_for",
    "write_theme_files",
]


def apply_theme(app: QApplication, mode: ThemeMode) -> None:
    """Apply global stylesheet and default font to the application."""
    app.setStyleSheet(build_stylesheet(mode))
    body = TYPE["body"]
    # Prefer installed system UI fonts — avoid unresolved family quirks on Windows
    font = QFont()
    font.setFamilies(["Segoe UI", "SF Pro Text", "IBM Plex Sans", "sans-serif"])
    size = int(body.get("size", 13))
    if size < 1:
        size = 13
    font.setPointSize(size)
    font.setWeight(QFont.Weight(int(body.get("weight", 400))))
    app.setFont(font)
    app.setProperty("themeMode", mode.value)


def write_theme_files() -> tuple[Path, Path]:
    """Write generated light/dark QSS files to assets/themes/."""
    THEMES_DIR.mkdir(parents=True, exist_ok=True)
    light_path = THEMES_DIR / "light.qss"
    dark_path = THEMES_DIR / "dark.qss"
    light_path.write_text(build_stylesheet(ThemeMode.LIGHT), encoding="utf-8")
    dark_path.write_text(build_stylesheet(ThemeMode.DARK), encoding="utf-8")
    return light_path, dark_path
