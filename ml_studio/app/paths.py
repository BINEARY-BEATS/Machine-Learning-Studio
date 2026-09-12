"""Centralized path resolution for the application."""

from __future__ import annotations

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT.parent
ASSETS_DIR = PACKAGE_ROOT / "assets"
ICONS_DIR = ASSETS_DIR / "icons"
FONTS_DIR = ASSETS_DIR / "fonts"
THEMES_DIR = ASSETS_DIR / "themes"
QSS_DIR = ASSETS_DIR / "qss"
LOGS_DIR = PROJECT_ROOT / "logs"
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_PROJECTS_DIR = Path.home() / "MLStudioProjects"


def ensure_runtime_dirs() -> None:
    """Create runtime directories if they do not exist."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    DEFAULT_PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
