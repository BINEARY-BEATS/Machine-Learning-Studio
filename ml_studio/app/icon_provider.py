"""Themed icon loading and tinting from bundled PNG assets."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QIcon, QImage, QPainter, QPixmap, QColor

ICONS_DIR = Path(__file__).resolve().parent.parent / "assets" / "icons"


def icon_path(name: str) -> Path:
    """Resolve icon filename (with or without .png extension)."""
    stem = name[:-4] if name.lower().endswith(".png") else name
    return ICONS_DIR / f"{stem}.png"


def load_pixmap(name: str, size: int = 24) -> QPixmap:
    """Load and scale an icon pixmap."""
    path = icon_path(name)
    pixmap = QPixmap(str(path))
    if pixmap.isNull():
        fallback = ICONS_DIR / "missing.png"
        pixmap = QPixmap(str(fallback))
    return pixmap.scaled(
        size,
        size,
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )


def tinted_icon(name: str, color: str, size: int = 24) -> QIcon:
    """Return an icon recolored to a theme token hex value."""
    source = load_pixmap(name, size)
    image = source.toImage().convertToFormat(QImage.Format.Format_ARGB32)
    tinted = QImage(image.size(), QImage.Format.Format_ARGB32)
    tinted.fill(Qt.GlobalColor.transparent)

    painter = QPainter(tinted)
    painter.drawImage(0, 0, image)
    painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceIn)
    painter.fillRect(tinted.rect(), QColor(color))
    painter.end()

    icon = QIcon(QPixmap.fromImage(tinted))
    icon.addPixmap(QPixmap.fromImage(tinted), QIcon.Mode.Disabled, QIcon.State.Off)
    return icon


def themed_icon(name: str, mode_value: str, token: str = "text_muted", size: int = 24) -> QIcon:
    """Load icon tinted with a semantic token for the active theme."""
    from ml_studio.app.theme import ThemeMode, color_token

    mode = ThemeMode(mode_value)
    return tinted_icon(name, color_token(mode, token), size)
