"""Design token definitions — single source of truth for colors."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ThemeMode(str, Enum):
    LIGHT = "light"
    DARK = "dark"


SPACE: dict[int, int] = {
    0: 0,
    1: 4,
    2: 8,
    3: 12,
    4: 16,
    5: 24,
    6: 32,
    7: 48,
    8: 64,
}

RADIUS: dict[str, int] = {
    "sm": 4,
    "md": 8,
    "lg": 12,
    "xl": 16,
    "full": 9999,
}

TYPE: dict[str, dict[str, int | str]] = {
    "display": {"size": 24, "weight": 700, "line": 32},
    "title": {"size": 18, "weight": 600, "line": 24},
    "body": {"size": 13, "weight": 400, "line": 20},
    "body_sm": {"size": 12, "weight": 400, "line": 16},
    "caption": {"size": 11, "weight": 500, "line": 14},
    "mono": {"size": 12, "weight": 400, "line": 16, "family": "Consolas"},
}

MOTION: dict[str, int] = {
    "instant": 0,
    "fast": 100,
    "normal": 200,
    "slow": 300,
    "toast": 4000,
}

FONT_SANS = '"Segoe UI", "SF Pro Text", system-ui, sans-serif'


@dataclass(frozen=True)
class ColorPalette:
    background: str
    surface: str
    surface_raised: str
    border: str
    border_subtle: str
    text: str
    text_muted: str
    text_disabled: str
    primary: str
    primary_hover: str
    primary_subtle: str
    success: str
    warning: str
    danger: str
    info: str
    metric_na: str
    scrim: str
    on_primary: str


LIGHT_PALETTE = ColorPalette(
    background="#F5F6F8",
    surface="#FFFFFF",
    surface_raised="#F0F2F5",
    border="#D8DCE2",
    border_subtle="#E8EAED",
    text="#1A1D21",
    text_muted="#6B7280",
    text_disabled="#9CA3AF",
    primary="#2563EB",
    primary_hover="#1D4ED8",
    primary_subtle="#EFF6FF",
    success="#16A34A",
    warning="#D97706",
    danger="#DC2626",
    info="#0891B2",
    metric_na="#9CA3AF",
    scrim="#0F11178C",
    on_primary="#FFFFFF",
)

DARK_PALETTE = ColorPalette(
    background="#0F1117",
    surface="#161B22",
    surface_raised="#1C2333",
    border="#30363D",
    border_subtle="#21262D",
    text="#E6EDF3",
    text_muted="#8B949E",
    text_disabled="#484F58",
    primary="#58A6FF",
    primary_hover="#79B8FF",
    primary_subtle="#1C3A5E",
    success="#3FB950",
    warning="#D29922",
    danger="#F85149",
    info="#39C5CF",
    metric_na="#484F58",
    scrim="#0F11178C",
    on_primary="#FFFFFF",
)


def palette_for(mode: ThemeMode) -> ColorPalette:
    return DARK_PALETTE if mode == ThemeMode.DARK else LIGHT_PALETTE


def color_token(mode: ThemeMode, name: str) -> str:
    """Return hex for a semantic token name."""
    p = palette_for(mode)
    return getattr(p, name)
