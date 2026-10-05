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
    "sm": 2,
    "md": 4,
    "lg": 6,
    "xl": 8,
    "full": 9999,
}

TYPE: dict[str, dict[str, int | str]] = {
    "display": {"size": 22, "weight": 700, "line": 28},
    "title": {"size": 16, "weight": 600, "line": 22},
    "body": {"size": 13, "weight": 400, "line": 20},
    "body_sm": {"size": 12, "weight": 400, "line": 16},
    "caption": {"size": 11, "weight": 500, "line": 14},
    "mono": {"size": 12, "weight": 400, "line": 16, "family": "Consolas, 'Cascadia Mono', monospace"},
}

MOTION: dict[str, int] = {
    "instant": 0,
    "fast": 100,
    "normal": 200,
    "slow": 300,
    "toast": 4000,
}

FONT_SANS = '"IBM Plex Sans", "Segoe UI", "SF Pro Text", system-ui, sans-serif'
FONT_MONO = "Consolas, 'Cascadia Mono', 'SF Mono', monospace"


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
    background="#F3F5F4",
    surface="#FFFFFF",
    surface_raised="#EEF2F0",
    border="#C5D0CB",
    border_subtle="#DCE4E0",
    text="#1A2421",
    text_muted="#5C6B66",
    text_disabled="#9AA8A3",
    primary="#0D9488",
    primary_hover="#0F766E",
    primary_subtle="#CCFBF1",
    success="#15803D",
    warning="#B45309",
    danger="#DC2626",
    info="#0E7490",
    metric_na="#9AA8A3",
    scrim="#0F14128C",
    on_primary="#FFFFFF",
)

DARK_PALETTE = ColorPalette(
    background="#0C1012",
    surface="#141A1C",
    surface_raised="#1A2226",
    border="#2A3538",
    border_subtle="#1F282C",
    text="#E8EEF0",
    text_muted="#8B9A9E",
    text_disabled="#4A5558",
    primary="#2DD4BF",
    primary_hover="#5EEAD4",
    primary_subtle="#134E4A",
    success="#4ADE80",
    warning="#FBBF24",
    danger="#F87171",
    info="#22D3EE",
    metric_na="#4A5558",
    scrim="#0C10128C",
    on_primary="#0C1012",
)


def palette_for(mode: ThemeMode) -> ColorPalette:
    return DARK_PALETTE if mode == ThemeMode.DARK else LIGHT_PALETTE


def color_token(mode: ThemeMode, name: str) -> str:
    """Return hex for a semantic token name."""
    p = palette_for(mode)
    return getattr(p, name)
