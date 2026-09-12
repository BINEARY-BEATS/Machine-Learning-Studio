"""Chart utilities using pyqtgraph when available."""

from __future__ import annotations


def create_plot_widget(title: str = ""):
    try:
        import pyqtgraph as pg
        widget = pg.PlotWidget(title=title)
        widget.setBackground("w")
        return widget
    except ImportError:
        return None


def downsample_xy(x, y, max_points: int = 10_000):
    """Largest-Triangle-Three-Buckets style simple downsampling."""
    if len(x) <= max_points:
        return x, y
    step = len(x) // max_points
    indices = list(range(0, len(x), step))[:max_points]
    return [x[i] for i in indices], [y[i] for i in indices]
