"""Chart widgets for the ML lab — matplotlib when available, graceful fallback."""

from __future__ import annotations

from typing import Any

import numpy as np
from PyQt6.QtWidgets import QLabel, QSizePolicy, QVBoxLayout, QWidget


def create_plot_widget(title: str = ""):
    """Legacy helper — prefer specific chart builders below."""
    try:
        import pyqtgraph as pg

        widget = pg.PlotWidget(title=title)
        widget.setBackground(None)
        return widget
    except ImportError:
        return None


def downsample_xy(x, y, max_points: int = 10_000):
    """Largest-Triangle-Three-Buckets style simple downsampling."""
    if len(x) <= max_points:
        return x, y
    step = max(1, len(x) // max_points)
    indices = list(range(0, len(x), step))[:max_points]
    return [x[i] for i in indices], [y[i] for i in indices]


def _matplotlib_canvas(fig) -> QWidget | None:
    try:
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

        canvas = FigureCanvasQTAgg(fig)
        canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        return canvas
    except Exception:
        return None


def _fallback(msg: str) -> QWidget:
    lab = QLabel(msg)
    lab.setWordWrap(True)
    lab.setObjectName("Breadcrumb")
    return lab


def histogram_widget(series, title: str = "Distribution", bins: int = 30) -> QWidget:
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(4.5, 2.8), dpi=100)
        vals = np.asarray(series.dropna(), dtype=float)
        if len(vals) == 0:
            ax.text(0.5, 0.5, "No numeric data", ha="center", va="center")
        else:
            ax.hist(vals, bins=min(bins, max(5, len(vals) // 3)), color="#2DD4BF", edgecolor="#0C1012")
        ax.set_title(title, fontsize=10)
        ax.tick_params(labelsize=8)
        fig.tight_layout()
        canvas = _matplotlib_canvas(fig)
        plt.close(fig)
        return canvas or _fallback("Install matplotlib for charts")
    except Exception as exc:
        return _fallback(f"Chart unavailable: {exc}")


def missingness_widget(df) -> QWidget:
    try:
        import matplotlib.pyplot as plt

        miss = (df.isna().mean() * 100).sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(4.5, max(2.5, 0.28 * len(miss))), dpi=100)
        ax.barh(miss.index.astype(str), miss.values, color="#FBBF24")
        ax.set_xlabel("% missing", fontsize=9)
        ax.set_title("Missingness", fontsize=10)
        ax.tick_params(labelsize=8)
        fig.tight_layout()
        canvas = _matplotlib_canvas(fig)
        plt.close(fig)
        return canvas or _fallback("Install matplotlib for charts")
    except Exception as exc:
        return _fallback(f"Chart unavailable: {exc}")


def correlation_heatmap(df, max_cols: int = 12) -> QWidget:
    try:
        import matplotlib.pyplot as plt

        num = df.select_dtypes(include="number")
        if num.shape[1] < 2:
            return _fallback("Need ≥2 numeric columns for correlation")
        cols = list(num.columns)[:max_cols]
        corr = num[cols].corr()
        fig, ax = plt.subplots(figsize=(4.8, 4.0), dpi=100)
        im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(cols)))
        ax.set_yticks(range(len(cols)))
        ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(cols, fontsize=7)
        ax.set_title("Correlation", fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        canvas = _matplotlib_canvas(fig)
        plt.close(fig)
        return canvas or _fallback("Install matplotlib for charts")
    except Exception as exc:
        return _fallback(f"Chart unavailable: {exc}")


def confusion_matrix_widget(cm: Any, labels: list | None = None) -> QWidget:
    try:
        import matplotlib.pyplot as plt

        arr = np.asarray(cm)
        fig, ax = plt.subplots(figsize=(3.8, 3.2), dpi=100)
        im = ax.imshow(arr, cmap="Blues")
        ax.set_title("Confusion matrix", fontsize=10)
        n = arr.shape[0]
        tick = labels[:n] if labels else [str(i) for i in range(n)]
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(tick, fontsize=8)
        ax.set_yticklabels(tick, fontsize=8)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        for i in range(n):
            for j in range(n):
                ax.text(j, i, int(arr[i, j]), ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout()
        canvas = _matplotlib_canvas(fig)
        plt.close(fig)
        return canvas or _fallback("Install matplotlib for charts")
    except Exception as exc:
        return _fallback(f"Chart unavailable: {exc}")


def residual_parity_widget(y_true, y_pred) -> QWidget:
    try:
        import matplotlib.pyplot as plt

        yt = np.asarray(y_true, dtype=float)
        yp = np.asarray(y_pred, dtype=float)
        fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.8), dpi=100)
        axes[0].scatter(yt, yp, s=12, alpha=0.6, color="#2DD4BF")
        lo = min(yt.min(), yp.min())
        hi = max(yt.max(), yp.max())
        axes[0].plot([lo, hi], [lo, hi], color="#F87171", lw=1)
        axes[0].set_title("Parity", fontsize=10)
        axes[0].set_xlabel("Actual")
        axes[0].set_ylabel("Predicted")
        resid = yt - yp
        axes[1].hist(resid, bins=20, color="#22D3EE", edgecolor="#0C1012")
        axes[1].set_title("Residuals", fontsize=10)
        fig.tight_layout()
        canvas = _matplotlib_canvas(fig)
        plt.close(fig)
        return canvas or _fallback("Install matplotlib for charts")
    except Exception as exc:
        return _fallback(f"Chart unavailable: {exc}")


def chart_panel(*widgets: QWidget) -> QWidget:
    """Stack chart widgets vertically."""
    box = QWidget()
    lay = QVBoxLayout(box)
    lay.setContentsMargins(0, 0, 0, 0)
    for w in widgets:
        if w is not None:
            lay.addWidget(w)
    lay.addStretch(1)
    return box
