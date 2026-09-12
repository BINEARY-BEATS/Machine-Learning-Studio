"""KPI stat card with semantic metric coloring."""

from __future__ import annotations

from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QLabel, QWidget

from ml_studio.app.metric_color import metric_color, metric_na_reason
from ml_studio.app.theme import ThemeMode
from ml_studio.gui.widgets.card import Card


class StatCard(Card):
    """Display a titled metric with optional semantic color."""

    def __init__(
        self,
        title: str,
        value: str = "—",
        metric_key: str = "",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent=parent)
        self._metric_key = metric_key
        self._mode = ThemeMode.LIGHT
        self._title = QLabel(title)
        self._title.setObjectName("StatTitle")
        self._value = QLabel(value)
        self._value.setObjectName("StatValue")
        self._value.setAlignment(Qt.AlignmentFlag.AlignLeft)
        value_font = QFont()
        value_font.setPointSize(22)
        value_font.setBold(True)
        self._value.setFont(value_font)
        self._hint = QLabel("")
        self._hint.setObjectName("StatDelta")
        self._hint.hide()
        self.add_widget(self._title)
        self.add_widget(self._value)
        self.add_widget(self._hint)

    def set_title(self, title: str) -> None:
        self._title.setText(title)

    def set_theme_mode(self, mode: ThemeMode) -> None:
        self._mode = mode

    def set_value(self, value: str, raw: Any = None, task: str = "") -> None:
        self._value.setText(value)
        self._hint.hide()
        if self._metric_key and raw is not None:
            color = metric_color(self._mode, self._metric_key, raw, task)
            self._value.setStyleSheet(f"color: {color};")
            if raw is None or (isinstance(raw, float) and str(raw) == "nan"):
                self._hint.setText(metric_na_reason(self._metric_key, task))
                self._hint.show()
        else:
            self._value.setStyleSheet("")
