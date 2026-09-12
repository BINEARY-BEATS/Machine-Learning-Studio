"""Icon button with label, tooltip, and full interaction states via QSS."""

from __future__ import annotations

from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QWidget

from ml_studio.app.theme_tokens import SPACE


class IconButton(QWidget):
    """Composite button: optional icon + text label."""

    def __init__(
        self,
        label: str = "",
        icon: QIcon | None = None,
        tooltip: str = "",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._button = QPushButton(label, self)
        self._button.setObjectName("IconButton")
        if icon:
            self._button.setIcon(icon)
            self._button.setIconSize(QSize(SPACE[4], SPACE[4]))
        if tooltip:
            self._button.setToolTip(tooltip)
        elif label:
            self._button.setToolTip(label)
        self._button.setCursor(Qt.CursorShape.PointingHandCursor)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._button)

    def button(self) -> QPushButton:
        return self._button

    def set_enabled(self, enabled: bool) -> None:
        self._button.setEnabled(enabled)

    def set_icon(self, icon: QIcon) -> None:
        self._button.setIcon(icon)

    def set_label(self, text: str) -> None:
        self._button.setText(text)
