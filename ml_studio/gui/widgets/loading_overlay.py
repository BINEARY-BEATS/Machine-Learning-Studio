"""Full-area loading overlay with scrim and message."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget


class LoadingOverlay(QWidget):
    """Semi-transparent overlay blocking interaction while work runs."""

    def __init__(self, message: str = "Loading…", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("LoadingOverlay")
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label = QLabel(message)
        self._label.setObjectName("LoadingMessage")
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._label)
        self.hide()

    def set_message(self, message: str) -> None:
        self._label.setText(message)

    def resize_to_parent(self) -> None:
        if self.parentWidget():
            self.setGeometry(self.parentWidget().rect())

    def show_overlay(self) -> None:
        self.resize_to_parent()
        if self.width() <= 0 or self.height() <= 0:
            self.setMinimumSize(200, 120)
        self.setVisible(True)
        self.raise_()

    def hide_overlay(self) -> None:
        self.hide()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.resize_to_parent()
