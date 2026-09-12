"""Reusable card container with optional header."""

from __future__ import annotations

from PyQt6.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from ml_studio.app.theme_tokens import SPACE


class Card(QFrame):
    """Surface card with consistent padding and optional title row."""

    def __init__(self, title: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("Card")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(SPACE[4], SPACE[4], SPACE[4], SPACE[4])
        outer.setSpacing(SPACE[2])
        self._body = QVBoxLayout()
        self._body.setSpacing(SPACE[2])
        if title:
            header = QLabel(title)
            header.setObjectName("CardHeader")
            outer.addWidget(header)
        outer.addLayout(self._body)

    def add_widget(self, widget: QWidget) -> None:
        self._body.addWidget(widget)

    def add_layout(self, layout: QHBoxLayout | QVBoxLayout) -> None:
        self._body.addLayout(layout)

    def add_stretch(self) -> None:
        self._body.addStretch()
