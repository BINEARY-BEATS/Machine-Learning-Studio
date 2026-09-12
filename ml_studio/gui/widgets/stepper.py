"""Workflow stepper with clickable step indicators."""

from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QHBoxLayout, QPushButton, QWidget

from ml_studio.app.theme_tokens import SPACE


class Stepper(QWidget):
    """Horizontal stepper showing progress through a multi-step workflow."""

    step_clicked = pyqtSignal(int)

    def __init__(self, steps: list[str], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._steps = steps
        self._current = 0
        layout = QHBoxLayout(self)
        layout.setSpacing(SPACE[2])
        self._buttons: list[QPushButton] = []
        for index, step in enumerate(steps):
            btn = QPushButton(f"{index + 1}. {step}")
            btn.setObjectName("StepButton")
            btn.clicked.connect(lambda _checked, i=index: self.step_clicked.emit(i))
            self._buttons.append(btn)
            layout.addWidget(btn)
        layout.addStretch()
        self.set_current(0)

    def set_current(self, index: int) -> None:
        self._current = max(0, min(index, len(self._steps) - 1))
        for i, btn in enumerate(self._buttons):
            state = "done" if i < self._current else ("active" if i == self._current else "pending")
            btn.setProperty("stepState", state)
            btn.style().unpolish(btn)
            btn.style().polish(btn)

    def current_index(self) -> int:
        return self._current

    def step_count(self) -> int:
        return len(self._steps)
