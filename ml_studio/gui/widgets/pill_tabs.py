"""Pill-style tab selector replacing default QTabWidget headers."""

from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QHBoxLayout, QPushButton, QStackedWidget, QVBoxLayout, QWidget

from ml_studio.app.icon_provider import themed_icon
from ml_studio.app.theme import ThemeMode
from ml_studio.app.theme_tokens import SPACE


class PillTabs(QWidget):
    """Icon pill tabs with stacked content panes."""

    changed = pyqtSignal(int)

    def __init__(
        self,
        tabs: list[tuple[str, str, QWidget]],
        mode: ThemeMode = ThemeMode.LIGHT,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._mode = mode
        self._buttons: list[QPushButton] = []
        self._stack = QStackedWidget()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(SPACE[3])

        row = QHBoxLayout()
        row.setSpacing(SPACE[2])
        for index, (label, icon_name, page) in enumerate(tabs):
            btn = QPushButton(label)
            btn.setObjectName("PillTab")
            btn.setProperty("pillIndex", index)
            btn.setIcon(themed_icon(icon_name, mode.value, "text_muted"))
            btn.clicked.connect(lambda _checked, i=index: self.set_index(i))
            self._buttons.append(btn)
            row.addWidget(btn)
            self._stack.addWidget(page)
        row.addStretch()
        layout.addLayout(row)
        layout.addWidget(self._stack, 1)
        self.set_index(0)

    def set_theme_mode(self, mode: ThemeMode) -> None:
        self._mode = mode
        for btn in self._buttons:
            icon_name = btn.text().lower().replace(" ", "-")
            btn.setIcon(themed_icon(icon_name if icon_name != "quality-issues" else "missing", mode.value, "text_muted"))

    def set_index(self, index: int) -> None:
        self._stack.setCurrentIndex(index)
        for i, btn in enumerate(self._buttons):
            btn.setProperty("active", "true" if i == index else "false")
            btn.style().unpolish(btn)
            btn.style().polish(btn)
        self.changed.emit(index)

    def widget(self, index: int) -> QWidget:
        return self._stack.widget(index)
