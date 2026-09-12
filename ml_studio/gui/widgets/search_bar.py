"""Debounced search input with clear affordance."""

from __future__ import annotations

from PyQt6.QtCore import QTimer, pyqtSignal
from PyQt6.QtWidgets import QLineEdit, QWidget


class SearchBar(QLineEdit):
    """Search field emitting debounced search_changed signal."""

    search_changed = pyqtSignal(str)

    def __init__(
        self,
        placeholder: str = "Search…",
        debounce_ms: int = 200,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("SearchBar")
        self.setPlaceholderText(placeholder)
        self.setClearButtonEnabled(True)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(debounce_ms)
        self._timer.timeout.connect(self._emit_search)
        self.textChanged.connect(self._on_text_changed)

    def _on_text_changed(self, _text: str) -> None:
        self._timer.start()

    def _emit_search(self) -> None:
        self.search_changed.emit(self.text())
