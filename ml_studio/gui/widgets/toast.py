"""Non-blocking toast notifications anchored to a parent window."""

from __future__ import annotations

from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget

from ml_studio.app.theme_tokens import MOTION, SPACE


class Toast(QWidget):
    """Ephemeral message bubble shown above a parent widget."""

    VARIANTS = ("default", "success", "warning", "danger")

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        flags = Qt.WindowType.SubWindow | Qt.WindowType.FramelessWindowHint
        self.setWindowFlags(flags)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self._label = QLabel()
        self._label.setObjectName("ToastLabel")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._label)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self.hide)

    def show_message(
        self,
        message: str,
        duration_ms: int | None = None,
        variant: str = "default",
    ) -> None:
        safe = variant if variant in self.VARIANTS else "default"
        self._label.setText(message)
        self._label.setProperty("toastVariant", safe)
        self._label.style().unpolish(self._label)
        self._label.style().polish(self._label)
        self.adjustSize()
        self._position()
        self.show()
        self.raise_()
        ms = duration_ms if duration_ms is not None else MOTION["toast"]
        self._timer.start(ms)

    def _position(self) -> None:
        parent = self.parentWidget()
        if not parent:
            return
        margin = SPACE[5]
        x = parent.width() - self.width() - margin
        y = parent.height() - self.height() - margin
        self.move(max(margin, x), max(margin, y))
