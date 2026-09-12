"""Visible progress panel for long-running background tasks."""

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFrame,
    QLabel,
    QListWidget,
    QProgressBar,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QPushButton,
)
from PyQt6.QtCore import pyqtSignal


class TaskProgressPanel(QWidget):
    """Centered card showing real-time task progress — not a blank black screen."""
    cancel_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet("TaskProgressPanel { background: rgba(15, 17, 23, 0.55); }")

        outer = QVBoxLayout(self)
        outer.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._card = QFrame()
        self._card.setObjectName("Card")
        self._card.setMinimumWidth(480)
        self._card.setMaximumWidth(560)
        card_layout = QVBoxLayout(self._card)
        card_layout.setSpacing(12)
        card_layout.setContentsMargins(24, 24, 24, 24)

        self._title = QLabel("Working…")
        title_font = self._title.font()
        title_font.setPointSize(16)
        title_font.setBold(True)
        self._title.setFont(title_font)

        self._subtitle = QLabel("")
        self._subtitle.setWordWrap(True)
        self._subtitle.setObjectName("MutedText")

        self._step = QLabel("Starting…")
        self._step.setWordWrap(True)

        self._bar = QProgressBar()
        self._bar.setRange(0, 100)
        self._bar.setValue(0)
        self._bar.setTextVisible(True)
        self._bar.setFormat("%p%")

        log_label = QLabel("Activity log")
        log_label.setObjectName("MutedText")
        self._log = QListWidget()
        self._log.setMaximumHeight(140)
        self._log.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        card_layout.addWidget(self._title)
        card_layout.addWidget(self._subtitle)
        card_layout.addWidget(self._step)
        card_layout.addWidget(self._bar)
        card_layout.addWidget(log_label)
        card_layout.addWidget(self._log)
        
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setObjectName("SecondaryButton")
        self._cancel_btn.clicked.connect(self._on_cancel)
        card_layout.addWidget(self._cancel_btn)

        outer.addWidget(self._card)
        self.hide()

    def _on_cancel(self):
        self._title.setText("Cancelling...")
        self._cancel_btn.setEnabled(False)
        self.cancel_requested.emit()

    def begin(self, title: str, subtitle: str = "") -> None:
        self._title.setText(title)
        self._subtitle.setText(subtitle)
        self._step.setText("Initializing…")
        self._bar.setValue(0)
        self._log.clear()
        self._cancel_btn.setEnabled(True)
        if self.parentWidget():
            self.setGeometry(self.parentWidget().rect())
        self.show()
        self.raise_()

    def update(self, percent: int, message: str) -> None:
        self._bar.setValue(max(0, min(100, percent)))
        self._step.setText(message)
        self._log.addItem(message)
        self._log.scrollToBottom()

    def end(self) -> None:
        self.hide()
