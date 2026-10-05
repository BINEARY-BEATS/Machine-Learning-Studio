"""Visible progress panel for long-running background tasks."""

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QFrame,
    QLabel,
    QListWidget,
    QProgressBar,
    QVBoxLayout,
    QWidget,
    QPushButton,
)


class TaskProgressPanel(QWidget):
    """Centered card showing real-time task progress."""

    cancel_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet("TaskProgressPanel { background: rgba(12, 16, 18, 0.72); }")
        self._elapsed_s = 0
        self._last_message = "Starting…"
        self._timer = QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self._tick_elapsed)

        outer = QVBoxLayout(self)
        outer.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._card = QFrame()
        self._card.setObjectName("Card")
        self._card.setMinimumWidth(520)
        self._card.setMaximumWidth(600)
        card_layout = QVBoxLayout(self._card)
        card_layout.setSpacing(10)
        card_layout.setContentsMargins(24, 24, 24, 24)

        self._title = QLabel("Working…")
        title_font = self._title.font()
        ps = title_font.pointSize()
        title_font.setPointSize(16 if ps < 1 else max(ps, 16))
        title_font.setBold(True)
        self._title.setFont(title_font)

        self._subtitle = QLabel("")
        self._subtitle.setWordWrap(True)
        self._subtitle.setObjectName("MutedText")

        self._step = QLabel("Starting…")
        self._step.setWordWrap(True)

        self._elapsed = QLabel("Elapsed  0:00")
        self._elapsed.setObjectName("MonoMetric")

        self._bar = QProgressBar()
        self._bar.setRange(0, 100)
        self._bar.setValue(0)
        self._bar.setTextVisible(True)
        self._bar.setFormat("%p%")
        self._bar.setMinimumHeight(18)

        log_label = QLabel("Activity")
        log_label.setObjectName("MutedText")
        self._log = QListWidget()
        self._log.setMaximumHeight(140)
        self._log.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._log.setAlternatingRowColors(True)

        card_layout.addWidget(self._title)
        card_layout.addWidget(self._subtitle)
        card_layout.addWidget(self._step)
        card_layout.addWidget(self._elapsed)
        card_layout.addWidget(self._bar)
        card_layout.addWidget(log_label)
        card_layout.addWidget(self._log)

        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setObjectName("GhostButton")
        self._cancel_btn.clicked.connect(self._on_cancel)
        card_layout.addWidget(self._cancel_btn)

        outer.addWidget(self._card)
        self.hide()

    def _format_elapsed(self) -> str:
        m, s = divmod(self._elapsed_s, 60)
        return f"Elapsed  {m}:{s:02d}"

    def _on_cancel(self):
        self._title.setText("Cancelling…")
        self._step.setText("Stopping after the current step…")
        self._cancel_btn.setEnabled(False)
        self.cancel_requested.emit()

    def _tick_elapsed(self) -> None:
        self._elapsed_s += 1
        self._elapsed.setText(self._format_elapsed())

    def begin(self, title: str, subtitle: str = "") -> None:
        self._title.setText(title)
        self._subtitle.setText(subtitle)
        self._step.setText("Initializing…")
        self._last_message = "Initializing…"
        self._bar.setValue(0)
        self._log.clear()
        self._elapsed_s = 0
        self._elapsed.setText(self._format_elapsed())
        self._cancel_btn.setEnabled(True)
        if self.parentWidget():
            self.setGeometry(self.parentWidget().rect())
        self._timer.start()
        self.show()
        self.raise_()

    def update(self, percent: int, message: str) -> None:
        self._bar.setValue(max(0, min(100, percent)))
        self._step.setText(message)
        if message != self._last_message:
            self._log.addItem(message)
            self._log.scrollToBottom()
            self._last_message = message

    def end(self) -> None:
        self._timer.stop()
        self.hide()
