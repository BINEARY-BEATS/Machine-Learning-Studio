"""Status bar widgets for context and resource usage."""

from __future__ import annotations

from PyQt6.QtWidgets import QLabel, QStatusBar, QWidget

from ml_studio.app.theme_tokens import SPACE


class AppStatusBar(QStatusBar):
    """Extended status bar with context chips."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("StatusBar")
        self._message = QLabel("Ready")
        self._context = QLabel("")
        self._context.setObjectName("StatusChip")
        self._memory = QLabel("")
        self._memory.setObjectName("StatusChip")
        self.addWidget(self._message, 1)
        self.addPermanentWidget(self._context)
        self.addPermanentWidget(self._memory)

    def set_message(self, text: str) -> None:
        self._message.setText(text)

    def set_context(self, text: str) -> None:
        self._context.setText(text)
        self._context.setVisible(bool(text))

    def set_memory(self, text: str) -> None:
        self._memory.setText(text)
        self._memory.setVisible(bool(text))

    def update_from_dataset(self, dataset) -> None:
        if dataset is None:
            self.set_context("")
            self.set_memory("")
            return
        self.set_context(f"Dataset: {dataset.name}")
        mb = dataset.memory_bytes / (1024 * 1024)
        self.set_memory(f"Memory: {mb:.1f} MB")
