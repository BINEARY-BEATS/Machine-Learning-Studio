"""Base page with container access."""

from PyQt6.QtWidgets import QVBoxLayout, QWidget

from ml_studio.app.container import AppContainer


class BasePage(QWidget):
    def __init__(self, container: AppContainer, parent=None):
        super().__init__(parent)
        self.container = container
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(24, 24, 24, 24)
        self._layout.setSpacing(16)
        self.setObjectName("ContentArea")
        self._build_ui()

    def _build_ui(self) -> None:
        raise NotImplementedError

    def on_show(self) -> None:
        """Called when page becomes visible."""
        pass
