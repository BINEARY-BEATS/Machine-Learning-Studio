"""Empty state panel with themed icon and optional action slot."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget

from ml_studio.app.icon_provider import load_pixmap
from ml_studio.app.theme import ThemeMode, color_token
from ml_studio.app.theme_tokens import SPACE


class EmptyState(QWidget):
    """Centered empty-state message with optional icon and action widget."""

    def __init__(
        self,
        title: str,
        description: str = "",
        icon_name: str = "missing",
        mode: ThemeMode = ThemeMode.LIGHT,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("EmptyState")
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(SPACE[3])

        icon_label = QLabel()
        icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pixmap = load_pixmap(icon_name, 64)
        tinted = pixmap.copy()
        icon_label.setPixmap(tinted)
        layout.addWidget(icon_label)

        title_label = QLabel(title)
        title_label.setObjectName("EmptyStateTitle")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)

        self._description = QLabel(description)
        self._description.setObjectName("EmptyStateDescription")
        self._description.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._description.setWordWrap(True)
        layout.addWidget(self._description)

        self._action_slot = QVBoxLayout()
        self._action_slot.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addLayout(self._action_slot)
        self._mode = mode
        self._icon_name = icon_name
        self._icon_label = icon_label
        self._apply_icon_tint()

    def _apply_icon_tint(self) -> None:
        from ml_studio.app.icon_provider import tinted_icon

        icon = tinted_icon(self._icon_name, color_token(self._mode, "text_muted"), 64)
        self._icon_label.setPixmap(icon.pixmap(64, 64))

    def set_theme_mode(self, mode: ThemeMode) -> None:
        self._mode = mode
        self._apply_icon_tint()

    def set_description(self, text: str) -> None:
        self._description.setText(text)

    def set_action(self, widget: QWidget) -> None:
        while self._action_slot.count():
            item = self._action_slot.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._action_slot.addWidget(widget)
