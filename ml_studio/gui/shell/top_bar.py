"""Top application bar with search, breadcrumb, and theme controls."""

from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QLabel, QLineEdit, QPushButton, QWidget

from ml_studio.app.icon_provider import themed_icon
from ml_studio.app.theme import ThemeMode
from ml_studio.app.theme_tokens import SPACE
from ml_studio.gui.widgets.search_bar import SearchBar


class TopBar(QFrame):
    """Header bar: palette trigger, breadcrumb, project name, theme toggle."""

    command_palette_requested = pyqtSignal()
    theme_toggle_requested = pyqtSignal()
    project_renamed = pyqtSignal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("TopBar")
        self._mode = ThemeMode.LIGHT
        layout = QHBoxLayout(self)
        layout.setContentsMargins(SPACE[4], SPACE[2], SPACE[4], SPACE[2])
        layout.setSpacing(SPACE[3])

        self._search = SearchBar("Search commands…  Ctrl+K")
        self._search.setReadOnly(True)
        self._search.setMaximumWidth(400)
        self._search.mousePressEvent = lambda _e: self.command_palette_requested.emit()  # type: ignore[method-assign]
        layout.addWidget(self._search, 0)

        self._breadcrumb = QLabel("ML Studio")
        self._breadcrumb.setObjectName("TextMuted")
        layout.addWidget(self._breadcrumb, 2)

        self._project = QLineEdit("Untitled Project")
        self._project.setObjectName("ProjectName")
        self._project.setToolTip("Project name")
        self._project.setMaximumWidth(280)
        self._project.editingFinished.connect(self._emit_rename)
        layout.addWidget(self._project, 1)

        self._theme_btn = QPushButton()
        self._theme_btn.setObjectName("IconButton")
        self._theme_btn.setToolTip("Toggle light/dark theme")
        self._theme_btn.clicked.connect(self.theme_toggle_requested.emit)
        layout.addWidget(self._theme_btn)
        self.set_theme_mode(ThemeMode.LIGHT)

    def set_theme_mode(self, mode: ThemeMode) -> None:
        self._mode = mode
        icon = themed_icon("dark-mode", mode.value, "text_muted")
        self._theme_btn.setIcon(icon)
        label = "Switch to light theme" if mode == ThemeMode.DARK else "Switch to dark theme"
        self._theme_btn.setToolTip(label)

    def set_breadcrumb(self, parts: list[str]) -> None:
        clean = [p for p in parts if p]
        self._breadcrumb.setText("  ›  ".join(clean) if clean else "ML Studio")

    def set_project_name(self, name: str) -> None:
        self._project.blockSignals(True)
        self._project.setText(name)
        self._project.blockSignals(False)

    def _emit_rename(self) -> None:
        self.project_renamed.emit(self._project.text().strip())
