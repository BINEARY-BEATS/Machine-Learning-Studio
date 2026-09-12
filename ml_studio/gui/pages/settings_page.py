"""Application settings page."""

from __future__ import annotations

from PyQt6.QtWidgets import QCheckBox, QComboBox, QFormLayout, QLabel

from ml_studio.app.theme import ThemeMode
from ml_studio.gui.pages.base_page import BasePage
from ml_studio.gui.widgets.card import Card


class SettingsPage(BasePage):
    def __init__(self, container, parent=None):
        self._theme_cb = None
        self._autosave_cb = None
        super().__init__(container, parent)

    def _build_ui(self) -> None:
        title = QLabel("Settings")
        title.setObjectName("PageTitle")
        self._layout.addWidget(title)

        card = Card("Appearance & behavior")
        form = QFormLayout()
        self._theme_cb = QComboBox()
        self._theme_cb.addItems(["Light", "Dark"])
        self._autosave_cb = QCheckBox("Autosave project on changes")
        self._autosave_cb.setChecked(True)
        form.addRow("Theme:", self._theme_cb)
        form.addRow("", self._autosave_cb)
        card.add_layout(form)
        self._layout.addWidget(card)
        self._layout.addStretch()

    def selected_theme(self) -> ThemeMode:
        return ThemeMode.DARK if self._theme_cb.currentText() == "Dark" else ThemeMode.LIGHT

    def set_theme_selection(self, mode: ThemeMode) -> None:
        self._theme_cb.setCurrentText("Dark" if mode == ThemeMode.DARK else "Light")

    def autosave_enabled(self) -> bool:
        return self._autosave_cb.isChecked()
