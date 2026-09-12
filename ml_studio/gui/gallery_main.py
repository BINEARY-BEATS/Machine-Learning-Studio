"""Standalone widget gallery for Slice 1 design system review."""

from __future__ import annotations

import sys

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ml_studio.app.icon_provider import themed_icon
from ml_studio.app.theme import ThemeMode, apply_theme, write_theme_files
from ml_studio.app.theme_tokens import SPACE
from ml_studio.gui.pages.theme_preview import ThemePreviewPage
from ml_studio.gui.widgets.card import Card
from ml_studio.gui.widgets.empty_state import EmptyState
from ml_studio.gui.widgets.icon_button import IconButton
from ml_studio.gui.widgets.loading_overlay import LoadingOverlay
from ml_studio.gui.widgets.search_bar import SearchBar
from ml_studio.gui.widgets.stat_card import StatCard
from ml_studio.gui.widgets.tag_chip import TagChip
from ml_studio.gui.widgets.toast import Toast


class WidgetGalleryWindow(QMainWindow):
    """Dev window showcasing Slice 1 widgets in light/dark themes."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("ML Studio — Design System Gallery (Slice 1)")
        self.resize(1100, 760)
        self._mode = ThemeMode.LIGHT
        self._toast: Toast | None = None
        self._overlay: LoadingOverlay | None = None

        root = QWidget()
        root.setObjectName("GalleryRoot")
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(SPACE[5], SPACE[5], SPACE[5], SPACE[5])
        outer.setSpacing(SPACE[4])

        header = QHBoxLayout()
        title = QLabel("Widget Gallery")
        title.setObjectName("PageTitle")
        header.addWidget(title)
        header.addStretch()
        self._theme_btn = QPushButton("Switch to Dark")
        self._theme_btn.setObjectName("GhostButton")
        self._theme_btn.clicked.connect(self._toggle_theme)
        header.addWidget(self._theme_btn)
        outer.addLayout(header)

        self._tabs = QTabWidget()
        self._tabs.addTab(self._build_tokens_tab(), "Tokens")
        self._tabs.addTab(self._build_widgets_tab(), "Widgets")
        self._tabs.addTab(self._build_states_tab(), "States")
        outer.addWidget(self._tabs)

        self._overlay = LoadingOverlay(parent=root)
        self._toast = Toast(parent=root)

    def _build_tokens_tab(self) -> QWidget:
        self._token_page = ThemePreviewPage(self._mode)
        return self._token_page

    def _build_widgets_tab(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        body = QWidget()
        layout = QVBoxLayout(body)
        layout.setSpacing(SPACE[5])

        cards_row = QHBoxLayout()
        cards_row.addWidget(StatCard("Rows", "12,480"))
        r2 = StatCard("R²", "-0.0002", metric_key="r2")
        r2.set_theme_mode(self._mode)
        r2.set_value("-0.0002", raw=-0.0002)
        cards_row.addWidget(r2)
        cards_row.addWidget(StatCard("Accuracy", "0.91", metric_key="accuracy"))
        layout.addLayout(cards_row)

        chips = QHBoxLayout()
        for variant in TagChip.VARIANTS:
            chips.addWidget(TagChip(variant.capitalize(), variant))
        chips.addStretch()
        layout.addLayout(chips)

        search = SearchBar("Filter columns…")
        layout.addWidget(search)

        icon_row = QHBoxLayout()
        for name, label in (("import", "Import"), ("train", "Train"), ("copy", "Copy")):
            icon = themed_icon(name, self._mode.value, "text_muted")
            icon_row.addWidget(IconButton(label, icon, tooltip=f"{label} action"))
        icon_row.addStretch()
        layout.addLayout(icon_row)

        card = Card("Card Container")
        card.add_widget(QLabel("Cards provide consistent surface, border, and padding."))
        layout.addWidget(card)
        layout.addStretch()
        scroll.setWidget(body)
        return scroll

    def _build_states_tab(self) -> QWidget:
        body = QWidget()
        layout = QVBoxLayout(body)
        layout.setSpacing(SPACE[5])

        empty = EmptyState(
            "No dataset loaded",
            "Import a CSV, Excel, or Parquet file to begin exploring your data.",
            icon_name="import",
            mode=self._mode,
        )
        layout.addWidget(empty)

        actions = QHBoxLayout()
        overlay_btn = QPushButton("Show Loading Overlay")
        overlay_btn.setObjectName("PrimaryButton")
        overlay_btn.clicked.connect(self._show_overlay_demo)
        actions.addWidget(overlay_btn)

        toast_btn = QPushButton("Show Success Toast")
        toast_btn.setObjectName("GhostButton")
        toast_btn.clicked.connect(lambda: self._show_toast("Pipeline saved.", "success"))
        actions.addWidget(toast_btn)
        actions.addStretch()
        layout.addLayout(actions)
        layout.addStretch()
        return body

    def _toggle_theme(self) -> None:
        self._mode = ThemeMode.DARK if self._mode == ThemeMode.LIGHT else ThemeMode.LIGHT
        app = QApplication.instance()
        if app:
            apply_theme(app, self._mode)
        label = "Switch to Light" if self._mode == ThemeMode.DARK else "Switch to Dark"
        self._theme_btn.setText(label)
        self._token_page.set_mode(self._mode)

    def _show_overlay_demo(self) -> None:
        if self._overlay:
            self._overlay.set_message("Processing…")
            self._overlay.show_overlay()
            QApplication.processEvents()
            QApplication.instance().processEvents() if QApplication.instance() else None
            from PyQt6.QtCore import QTimer

            QTimer.singleShot(1200, self._overlay.hide_overlay)

    def _show_toast(self, message: str, variant: str) -> None:
        if self._toast:
            self._toast.show_message(message, variant=variant)


def main() -> int:
    write_theme_files()
    app = QApplication(sys.argv)
    app.setApplicationName("ML Studio Gallery")
    apply_theme(app, ThemeMode.LIGHT)
    window = WidgetGalleryWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
