"""Widget instantiation and behavior tests."""

from __future__ import annotations

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication, QPushButton

from ml_studio.app.theme import ThemeMode
from ml_studio.gui.widgets.card import Card
from ml_studio.gui.widgets.empty_state import EmptyState
from ml_studio.gui.widgets.icon_button import IconButton
from ml_studio.gui.widgets.loading_overlay import LoadingOverlay
from ml_studio.gui.widgets.search_bar import SearchBar
from ml_studio.gui.widgets.stat_card import StatCard
from ml_studio.gui.widgets.tag_chip import TagChip
from ml_studio.gui.widgets.toast import Toast


def test_card_object_name(qapp):
    card = Card("Metrics")
    assert card.objectName() == "Card"


def test_stat_card_semantic_color(qapp):
    card = StatCard("R²", metric_key="r2")
    card.set_theme_mode(ThemeMode.LIGHT)
    card.set_value("-0.02", raw=-0.02)
    assert "color:" in card._value.styleSheet()


def test_search_bar_debounce(qapp):
    bar = SearchBar(debounce_ms=30)
    seen: list[str] = []
    bar.search_changed.connect(seen.append)
    bar.setText("hello")
    QTimer.singleShot(80, qapp.quit)
    qapp.exec()
    assert seen and seen[-1] == "hello"


def test_tag_chip_variant_property(qapp):
    chip = TagChip("OK", "success")
    assert chip.property("chipVariant") == "success"


def test_empty_state_action_slot(qapp):
    state = EmptyState("Title", "Desc")
    btn = QPushButton("Import")
    state.set_action(btn)
    assert state.findChild(QPushButton) is btn


def test_loading_overlay_message(qapp):
    parent = Card()
    parent.resize(400, 300)
    parent.show()
    overlay = LoadingOverlay(parent=parent)
    overlay.set_message("Working")
    overlay.show_overlay()
    qapp.processEvents()
    assert overlay.isVisible()
    overlay.hide_overlay()


def test_toast_variant(qapp):
    parent = Card()
    toast = Toast(parent=parent)
    toast.show_message("Done", duration_ms=100, variant="success")
    assert toast._label.property("toastVariant") == "success"


def test_icon_button_tooltip(qapp):
    btn = IconButton("Save", tooltip="Save project")
    assert btn.button().toolTip() == "Save project"
