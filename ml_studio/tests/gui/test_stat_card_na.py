"""StatCard N/A coloring regression."""

from ml_studio.app.theme import ThemeMode
from ml_studio.gui.widgets.stat_card import StatCard


def test_stat_card_na_shows_hint(qtbot):
    card = StatCard("Score", metric_key="silhouette")
    card.set_theme_mode(ThemeMode.LIGHT)
    qtbot.addWidget(card)
    card.show()
    card.set_value("—", raw=None, task="CLUSTERING")
    assert not card._hint.isHidden()
    assert card._hint.text()
    assert "color:" in card._value.styleSheet()
