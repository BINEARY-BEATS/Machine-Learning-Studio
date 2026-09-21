import pytest
from ml_studio.gui.pages.theme_preview import ThemePreviewPage
from ml_studio.app.theme import ThemeMode
from ml_studio.app.container import AppContainer
from ml_studio.app.config import AppConfig

@pytest.fixture
def container():
    return AppContainer(AppConfig())

def test_theme_preview_page_init(container, qtbot):
    page = ThemePreviewPage(container)
    qtbot.addWidget(page)
    assert page is not None

def test_theme_preview_set_mode(container, qtbot):
    page = ThemePreviewPage(container)
    qtbot.addWidget(page)
    page.set_mode(ThemeMode.DARK)
    assert page._mode == ThemeMode.DARK
