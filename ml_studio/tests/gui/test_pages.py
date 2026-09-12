import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication
from ml_studio.app.config import AppConfig
from ml_studio.app.container import AppContainer
from ml_studio.gui.pages.home import HomePage
from ml_studio.gui.pages.data_page import DataPage
from ml_studio.gui.pages.settings_page import SettingsPage
from ml_studio.gui.pages.prepare_page import PreparePage
from ml_studio.gui.pages.train_page import TrainPage
from ml_studio.gui.pages.evaluate_page import EvaluatePage
from ml_studio.gui.pages.predict_page import PredictPage
from ml_studio.gui.pages.models_page import ModelsPage

@pytest.fixture
def container():
    config = AppConfig()
    return AppContainer(config=config)

def test_home_page_init(qtbot, container):
    page = HomePage(container)
    qtbot.addWidget(page)
    assert page is not None
    assert page.layout() is not None
    # Test refresh button
    page._new_btn.click()
    page._open_btn.click()
    page._import_btn.click()

def test_data_page_init(qtbot, container):
    page = DataPage(container)
    qtbot.addWidget(page)
    assert page is not None
    assert page.layout() is not None
    page._import_btn.click()
    page._optimize_btn.click()
    page._profile_btn.click()

def test_settings_page_init(qtbot, container):
    page = SettingsPage(container)
    qtbot.addWidget(page)
    assert page is not None
    assert page.layout() is not None

def test_prepare_page_init(qtbot, container):
    page = PreparePage(container)
    qtbot.addWidget(page)
    assert page is not None
    assert page.layout() is not None

def test_train_page_init(qtbot, container):
    page = TrainPage(container)
    qtbot.addWidget(page)
    assert page is not None
    assert page.layout() is not None

def test_evaluate_page_init(qtbot, container):
    page = EvaluatePage(container)
    qtbot.addWidget(page)
    assert page is not None
    assert page.layout() is not None

def test_predict_page_init(qtbot, container):
    page = PredictPage(container)
    qtbot.addWidget(page)
    assert page is not None
    assert page.layout() is not None

def test_models_page_init(qtbot, container):
    page = ModelsPage(container)
    qtbot.addWidget(page)
    assert page is not None
    assert page.layout() is not None
