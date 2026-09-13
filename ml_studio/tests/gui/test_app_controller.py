import pytest
from ml_studio.gui.app_controller import AppController
from ml_studio.app.config import AppConfig
from ml_studio.app.container import AppContainer

@pytest.fixture
def container():
    config = AppConfig()
    return AppContainer(config=config)

def test_app_controller_initial_state(container):
    controller = AppController(container)
    assert controller.container.project_manager.current is None
    assert controller.container is container

def test_app_controller_new_project(container):
    controller = AppController(container)
    controller.new_project()
    assert controller.container.project_manager.current is not None
    assert controller.container.project_manager.current.metadata.name == "Untitled Project"
