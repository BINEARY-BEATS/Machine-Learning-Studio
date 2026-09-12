import sys
import pytest
from PyQt6.QtWidgets import QApplication
from ml_studio.gui.pages.train_page import TrainPage
from ml_studio.app.container import AppContainer
from ml_studio.app.config import AppConfig
from ml_studio.core.training.task import TaskType

app = QApplication.instance() or QApplication(sys.argv)

def test_train_page_summary_sync():
    container = AppContainer(AppConfig())
    page = TrainPage(container)
    
    page._task_combo.setCurrentText(TaskType.REGRESSION.value)
    
    # Must have target and features for build_config to not return None
    page._target_combo.addItem("price")
    page._target_combo.setCurrentText("price")
    
    from PyQt6.QtWidgets import QListWidgetItem
    item = QListWidgetItem("sqft")
    page._feature_list.addItem(item)
    item.setSelected(True)
    
    page._model_combo.setCurrentText('Random Forest Regressor')
    
    assert 'Random Forest Regressor' in page._summary.text()
