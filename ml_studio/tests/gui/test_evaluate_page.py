import pytest
from unittest.mock import MagicMock
from ml_studio.gui.pages.evaluate_page import EvaluatePage
from ml_studio.app.theme import ThemeMode
from ml_studio.app.container import AppContainer
from ml_studio.app.config import AppConfig

@pytest.fixture
def container():
    return AppContainer(AppConfig())

def test_evaluate_page_init(container, qtbot):
    page = EvaluatePage(container)
    qtbot.addWidget(page)
    assert page._metrics_table is not None


def test_evaluate_page_run_count_and_metric_key(container, qtbot):
    from ml_studio.gui.pages.evaluate_page import ExperimentRun

    page = EvaluatePage(container)
    qtbot.addWidget(page)
    assert page.run_count() == 0
    page._runs.append(
        ExperimentRun(
            run_id="1",
            model_name="m",
            model_id="m",
            task="CLASSIFICATION",
            dataset_name="d",
            target="y",
            metrics={"f1": 0.9, "accuracy": 0.91},
            cv_mean=None,
            cv_std=None,
            duration_sec=1.0,
            train_rows=10,
            test_rows=5,
        )
    )
    page._show_run_detail(page._runs[0])
    assert page._primary_card._metric_key == "f1"
    assert page.run_count() == 1

def test_evaluate_page_add_run(container, qtbot):
    page = EvaluatePage(container)
    qtbot.addWidget(page)
    
    result_mock = MagicMock()
    result_mock.experiment_id = "run-1"
    result_mock.model_id = "rf"
    result_mock.task.value = "classification"
    result_mock.target_column = "target"
    result_mock.metrics = {"accuracy": 0.95}
    result_mock.cv_scores = {"mean": 0.94, "std": 0.01}
    result_mock.training_duration = 10.0
    result_mock.train_size = 1000
    result_mock.test_size = 200
    
    page.add_run(result_mock, model_display_name="Random Forest", dataset_name="data")
    assert page._leaderboard.rowCount() == 1
    assert page._count_label.text() == "1 experiment"

def test_evaluate_page_show_metrics(container, qtbot):
    page = EvaluatePage(container)
    qtbot.addWidget(page)
    
    result_mock = MagicMock()
    result_mock.experiment_id = "run-1"
    result_mock.model_id = "rf"
    result_mock.task.value = "classification"
    result_mock.target_column = "target"
    result_mock.metrics = {"accuracy": 0.95, "f1_score": 0.94, "mse": 0.1, "r2_score": 0.99, "silhouette": 0.8}
    result_mock.cv_scores = None
    result_mock.training_duration = 10.0
    result_mock.train_size = 1000
    result_mock.test_size = 200
    
    page.add_run(result_mock, model_display_name="Random Forest", dataset_name="data")
    assert page._metrics_table.rowCount() == 5

def test_evaluate_page_set_theme_mode(container, qtbot):
    page = EvaluatePage(container)
    qtbot.addWidget(page)
    page.set_theme_mode(ThemeMode.DARK)
    assert page._mode == ThemeMode.DARK

def test_evaluate_page_on_selection_changed(container, qtbot):
    page = EvaluatePage(container)
    qtbot.addWidget(page)
    
    result_mock = MagicMock()
    result_mock.experiment_id = "run-1"
    result_mock.model_id = "rf"
    result_mock.task.value = "classification"
    result_mock.target_column = "target"
    result_mock.metrics = {"accuracy": 0.95}
    result_mock.cv_scores = None
    result_mock.training_duration = 10.0
    result_mock.train_size = 1000
    result_mock.test_size = 200
    
    page.add_run(result_mock, model_display_name="Random Forest", dataset_name="data")
    
    # Selection change should trigger update details
    page._leaderboard.selectRow(0)
    page._leaderboard.itemSelectionChanged.emit()
    assert page._metrics_table.rowCount() == 1
