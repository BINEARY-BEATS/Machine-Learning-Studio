"""B3 acceptance: schema-driven Predict page + batch preflight."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest
from PyQt6.QtWidgets import QComboBox, QLineEdit, QProgressBar

from ml_studio.app.config import AppConfig
from ml_studio.app.container import AppContainer
from ml_studio.core.training.task import TaskType
from ml_studio.gui.pages.predict_page import PredictPage
from ml_studio.gui.workers.batch_predict_worker import BatchPredictWorker
from ml_studio.gui.widgets.schema_form import MISSING, collect_values


@pytest.fixture
def container():
    return AppContainer(AppConfig())


def _schema_predictor(classes=("no", "yes")):
    pipe = MagicMock()
    pipe.feature_schema = {
        "city": {
            "kind": "categorical",
            "categories": ["Lahore", "Karachi"],
            "nullable": True,
            "ref": "Lahore",
        },
        "age": {
            "kind": "numeric",
            "min": 18.0,
            "max": 90.0,
            "ref": 35.0,
            "nullable": True,
        },
    }
    pipe.input_feature_columns = ["city", "age"]
    pipe.target_classes = list(classes)

    def coerce_row(row):
        from ml_studio.core.persistence.serializer import InferencePipeline
        from ml_studio.core.training.task import TaskType as TT

        helper = InferencePipeline(
            estimator=None,
            preprocessing=None,
            feature_columns=["city", "age"],
            target_column="churn",
            task=TT.CLASSIFICATION,
            feature_schema=pipe.feature_schema,
            target_classes=list(classes),
            input_feature_columns=["city", "age"],
        )
        return helper.coerce_row(row)

    pipe.coerce_row.side_effect = coerce_row
    predictor = MagicMock()
    predictor.pipeline = pipe
    predictor.predict_single.return_value = MagicMock(
        prediction="yes",
        probabilities={"no": 0.2, "yes": 0.8},
        explanation=None,
    )
    return predictor


def test_categorical_combobox_populated_from_schema(qtbot, container):
    page = PredictPage(container)
    qtbot.addWidget(page)
    page.bind_predictor(_schema_predictor(), ["city", "age"], TaskType.CLASSIFICATION)
    city = page._inputs["city"]
    assert isinstance(city, QComboBox)
    items = [city.itemText(i) for i in range(city.count())]
    assert MISSING in items
    assert "Lahore" in items
    assert "Karachi" in items
    age = page._inputs["age"]
    assert isinstance(age, QLineEdit)
    assert "18" in age.placeholderText() or "ref" in age.placeholderText()


def test_invalid_numeric_field_flagged(qtbot, container):
    page = PredictPage(container)
    qtbot.addWidget(page)
    page.bind_predictor(_schema_predictor(), ["city", "age"], TaskType.CLASSIFICATION)
    page._inputs["city"].setCurrentText("Lahore")
    page._inputs["age"].setText("thirty")
    page._run_single()
    assert page._inputs["age"].property("error") is True
    assert "age" in page._validation_banner.text()
    page._predictor.predict_single.assert_not_called()


def test_predict_returns_decoded_label_and_proba_bars(qtbot, container):
    page = PredictPage(container)
    qtbot.addWidget(page)
    page.bind_predictor(_schema_predictor(), ["city", "age"], TaskType.CLASSIFICATION)
    page._inputs["city"].setCurrentText("Lahore")
    page._inputs["age"].setText("31")
    page._run_single()
    assert "Prediction: yes" in page._result_label.text()
    bars = page.findChildren(QProgressBar)
    assert len(bars) >= 2
    assert bars[0].objectName() == "ClassProbBar"


def test_batch_preflight_lists_missing_columns(tmp_path: Path):
    csv = tmp_path / "batch.csv"
    pd.DataFrame({"age": [1, 2]}).to_csv(csv, index=False)
    missing, extra = BatchPredictWorker.preflight(csv, ["city", "age"])
    assert missing == ["city"]
    assert "age" not in missing
    assert extra == []


def test_batch_preflight_reports_extra(tmp_path: Path):
    csv = tmp_path / "batch.csv"
    pd.DataFrame({"city": ["Lahore"], "age": [30], "noise": [1]}).to_csv(csv, index=False)
    missing, extra = BatchPredictWorker.preflight(csv, ["city", "age"])
    assert missing == []
    assert "noise" in extra


def test_fallback_lineedits_without_schema(qtbot, container):
    page = PredictPage(container)
    qtbot.addWidget(page)
    predictor = MagicMock()
    predictor.pipeline.feature_schema = {}
    predictor.predict_single.return_value = MagicMock(
        prediction=1, probabilities=None, explanation=None
    )
    page.bind_predictor(predictor, ["Feature1", "Feature2"], TaskType.CLASSIFICATION)
    assert isinstance(page._inputs["Feature1"], QLineEdit)
    page._inputs["Feature1"].setText("1")
    page._inputs["Feature2"].setText("2")
    page._run_single()
    predictor.predict_single.assert_called_once()
    vals = collect_values(page._inputs)
    assert vals["Feature1"] == "1"
