"""Prediction page with single, batch, and explain tabs."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ml_studio.app.theme import ThemeMode
from ml_studio.gui.layout_utils import constrain_primary_button
from ml_studio.gui.pages.base_page import BasePage
from ml_studio.gui.pages.predict_tabs import (
    build_drift_tab,
    build_explain_tab,
    pick_drift_file,
    run_drift,
    run_importance,
)
from ml_studio.gui.widgets.card import Card
from ml_studio.gui.widgets.empty_state import EmptyState
from ml_studio.gui.widgets.pill_tabs import PillTabs
from ml_studio.gui.widgets.schema_form import (
    build_feature_row,
    clear_field_errors,
    clear_layout,
    collect_values,
    mark_field_errors,
    render_proba_bars,
    schema_or_empty,
)
from ml_studio.gui.workers.batch_predict_worker import BatchPredictWorker


class PredictPage(BasePage):
    batch_predict_requested = pyqtSignal(str, str)

    def __init__(self, container, parent=None):
        self._predictor = None
        self._features: list[str] = []
        self._task = None
        self._mode = ThemeMode.LIGHT
        self._inputs: dict[str, QWidget] = {}
        self._error_labels: dict[str, QLabel] = {}
        super().__init__(container, parent)

    def _build_ui(self) -> None:
        title = QLabel("Predict")
        title.setObjectName("PageTitle")
        self._layout.addWidget(title)
        self._tabs = PillTabs(
            [
                ("Single", "ai", self._build_single_tab()),
                ("Batch", "import", self._build_batch_tab()),
                ("Explain", "chart", build_explain_tab(self)),
                ("Drift", "scale", build_drift_tab(self)),
            ]
        )
        self._layout.addWidget(self._tabs, 1)
        self._empty = EmptyState(
            "No model loaded",
            "Train and register a model to run predictions.",
            icon_name="ai",
        )
        go_train = QPushButton("Go to Train")
        go_train.setObjectName("PrimaryButton")
        constrain_primary_button(go_train)
        go_train.clicked.connect(self._goto_train)
        self._empty.set_action(go_train)
        self._layout.addWidget(self._empty, 1)

    def set_theme_mode(self, mode: ThemeMode) -> None:
        self._mode = mode
        self._empty.set_theme_mode(mode)
        self._tabs.set_theme_mode(mode)

    def _goto_train(self) -> None:
        win = self.window()
        if hasattr(win, "_navigate"):
            win._navigate("train")

    def _build_single_tab(self) -> QWidget:
        box = QWidget()
        layout = QVBoxLayout(box)
        card = Card("Single prediction")
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        form_host = QWidget()
        self._inputs_layout = QVBoxLayout(form_host)
        self._inputs_layout.setContentsMargins(0, 0, 0, 0)
        scroll.setWidget(form_host)
        card.add_widget(scroll, stretch=1)
        self._explain_check = QCheckBox("Include local explanation (SHAP if available)")
        card.add_widget(self._explain_check)
        self._predict_btn = QPushButton("Predict")
        self._predict_btn.setObjectName("PrimaryButton")
        constrain_primary_button(self._predict_btn)
        self._predict_btn.clicked.connect(self._run_single)
        self._validation_banner = QLabel("")
        self._validation_banner.setObjectName("ValidationError")
        self._validation_banner.hide()
        self._result_label = QLabel("")
        self._result_label.setWordWrap(True)
        self._result_label.setObjectName("SectionTitle")
        self._proba_host = QVBoxLayout()
        card.add_widget(self._predict_btn)
        card.add_widget(self._validation_banner)
        card.add_widget(self._result_label)
        card.add_layout(self._proba_host)
        layout.addWidget(card, 1)
        return box

    def _build_batch_tab(self) -> QWidget:
        box = QWidget()
        layout = QVBoxLayout(box)
        card = Card("Batch prediction")
        self._batch_path = QLabel("No file selected")
        self._batch_btn = QPushButton("Select CSV / Parquet")
        self._batch_btn.setObjectName("GhostButton")
        self._batch_run = QPushButton("Run batch prediction")
        self._batch_run.setObjectName("PrimaryButton")
        constrain_primary_button(self._batch_run)
        self._batch_status = QLabel("")
        self._batch_status.setObjectName("ValidationError")
        self._batch_status.setWordWrap(True)
        self._batch_btn.clicked.connect(self._pick_batch_file)
        self._batch_run.clicked.connect(self._run_batch)
        card.add_widget(self._batch_path)
        card.add_widget(self._batch_btn)
        card.add_widget(self._batch_run)
        card.add_widget(self._batch_status)
        layout.addWidget(card, 1)
        return box

    def bind_predictor(self, predictor, features: list[str], task) -> None:
        self._predictor = predictor
        self._features = list(features)
        self._task = task
        self._empty.hide()
        self._tabs.show()
        clear_layout(self._inputs_layout)
        self._inputs.clear()
        self._error_labels.clear()
        schema = schema_or_empty(getattr(predictor, "pipeline", None))
        cols = list(schema.keys()) if schema else list(features)
        for col in cols:
            wrap, inp, err = build_feature_row(col, schema.get(col), self)
            self._inputs[col] = inp
            self._error_labels[col] = err
            self._inputs_layout.addWidget(wrap)
        self._inputs_layout.addStretch(1)
        self._result_label.clear()
        clear_layout(self._proba_host)
        self._validation_banner.hide()

    def _pick_batch_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Batch file", str(Path.home()), "Data (*.csv *.parquet)"
        )
        if path:
            self._batch_path.setText(path)
            self._batch_status.clear()

    def _run_single(self) -> None:
        if not self._predictor:
            return
        clear_field_errors(self._inputs, self._error_labels)
        self._validation_banner.hide()
        values = collect_values(self._inputs)
        pipe = getattr(self._predictor, "pipeline", None)
        try:
            if pipe is not None and schema_or_empty(pipe):
                pipe.coerce_row(values)
            result = self._predictor.predict_single(
                values, explain=self._explain_check.isChecked()
            )
            self.show_prediction(result)
        except ValueError as exc:
            errors = exc.args[0] if exc.args and isinstance(exc.args[0], dict) else {}
            if errors:
                mark_field_errors(self._inputs, self._error_labels, errors)
                self._validation_banner.setText(
                    "; ".join(f"{k}: {v}" for k, v in errors.items())
                )
                self._validation_banner.show()
                return
            self._validation_banner.setText(str(exc))
            self._validation_banner.show()
        except Exception as exc:
            self._result_label.setText(f"Error: {exc}")

    def show_prediction(self, result) -> None:
        self._result_label.setText(f"Prediction: {result.prediction}")
        probs = getattr(result, "probabilities", None)
        render_proba_bars(self._proba_host, probs if isinstance(probs, dict) else None)
        if getattr(result, "explanation", None):
            note = QLabel("Local explanation available.")
            note.setObjectName("TextMuted")
            self._proba_host.addWidget(note)

    def _run_batch(self) -> None:
        path_text = self._batch_path.text()
        if not path_text or path_text == "No file selected" or not self._predictor:
            return
        path = Path(path_text)
        expected = list(
            getattr(self._predictor.pipeline, "input_feature_columns", None) or self._features
        )
        try:
            missing, extra = BatchPredictWorker.preflight(path, expected)
        except Exception as exc:
            self._batch_status.setText(str(exc))
            return
        if missing:
            self._batch_status.setText(f"Missing columns: {', '.join(missing)}")
            return
        self._batch_status.setText(
            f"Note: extra columns ignored: {', '.join(extra[:8])}" if extra else ""
        )
        default = str(path.with_name(f"{path.stem}_predictions.csv"))
        out, _ = QFileDialog.getSaveFileName(self, "Save predictions", default, "CSV (*.csv)")
        if out:
            self.batch_predict_requested.emit(path_text, out)

    def _pick_drift_file(self) -> None:
        pick_drift_file(self)

    def _run_drift(self) -> None:
        run_drift(self)

    def _run_importance(self) -> None:
        run_importance(self)
