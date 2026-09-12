"""Prediction page with single, batch, and explain tabs."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ml_studio.gui.pages.base_page import BasePage
from ml_studio.gui.widgets.card import Card
from ml_studio.gui.widgets.empty_state import EmptyState
from ml_studio.gui.widgets.pill_tabs import PillTabs


class PredictPage(BasePage):
    def __init__(self, container, parent=None):
        self._predictor = None
        self._features: list[str] = []
        self._task = None
        super().__init__(container, parent)

    def _build_ui(self) -> None:
        title = QLabel("Predict")
        title.setObjectName("PageTitle")
        self._layout.addWidget(title)

        single = self._build_single_tab()
        batch = self._build_batch_tab()
        explain = self._wrap(QLabel("Permutation importance and SHAP run against the loaded model."))
        drift = self._wrap(QLabel("Drift monitoring compares live inputs to training schema statistics."))

        self._tabs = PillTabs(
            [
                ("Single", "ai", single),
                ("Batch", "import", batch),
                ("Explain", "chart", explain),
                ("Drift", "scale", drift),
            ]
        )
        self._layout.addWidget(self._tabs, 1)
        self._empty = EmptyState("No model loaded", "Train and register a model to run predictions.", icon_name="ai")
        self._layout.addWidget(self._empty)

    def _wrap(self, widget: QWidget) -> QWidget:
        box = QWidget()
        layout = QVBoxLayout(box)
        layout.addWidget(widget)
        layout.addStretch()
        return box

    def _build_single_tab(self) -> QWidget:
        box = QWidget()
        layout = QVBoxLayout(box)
        card = Card("Single prediction")
        self._inputs: dict[str, QLineEdit] = {}
        self._inputs_layout = QVBoxLayout()
        card.add_layout(self._inputs_layout)
        self._predict_btn = QPushButton("Predict")
        self._predict_btn.setObjectName("PrimaryButton")
        self._result_label = QLabel("")
        card.add_widget(self._predict_btn)
        card.add_widget(self._result_label)
        layout.addWidget(card)
        layout.addStretch()
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
        self._batch_btn.clicked.connect(self._pick_batch_file)
        self._batch_run.clicked.connect(self._run_batch)
        card.add_widget(self._batch_path)
        card.add_widget(self._batch_btn)
        card.add_widget(self._batch_run)
        layout.addWidget(card)
        layout.addStretch()
        return box

    def bind_predictor(self, predictor, features: list[str], task) -> None:
        self._predictor = predictor
        self._features = features
        self._task = task
        self._empty.hide()
        while self._inputs_layout.count():
            item = self._inputs_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._inputs.clear()
        for col in features:
            row = QHBoxLayout()
            row.addWidget(QLabel(col))
            inp = QLineEdit()
            self._inputs[col] = inp
            row.addWidget(inp)
            wrap = QWidget()
            wrap.setLayout(row)
            self._inputs_layout.addWidget(wrap)
        self._predict_btn.clicked.connect(self._run_single)

    def _pick_batch_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Batch file", str(Path.home()), "Data (*.csv *.parquet)")
        if path:
            self._batch_path.setText(path)

    def _run_single(self) -> None:
        if not self._predictor:
            return
        import pandas as pd

        row = {col: inp.text() for col, inp in self._inputs.items()}
        df = pd.DataFrame([row])
        pred = self._predictor.predict(df)
        self._result_label.setText(f"Prediction: {pred[0]}")

    def show_prediction(self, result) -> None:
        self._result_label.setText(f"Prediction: {result.prediction}")

    def _run_batch(self) -> None:
        path = self._batch_path.text()
        if not path or path == "No file selected" or not self._predictor:
            return
        import pandas as pd
        from PyQt6.QtWidgets import QMessageBox

        try:
            if path.endswith(".csv"):
                df = pd.read_csv(path)
            elif path.endswith(".parquet"):
                df = pd.read_parquet(path)
            else:
                raise ValueError("Unsupported file format")

            preds = self._predictor.predict(df)
            df["Prediction"] = preds
            
            out_path = path.replace(".csv", "_predictions.csv").replace(".parquet", "_predictions.csv")
            if out_path == path:
                out_path += "_predictions.csv"
            df.to_csv(out_path, index=False)
            
            QMessageBox.information(self, "Success", f"Batch predictions saved to:\n{out_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to run batch prediction:\n{e}")
