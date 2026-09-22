"""Prediction page with single, batch, and explain tabs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ml_studio.app.theme import ThemeMode
from ml_studio.gui.layout_utils import constrain_primary_button, configure_table_header
from ml_studio.gui.pages.base_page import BasePage
from ml_studio.gui.widgets.card import Card
from ml_studio.gui.widgets.empty_state import EmptyState
from ml_studio.gui.widgets.pill_tabs import PillTabs


class PredictPage(BasePage):
    batch_predict_requested = pyqtSignal(str)

    def __init__(self, container, parent=None):
        self._predictor = None
        self._features: list[str] = []
        self._task = None
        self._predict_connected = False
        self._mode = ThemeMode.LIGHT
        super().__init__(container, parent)

    def _build_ui(self) -> None:
        title = QLabel("Predict")
        title.setObjectName("PageTitle")
        self._layout.addWidget(title)

        single = self._build_single_tab()
        batch = self._build_batch_tab()
        explain = self._build_explain_tab()
        drift = self._build_drift_tab()

        self._tabs = PillTabs(
            [
                ("Single", "ai", single),
                ("Batch", "import", batch),
                ("Explain", "chart", explain),
                ("Drift", "scale", drift),
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

    def _wrap(self, widget: QWidget) -> QWidget:
        box = QWidget()
        layout = QVBoxLayout(box)
        layout.setContentsMargins(0, 0, 0, 0)
        widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(widget, 1)
        return box

    def _build_single_tab(self) -> QWidget:
        box = QWidget()
        layout = QVBoxLayout(box)
        card = Card("Single prediction")
        self._inputs: dict[str, QLineEdit] = {}
        self._inputs_layout = QVBoxLayout()
        card.add_layout(self._inputs_layout)
        self._explain_check = QCheckBox("Include local explanation (SHAP if available)")
        card.add_widget(self._explain_check)
        self._predict_btn = QPushButton("Predict")
        self._predict_btn.setObjectName("PrimaryButton")
        constrain_primary_button(self._predict_btn)
        self._result_label = QLabel("")
        self._result_label.setWordWrap(True)
        card.add_widget(self._predict_btn)
        card.add_widget(self._result_label)
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
        self._batch_btn.clicked.connect(self._pick_batch_file)
        self._batch_run.clicked.connect(self._run_batch)
        card.add_widget(self._batch_path)
        card.add_widget(self._batch_btn)
        card.add_widget(self._batch_run)
        layout.addWidget(card, 1)
        return box

    def _build_explain_tab(self) -> QWidget:
        box = QWidget()
        layout = QVBoxLayout(box)
        card = Card("Feature importance")
        hint = QLabel(
            "Permutation importance on the current dataset (sampled). "
            "Requires a loaded model and an imported dataset with the training target."
        )
        hint.setObjectName("TextMuted")
        hint.setWordWrap(True)
        card.add_widget(hint)
        self._explain_btn = QPushButton("Compute importance")
        self._explain_btn.setObjectName("PrimaryButton")
        constrain_primary_button(self._explain_btn)
        self._explain_btn.clicked.connect(self._run_importance)
        card.add_widget(self._explain_btn)
        self._importance_table = QTableWidget()
        self._importance_table.setColumnCount(2)
        self._importance_table.setHorizontalHeaderLabels(["Feature", "Importance"])
        configure_table_header(
            self._importance_table.horizontalHeader(),
            contents_cols=(0,),
            stretch_cols=(1,),
        )
        self._importance_table.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._importance_placeholder = QLabel("No importance scores yet. Click Compute importance.")
        self._importance_placeholder.setObjectName("TextMuted")
        self._importance_placeholder.setAlignment(
            __import__("PyQt6.QtCore", fromlist=["Qt"]).Qt.AlignmentFlag.AlignCenter
        )
        card.add_widget(self._importance_placeholder)
        card.add_widget(self._importance_table, stretch=1)
        self._importance_table.hide()
        self._explain_status = QLabel("")
        self._explain_status.setWordWrap(True)
        card.add_widget(self._explain_status)
        layout.addWidget(card, 1)
        return box

    def _build_drift_tab(self) -> QWidget:
        box = QWidget()
        layout = QVBoxLayout(box)
        card = Card("Drift monitoring")
        label = QLabel(
            "Coming soon — drift will compare live batch inputs to training "
            "schema statistics (mean/std, category frequency)."
        )
        label.setWordWrap(True)
        card.add_widget(label)
        layout.addWidget(card, 1)
        return box

    def bind_predictor(self, predictor, features: list[str], task) -> None:
        self._predictor = predictor
        self._features = list(features)
        self._task = task
        self._empty.hide()
        self._tabs.show()
        while self._inputs_layout.count():
            item = self._inputs_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._inputs.clear()
        for col in self._features:
            row = QHBoxLayout()
            row.addWidget(QLabel(col))
            inp = QLineEdit()
            self._inputs[col] = inp
            row.addWidget(inp)
            wrap = QWidget()
            wrap.setLayout(row)
            self._inputs_layout.addWidget(wrap)
        if not self._predict_connected:
            self._predict_btn.clicked.connect(self._run_single)
            self._predict_connected = True

    def _pick_batch_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Batch file", str(Path.home()), "Data (*.csv *.parquet)"
        )
        if path:
            self._batch_path.setText(path)

    def _run_single(self) -> None:
        if not self._predictor:
            return
        features = {col: inp.text() for col, inp in self._inputs.items()}
        try:
            result = self._predictor.predict_single(
                features, explain=self._explain_check.isChecked()
            )
            self.show_prediction(result)
        except Exception as exc:
            self._result_label.setText(f"Error: {exc}")
            QMessageBox.critical(self, "Predict", f"Prediction failed:\n{exc}")

    def show_prediction(self, result) -> None:
        text = f"Prediction: {result.prediction}"
        if getattr(result, "probabilities", None):
            probs = ", ".join(f"{p:.3f}" for p in result.probabilities)
            text += f"  (proba: {probs})"
        if getattr(result, "explanation", None):
            text += "\nLocal explanation available (see Explain details in result metadata)."
            shap = result.explanation.get("shap") if isinstance(result.explanation, dict) else None
            if shap and shap.get("feature_names"):
                text += f"\nSHAP features: {', '.join(map(str, shap['feature_names'][:8]))}"
        self._result_label.setText(text)

    def _run_batch(self) -> None:
        path_text = self._batch_path.text()
        if not path_text or path_text == "No file selected" or not self._predictor:
            return
        self.batch_predict_requested.emit(path_text)

    def _run_importance(self) -> None:
        if not self._predictor:
            self._explain_status.setText("Load a model first.")
            return
        win = self.window()
        controller = getattr(win, "controller", None)
        dataset = getattr(controller, "current_dataset", None) if controller else None
        result = getattr(controller, "current_result", None) if controller else None
        if dataset is None:
            self._explain_status.setText("Import the training dataset on the Data page first.")
            return

        target = None
        if result is not None:
            target = getattr(result, "target_column", None)
        target = target or getattr(dataset, "target_column", None)
        features = self._features or list(
            getattr(self._predictor.pipeline, "feature_columns", []) or []
        )
        if not features:
            self._explain_status.setText("No feature columns on the loaded model.")
            return

        try:
            from ml_studio.core.evaluation.explain import compute_permutation_importance
            from ml_studio.core.training.task import TaskType

            df = dataset.dataframe
            missing = [c for c in features if c not in df.columns]
            if missing:
                self._explain_status.setText(
                    f"Dataset is missing model features: {', '.join(missing[:5])}"
                )
                return

            sample = df[features].head(2000)
            task = self._task or getattr(self._predictor.pipeline, "task", None)
            unsupervised = task in (
                TaskType.CLUSTERING,
                TaskType.ANOMALY_DETECTION,
            ) if task is not None else False

            if unsupervised or not target or target not in df.columns:
                # Fall back: variance of predictions under column shuffle proxy — skip
                self._explain_status.setText(
                    "Permutation importance needs a target column on the current dataset."
                )
                return

            y = df.loc[sample.index, target]
            aligned = pd.concat([sample, y], axis=1).dropna()
            if len(aligned) < 10:
                self._explain_status.setText("Not enough complete rows to compute importance.")
                return
            X_raw = aligned[features]
            y_s = aligned[target]
            X = self._predictor.pipeline.transform(X_raw)
            if not isinstance(X, pd.DataFrame):
                X = pd.DataFrame(X, columns=[f"f{i}" for i in range(getattr(X, "shape", [0, 0])[1])])
            model = self._predictor.pipeline.estimator
            self._explain_status.setText("Computing permutation importance…")
            scores = compute_permutation_importance(model, X, y_s, n_repeats=5)
            ranked = sorted(scores.items(), key=lambda kv: abs(kv[1]), reverse=True)
            self._importance_placeholder.hide()
            self._importance_table.show()
            self._importance_table.setRowCount(len(ranked))
            for i, (name, score) in enumerate(ranked):
                self._importance_table.setItem(i, 0, QTableWidgetItem(str(name)))
                self._importance_table.setItem(i, 1, QTableWidgetItem(f"{score:.6f}"))
            self._explain_status.setText(f"Computed on {len(aligned):,} rows (max 2000 sample).")
        except Exception as exc:
            self._explain_status.setText(f"Explain failed: {exc}")
            QMessageBox.warning(self, "Explain", str(exc))
