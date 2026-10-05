"""Explain + drift tab builders/handlers for PredictPage (keeps page under size limits)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from PyQt6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ml_studio.gui.layout_utils import constrain_primary_button, configure_table_header
from ml_studio.gui.widgets.card import Card


def build_explain_tab(page) -> QWidget:
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
    page._explain_btn = QPushButton("Compute importance")
    page._explain_btn.setObjectName("PrimaryButton")
    constrain_primary_button(page._explain_btn)
    page._explain_btn.clicked.connect(page._run_importance)
    card.add_widget(page._explain_btn)
    page._importance_table = QTableWidget()
    page._importance_table.setColumnCount(2)
    page._importance_table.setHorizontalHeaderLabels(["Feature", "Importance"])
    configure_table_header(
        page._importance_table.horizontalHeader(), contents_cols=(0,), stretch_cols=(1,)
    )
    page._importance_placeholder = QLabel("No importance scores yet. Click Compute importance.")
    page._importance_placeholder.setObjectName("TextMuted")
    card.add_widget(page._importance_placeholder)
    card.add_widget(page._importance_table, stretch=1)
    page._importance_table.hide()
    page._explain_status = QLabel("")
    page._explain_status.setWordWrap(True)
    card.add_widget(page._explain_status)
    layout.addWidget(card, 1)
    return box


def build_drift_tab(page) -> QWidget:
    box = QWidget()
    layout = QVBoxLayout(box)
    card = Card("Drift monitoring")
    hint = QLabel(
        "Compare a batch file to the training feature distribution (PSI / KS). "
        "Requires a bound model and a CSV/Parquet with the same feature columns."
    )
    hint.setWordWrap(True)
    card.add_widget(hint)
    row = QHBoxLayout()
    page._drift_path = QLineEdit()
    page._drift_path.setPlaceholderText("Path to batch file…")
    pick = QPushButton("Browse")
    pick.setObjectName("GhostButton")
    pick.clicked.connect(page._pick_drift_file)
    run = QPushButton("Compute drift")
    run.setObjectName("PrimaryButton")
    constrain_primary_button(run)
    run.clicked.connect(page._run_drift)
    row.addWidget(page._drift_path, 1)
    row.addWidget(pick)
    row.addWidget(run)
    card.add_layout(row)
    page._drift_table = QTableWidget()
    page._drift_table.setColumnCount(4)
    page._drift_table.setHorizontalHeaderLabels(["Column", "PSI", "KS", "Status"])
    configure_table_header(
        page._drift_table.horizontalHeader(), contents_cols=(1, 2, 3), stretch_cols=(0,)
    )
    card.add_widget(page._drift_table, stretch=1)
    page._drift_status = QLabel("")
    page._drift_status.setObjectName("MonoMetric")
    card.add_widget(page._drift_status)
    layout.addWidget(card, 1)
    return box


def pick_drift_file(page) -> None:
    path, _ = QFileDialog.getOpenFileName(
        page, "Drift batch file", str(Path.home()), "Data (*.csv *.parquet)"
    )
    if path:
        page._drift_path.setText(path)


def run_drift(page) -> None:
    if not page._predictor:
        QMessageBox.warning(page, "Drift", "Bind a trained model first.")
        return
    path = page._drift_path.text().strip()
    if not path:
        QMessageBox.warning(page, "Drift", "Choose a batch file.")
        return
    try:
        from ml_studio.core.evaluation.drift import drift_report

        win = page.window()
        controller = getattr(win, "controller", None)
        if controller is None or controller.current_dataset is None:
            QMessageBox.warning(page, "Drift", "Load the training dataset for reference.")
            return
        ref = controller.current_dataset.dataframe
        p = Path(path)
        cur = pd.read_csv(p) if p.suffix.lower() == ".csv" else pd.read_parquet(p)
        cols = list(
            getattr(page._predictor.pipeline, "input_feature_columns", None) or page._features
        )
        report = drift_report(ref, cur, columns=[c for c in cols if c in ref.columns])
        page._drift_status.setText(
            f"Overall: {report['overall'].upper()}  ·  "
            f"{report['n_drift']} drift / {report['n_shift']} shift / {report['n_columns']} cols"
        )
        rows = report["columns"]
        page._drift_table.setRowCount(len(rows))
        for i, r in enumerate(rows):
            page._drift_table.setItem(i, 0, QTableWidgetItem(r["column"]))
            page._drift_table.setItem(
                i, 1, QTableWidgetItem("—" if r["psi"] is None else f"{r['psi']:.4f}")
            )
            page._drift_table.setItem(
                i, 2, QTableWidgetItem("—" if r["ks"] is None else f"{r['ks']:.4f}")
            )
            page._drift_table.setItem(i, 3, QTableWidgetItem(r["status"]))
    except Exception as exc:
        QMessageBox.critical(page, "Drift failed", str(exc))


def run_importance(page) -> None:
    if not page._predictor:
        page._explain_status.setText("Load a model first.")
        return
    win = page.window()
    controller = getattr(win, "controller", None)
    dataset = getattr(controller, "current_dataset", None) if controller else None
    result = getattr(controller, "current_result", None) if controller else None
    if dataset is None:
        page._explain_status.setText("Import the training dataset on the Data page first.")
        return
    target = getattr(result, "target_column", None) if result else None
    target = target or getattr(dataset, "target_column", None)
    features = page._features or list(
        getattr(page._predictor.pipeline, "input_feature_columns", None)
        or getattr(page._predictor.pipeline, "feature_columns", [])
        or []
    )
    if not features:
        page._explain_status.setText("No feature columns on the loaded model.")
        return
    try:
        _compute_importance(page, dataset, target, features)
    except Exception as exc:
        page._explain_status.setText(f"Explain failed: {exc}")


def _compute_importance(page, dataset, target, features) -> None:
    from ml_studio.core.evaluation.explain import compute_permutation_importance
    from ml_studio.core.training.task import TaskType

    df = dataset.dataframe
    missing = [c for c in features if c not in df.columns]
    if missing:
        page._explain_status.setText(
            f"Dataset is missing model features: {', '.join(missing[:5])}"
        )
        return
    sample = df[features].head(2000)
    task = page._task or getattr(page._predictor.pipeline, "task", None)
    unsupervised = task in (TaskType.CLUSTERING, TaskType.ANOMALY_DETECTION) if task else False
    if unsupervised or not target or target not in df.columns:
        page._explain_status.setText(
            "Permutation importance needs a target column on the current dataset."
        )
        return
    y = df.loc[sample.index, target]
    aligned = pd.concat([sample, y], axis=1).dropna()
    if len(aligned) < 10:
        page._explain_status.setText("Not enough complete rows to compute importance.")
        return
    X = page._predictor.pipeline.transform(aligned[features])
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    scores = compute_permutation_importance(
        page._predictor.pipeline.estimator, X, aligned[target], n_repeats=5
    )
    ranked = sorted(scores.items(), key=lambda kv: abs(kv[1]), reverse=True)
    page._importance_placeholder.hide()
    page._importance_table.show()
    page._importance_table.setRowCount(len(ranked))
    for i, (name, score) in enumerate(ranked):
        page._importance_table.setItem(i, 0, QTableWidgetItem(str(name)))
        page._importance_table.setItem(i, 1, QTableWidgetItem(f"{score:.6f}"))
    page._explain_status.setText(f"Computed on {len(aligned):,} rows (max 2000 sample).")
