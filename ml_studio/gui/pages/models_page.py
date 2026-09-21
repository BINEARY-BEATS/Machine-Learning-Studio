"""Model registry with search, detail panel, and actions."""

from __future__ import annotations

from PyQt6.QtWidgets import (
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from PyQt6.QtCore import pyqtSignal
from ml_studio.app.metric_color import metric_color
from ml_studio.app.theme import ThemeMode
from ml_studio.gui.pages.base_page import BasePage
from ml_studio.gui.widgets.card import Card
from ml_studio.gui.widgets.empty_state import EmptyState
from ml_studio.gui.widgets.search_bar import SearchBar
from ml_studio.gui.widgets.tag_chip import TagChip


class ModelsPage(BasePage):
    predict_requested = pyqtSignal(str)
    def __init__(self, container, parent=None):
        self._registry = None
        self._models = []
        self._mode = ThemeMode.LIGHT
        super().__init__(container, parent)

    def _build_ui(self) -> None:
        header = QHBoxLayout()
        title = QLabel("Model Registry")
        title.setObjectName("PageTitle")
        header.addWidget(title)
        header.addStretch()
        self._refresh_btn = QPushButton("Refresh")
        self._refresh_btn.setObjectName("GhostButton")
        header.addWidget(self._refresh_btn)
        self._layout.addLayout(header)

        self._search = SearchBar("Search models…")
        self._search.search_changed.connect(self._apply_filter)
        self._layout.addWidget(self._search)

        splitter = QSplitter()
        self._table = QTableWidget()
        self._table.setColumnCount(8)
        self._table.setHorizontalHeaderLabels(
            ["Name", "Version", "Task", "Score", "Dataset", "Date", "Tags", "Status"]
        )
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._table.itemSelectionChanged.connect(self._show_detail)
        splitter.addWidget(self._table)

        self._detail = Card("Model details")
        self._detail_title = QLabel("Select a model")
        self._detail_meta = QLabel("")
        self._detail_meta.setWordWrap(True)
        self._detail_actions = QHBoxLayout()
        self._btn_view = QPushButton("View")
        self._btn_compare = QPushButton("Compare")
        self._btn_predict = QPushButton("Predict")
        self._btn_export = QPushButton("Export")
        for btn in (self._btn_view, self._btn_compare, self._btn_predict, self._btn_export):
            btn.setObjectName("GhostButton")
            self._detail_actions.addWidget(btn)
        
        self._btn_view.clicked.connect(self._action_view)
        self._btn_compare.clicked.connect(self._action_compare)
        self._btn_predict.clicked.connect(self._action_predict)
        self._btn_export.clicked.connect(self._action_export)
        self._detail.add_widget(self._detail_title)
        self._detail.add_widget(self._detail_meta)
        self._detail.add_layout(self._detail_actions)
        splitter.addWidget(self._detail)
        splitter.setSizes([620, 380])
        self._layout.addWidget(splitter, 1)

        self._empty = EmptyState("No models registered", "Train a model to populate the registry.", icon_name="clipboard")
        self._layout.addWidget(self._empty)

    def set_theme_mode(self, mode: ThemeMode) -> None:
        self._mode = mode
        self._empty.set_theme_mode(mode)

    def refresh(self, registry) -> None:
        self._registry = registry
        self._models = registry.list_models()
        self._empty.setVisible(len(self._models) == 0)
        self._apply_filter("")

    def _apply_filter(self, query: str) -> None:
        q = query.lower().strip()
        self._filtered_models = [
            m
            for m in self._models
            if not q or q in m.name.lower() or q in m.task.value.lower()
        ]
        self._table.setRowCount(len(self._filtered_models))
        for i, model in enumerate(self._filtered_models):
            score = model.metrics.get("r2") or model.metrics.get("f1") or model.metrics.get("accuracy")
            score_text = f"{score:.4f}" if isinstance(score, float) else "—"
            self._table.setItem(i, 0, QTableWidgetItem(model.name))
            self._table.setItem(i, 1, QTableWidgetItem(str(model.version)))
            self._table.setItem(i, 2, QTableWidgetItem(model.task.value))
            score_item = QTableWidgetItem(score_text)
            if isinstance(score, float):
                color = metric_color(self._mode, "r2" if model.task.value == "REGRESSION" else "f1", score, model.task.value)
                score_item.setForeground(__import__("PyQt6.QtGui", fromlist=["QColor"]).QColor(color))
            self._table.setItem(i, 3, score_item)
            self._table.setItem(i, 4, QTableWidgetItem(model.dataset_id or "—"))
            self._table.setItem(i, 5, QTableWidgetItem(model.training_timestamp.strftime("%Y-%m-%d")))
            self._table.setCellWidget(i, 6, TagChip(", ".join(model.tags) if model.tags else "none"))
            status = "archived" if "archived" in model.tags else "active"
            self._table.setCellWidget(i, 7, TagChip(status, "success" if status == "active" else "warning"))

    def _show_detail(self) -> None:
        rows = self._table.selectionModel().selectedRows()
        if not rows:
            return
        row_idx = rows[0].row()
        self._selected_model = self._filtered_models[row_idx]
        self._detail_title.setText(self._selected_model.name)
        self._detail_meta.setText("Use actions to compare, export, or deploy this model version.")

    def _action_view(self) -> None:
        if not getattr(self, "_selected_model", None):
            return
        from PyQt6.QtWidgets import QDialog, QVBoxLayout, QTextEdit, QPushButton
        import json
        
        d = QDialog(self)
        d.setWindowTitle(f"Model Details - {self._selected_model.name}")
        d.resize(500, 400)
        
        layout = QVBoxLayout(d)
        txt = QTextEdit()
        txt.setReadOnly(True)
        txt.setText(json.dumps(self._selected_model.to_dict(), indent=2))
        layout.addWidget(txt)
        
        btn = QPushButton("Close")
        btn.clicked.connect(d.accept)
        layout.addWidget(btn)
        
        d.exec()

    def _action_compare(self) -> None:
        rows = self._table.selectionModel().selectedRows()
        if len(rows) < 2:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Compare Models", "Please select at least two models to compare (use Ctrl+Click).")
            return
            
        selected_models = [self._filtered_models[row.row()] for row in rows]
        
        from PyQt6.QtWidgets import QDialog, QVBoxLayout, QTableWidget, QTableWidgetItem, QHeaderView, QPushButton
        
        d = QDialog(self)
        d.setWindowTitle("Compare Models")
        d.resize(800, 600)
        
        layout = QVBoxLayout(d)
        table = QTableWidget()
        table.setColumnCount(len(selected_models) + 1)
        headers = ["Metric/Hyperparameter"] + [m.name for m in selected_models]
        table.setHorizontalHeaderLabels(headers)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        
        metrics_keys = set()
        hp_keys = set()
        for m in selected_models:
            metrics_keys.update(m.metrics.keys())
            hp_keys.update(m.hyperparameters.keys())
            
        metrics_keys = sorted(list(metrics_keys))
        hp_keys = sorted(list(hp_keys))
        
        table.setRowCount(len(metrics_keys) + len(hp_keys) + 2)
        
        row_idx = 0
        table.setItem(row_idx, 0, QTableWidgetItem("--- METRICS ---"))
        row_idx += 1
        for key in metrics_keys:
            table.setItem(row_idx, 0, QTableWidgetItem(key))
            for col_idx, m in enumerate(selected_models):
                val = m.metrics.get(key, "N/A")
                if isinstance(val, float): val = f"{val:.4f}"
                table.setItem(row_idx, col_idx + 1, QTableWidgetItem(str(val)))
            row_idx += 1
            
        table.setItem(row_idx, 0, QTableWidgetItem("--- HYPERPARAMETERS ---"))
        row_idx += 1
        for key in hp_keys:
            table.setItem(row_idx, 0, QTableWidgetItem(key))
            for col_idx, m in enumerate(selected_models):
                val = m.hyperparameters.get(key, "N/A")
                table.setItem(row_idx, col_idx + 1, QTableWidgetItem(str(val)))
            row_idx += 1
            
        layout.addWidget(table)
        
        btn = QPushButton("Close")
        btn.clicked.connect(d.accept)
        layout.addWidget(btn)
        
        d.exec()

    def _action_predict(self) -> None:
        if not getattr(self, "_selected_model", None):
            return
        self.predict_requested.emit(self._selected_model.model_id)

    def _action_export(self) -> None:
        if not getattr(self, "_selected_model", None):
            return
        from PyQt6.QtWidgets import QFileDialog, QMessageBox
        from pathlib import Path
        import shutil
        
        save_path, _ = QFileDialog.getSaveFileName(self, "Export Model Directory", f"{self._selected_model.name}.zip", "ZIP Archives (*.zip)")
        if not save_path:
            return
        
        try:
            artifact_dir = Path(self._selected_model.artifact_dir)
            if save_path.endswith('.zip'):
                save_path = save_path[:-4]
            shutil.make_archive(save_path, 'zip', artifact_dir)
            QMessageBox.information(self, "Export Successful", f"Model exported successfully to:\n{save_path}.zip")
        except Exception as e:
            QMessageBox.critical(self, "Export Failed", f"Failed to export model:\n{e}")
