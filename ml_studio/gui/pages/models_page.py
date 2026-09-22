"""Model registry with search, detail panel, and actions."""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ml_studio.app.metric_color import metric_color
from ml_studio.app.theme import ThemeMode
from ml_studio.gui.layout_utils import configure_table_header
from ml_studio.gui.pages.base_page import BasePage
from ml_studio.gui.widgets.card import Card
from ml_studio.gui.widgets.empty_state import EmptyState
from ml_studio.gui.widgets.search_bar import SearchBar


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

        hint = QLabel("Click a row to select a model. Hold Ctrl and click to select several for Compare.")
        hint.setObjectName("TextMuted")
        hint.setWordWrap(True)
        self._layout.addWidget(hint)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        self._table = QTableWidget()
        self._table.setObjectName("ModelRegistryTable")
        self._table.setColumnCount(8)
        self._table.setHorizontalHeaderLabels(
            ["Name", "Version", "Task", "Score", "Dataset", "Date", "Tags", "Status"]
        )
        configure_table_header(
            self._table.horizontalHeader(),
            contents_cols=(1, 2, 3, 5, 7),
            stretch_cols=(0, 4),
        )
        self._table.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QTableWidget.SelectionMode.ExtendedSelection)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setAlternatingRowColors(True)
        self._table.setShowGrid(False)
        self._table.verticalHeader().setVisible(False)
        self._table.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._table.itemSelectionChanged.connect(self._show_detail)
        self._table.cellClicked.connect(lambda _r, _c: self._show_detail())
        splitter.addWidget(self._table)

        self._detail = Card("Model details")
        self._detail.setMinimumWidth(220)
        self._detail_title = QLabel("No model selected")
        self._detail_title.setObjectName("PageTitle")
        self._detail_meta = QLabel("Click a row in the table to select a model.")
        self._detail_meta.setWordWrap(True)
        self._detail_meta.setObjectName("TextMuted")
        self._selection_badge = QLabel("")
        self._selection_badge.setObjectName("SelectionBadge")
        self._selection_badge.hide()
        self._detail_actions = QHBoxLayout()
        self._btn_view = QPushButton("View")
        self._btn_compare = QPushButton("Compare")
        self._btn_predict = QPushButton("Predict")
        self._btn_export = QPushButton("Export")
        for btn in (self._btn_view, self._btn_compare, self._btn_predict, self._btn_export):
            btn.setObjectName("GhostButton")
            btn.setEnabled(False)
            self._detail_actions.addWidget(btn)

        self._btn_view.clicked.connect(self._action_view)
        self._btn_compare.clicked.connect(self._action_compare)
        self._btn_predict.clicked.connect(self._action_predict)
        self._btn_export.clicked.connect(self._action_export)
        self._detail.add_widget(self._selection_badge)
        self._detail.add_widget(self._detail_title)
        self._detail.add_widget(self._detail_meta, stretch=1)
        self._detail.add_layout(self._detail_actions)
        splitter.addWidget(self._detail)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([720, 360])
        self._content_splitter = splitter
        self._layout.addWidget(splitter, 1)

        self._empty = EmptyState(
            "No models registered",
            "Train a model to populate the registry.",
            icon_name="clipboard",
        )
        self._layout.addWidget(self._empty, 1)
        self._selected_model = None

    def set_theme_mode(self, mode: ThemeMode) -> None:
        self._mode = mode
        self._empty.set_theme_mode(mode)

    def refresh(self, registry) -> None:
        self._registry = registry
        self._models = registry.list_models()
        empty = len(self._models) == 0
        self._empty.setVisible(empty)
        self._content_splitter.setVisible(not empty)
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
            name_item = QTableWidgetItem(model.name)
            name_item.setToolTip(model.name)
            self._table.setItem(i, 0, name_item)
            self._table.setItem(i, 1, QTableWidgetItem(str(model.version)))
            self._table.setItem(i, 2, QTableWidgetItem(model.task.value))
            score_item = QTableWidgetItem(score_text)
            if isinstance(score, float):
                color = metric_color(
                    self._mode,
                    "r2" if model.task.value == "REGRESSION" else "f1",
                    score,
                    model.task.value,
                )
                score_item.setForeground(
                    __import__("PyQt6.QtGui", fromlist=["QColor"]).QColor(color)
                )
            self._table.setItem(i, 3, score_item)
            self._table.setItem(i, 4, QTableWidgetItem(model.dataset_id or "—"))
            self._table.setItem(
                i, 5, QTableWidgetItem(model.training_timestamp.strftime("%Y-%m-%d"))
            )
            tags = ", ".join(model.tags) if model.tags else "—"
            self._table.setItem(i, 6, QTableWidgetItem(tags))
            status = "archived" if "archived" in (model.tags or []) else "active"
            status_item = QTableWidgetItem(status)
            if status == "active":
                status_item.setForeground(
                    __import__("PyQt6.QtGui", fromlist=["QColor"]).QColor("#3FB950")
                )
            self._table.setItem(i, 7, status_item)

        if self._filtered_models:
            self._table.selectRow(0)
            self._show_detail()
        else:
            self._clear_selection_ui()

    def _clear_selection_ui(self) -> None:
        self._selected_model = None
        self._detail_title.setText("No model selected")
        self._detail_meta.setText("Click a row in the table to select a model.")
        self._selection_badge.hide()
        for btn in (self._btn_view, self._btn_compare, self._btn_predict, self._btn_export):
            btn.setEnabled(False)

    def _show_detail(self) -> None:
        rows = self._table.selectionModel().selectedRows()
        if not rows:
            self._clear_selection_ui()
            return
        row_idx = rows[0].row()
        if row_idx < 0 or row_idx >= len(self._filtered_models):
            self._clear_selection_ui()
            return
        self._selected_model = self._filtered_models[row_idx]
        n = len(rows)
        self._selection_badge.setText(
            "Selected" if n == 1 else f"{n} models selected"
        )
        self._selection_badge.show()
        self._detail_title.setText(self._selected_model.name)
        task = self._selected_model.task.value
        score = (
            self._selected_model.metrics.get("r2")
            or self._selected_model.metrics.get("f1")
            or self._selected_model.metrics.get("accuracy")
        )
        score_txt = f"{score:.4f}" if isinstance(score, float) else "—"
        self._detail_meta.setText(
            f"Task: {task}\nScore: {score_txt}\n"
            f"Version: {self._selected_model.version}\n\n"
            "View opens details. Compare needs 2+ rows (Ctrl+click)."
        )
        self._btn_view.setEnabled(True)
        self._btn_predict.setEnabled(True)
        self._btn_export.setEnabled(True)
        self._btn_compare.setEnabled(n >= 1)

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
