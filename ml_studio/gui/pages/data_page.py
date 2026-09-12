"""Full data exploration page with table, profiling, and quality tabs."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableView,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ml_studio.app.metric_color import metric_color
from ml_studio.app.theme import ThemeMode
from ml_studio.gui.pages.base_page import BasePage
from ml_studio.gui.widgets.card import Card
from ml_studio.gui.widgets.empty_state import EmptyState
from ml_studio.gui.widgets.icon_button import IconButton
from ml_studio.gui.widgets.loading_overlay import LoadingOverlay
from ml_studio.gui.widgets.pill_tabs import PillTabs
from ml_studio.gui.widgets.search_bar import SearchBar
from ml_studio.gui.widgets.tag_chip import TagChip
from ml_studio.gui.widgets.data_table import DataFrameTableModel


class DataPage(BasePage):
    def __init__(self, container, parent=None):
        self._dataset = None
        self._mode = ThemeMode.LIGHT
        self._model = DataFrameTableModel()
        super().__init__(container, parent)

    def _build_ui(self) -> None:
        header = QHBoxLayout()
        title = QLabel("Data")
        title.setObjectName("PageTitle")
        header.addWidget(title)
        header.addStretch()
        self._import_btn = QPushButton("Import")
        self._import_btn.setObjectName("PrimaryButton")
        self._optimize_btn = QPushButton("Optimize Memory")
        self._optimize_btn.setObjectName("GhostButton")
        self._profile_btn = QPushButton("Refresh Profile")
        self._profile_btn.setObjectName("GhostButton")
        header.addWidget(self._import_btn)
        header.addWidget(self._optimize_btn)
        header.addWidget(self._profile_btn)
        self._layout.addLayout(header)

        self._table_view = QTableView()
        self._table_view.setModel(self._model)
        self._table_view.setSortingEnabled(True)
        self._table_view.setAlternatingRowColors(True)
        self._table_view.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)

        filter_row = QHBoxLayout()
        self._search = SearchBar("Filter rows…")
        self._search.search_changed.connect(self._model.apply_filter)
        self._column_filter = QComboBox()
        self._column_filter.addItem("All columns")
        filter_row.addWidget(self._search, 3)
        filter_row.addWidget(self._column_filter, 1)
        table_card = Card("Dataset table")
        table_layout = QVBoxLayout()
        table_layout.addLayout(filter_row)
        table_layout.addWidget(self._table_view)
        table_card.add_layout(table_layout)

        self._profile_table = self._make_profile_table()
        self._issues_table = self._make_issues_table()
        self._tabs = PillTabs(
            [
                ("Table", "table", self._wrap(table_card)),
                ("Profiling", "chart", self._wrap(self._profile_table)),
                ("Quality Issues", "missing", self._wrap(self._issues_table)),
            ]
        )
        self._layout.addWidget(self._tabs, 1)

        self._empty = EmptyState(
            "No dataset loaded",
            "Import a CSV, Excel, Parquet, or JSON file to explore your data.",
            icon_name="import",
        )
        self._error_label = QLabel("")
        self._error_label.setObjectName("ErrorBanner")
        self._error_label.hide()
        self._layout.addWidget(self._error_label)
        self._layout.addWidget(self._empty)
        self._overlay = LoadingOverlay(parent=self)

    def _wrap(self, widget: QWidget) -> QWidget:
        box = QWidget()
        layout = QVBoxLayout(box)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(widget)
        return box

    def _make_profile_table(self) -> QTableWidget:
        table = QTableWidget()
        table.setColumnCount(10)
        table.setHorizontalHeaderLabels(
            ["Column", "Type", "Missing", "Unique", "Min", "Max", "Mean", "Median", "Std", "Top values"]
        )
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        return table

    def _make_issues_table(self) -> QTableWidget:
        table = QTableWidget()
        table.setColumnCount(5)
        table.setHorizontalHeaderLabels(["Severity", "Type", "Column", "Details", "Action"])
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        table.setAlternatingRowColors(True)
        return table

    def set_theme_mode(self, mode: ThemeMode) -> None:
        self._mode = mode
        self._empty.set_theme_mode(mode)
        self._tabs.set_theme_mode(mode)

    def set_loading(self, active: bool, message: str = "Loading dataset…") -> None:
        if active:
            self._overlay.set_message(message)
            self._overlay.show_overlay()
        else:
            self._overlay.hide_overlay()

    def show_error(self, message: str) -> None:
        self._error_label.setText(message)
        self._error_label.show()

    def clear_error(self) -> None:
        self._error_label.hide()

    def set_dataset(self, dataset) -> None:
        self._dataset = dataset
        self.clear_error()
        if dataset is None:
            self._model.set_dataframe(__import__("pandas").DataFrame())
            self._empty.show()
            self._tabs.hide()
            return
        self._empty.hide()
        self._tabs.show()
        self._model.set_dataframe(dataset.dataframe)
        self._column_filter.clear()
        self._column_filter.addItem("All columns")
        self._column_filter.addItems([str(c) for c in dataset.dataframe.columns])
        self._populate_profile()

    def _populate_profile(self) -> None:
        if not self._dataset:
            return
        from ml_studio.core.profiling import detect_quality_issues, profile_dataset

        profile = profile_dataset(self._dataset)
        self._profile_table.setRowCount(len(profile.columns))
        for i, col in enumerate(profile.columns):
            self._profile_table.setItem(i, 0, QTableWidgetItem(col.name))
            self._profile_table.setItem(i, 1, QTableWidgetItem(col.dtype))
            self._profile_table.setItem(i, 2, QTableWidgetItem(f"{col.missing_pct:.1f}%"))
            self._profile_table.setItem(i, 3, QTableWidgetItem(str(col.unique)))
            self._profile_table.setItem(i, 4, QTableWidgetItem(str(col.min or "")))
            self._profile_table.setItem(i, 5, QTableWidgetItem(str(col.max or "")))
            self._profile_table.setItem(i, 6, QTableWidgetItem(f"{col.mean:.4g}" if col.mean else ""))
            self._profile_table.setItem(i, 7, QTableWidgetItem(f"{col.median:.4g}" if col.median else ""))
            self._profile_table.setItem(i, 8, QTableWidgetItem(f"{col.std:.4g}" if col.std else ""))
            self._profile_table.setItem(i, 9, QTableWidgetItem("—"))

        issues = detect_quality_issues(self._dataset)
        self._issues_table.setRowCount(max(1, len(issues)))
        if not issues:
            self._issues_table.setItem(0, 0, QTableWidgetItem("—"))
            self._issues_table.setItem(0, 1, QTableWidgetItem("none"))
            self._issues_table.setItem(0, 2, QTableWidgetItem("—"))
            self._issues_table.setItem(0, 3, QTableWidgetItem("No quality issues detected."))
            return

        for i, issue in enumerate(issues):
            severity = issue.get("severity", "info")
            chip = TagChip(severity.upper(), severity if severity in TagChip.VARIANTS else "info")
            self._issues_table.setCellWidget(i, 0, chip)
            self._issues_table.setItem(i, 1, QTableWidgetItem(issue.get("type", "")))
            self._issues_table.setItem(i, 2, QTableWidgetItem(str(issue.get("column", "—"))))
            detail = f"{issue['pct']:.1f}% missing" if "pct" in issue else str(issue.get("count", ""))
            self._issues_table.setItem(i, 3, QTableWidgetItem(detail or "—"))
            fix_btn = QPushButton("Review")
            fix_btn.setObjectName("GhostButton")
            self._issues_table.setCellWidget(i, 4, fix_btn)
