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
from ml_studio.gui.layout_utils import constrain_primary_button, configure_table_header
from ml_studio.gui.pages.base_page import BasePage
from ml_studio.gui.widgets.card import Card
from ml_studio.gui.widgets.empty_state import EmptyState
from ml_studio.gui.widgets.icon_button import IconButton
from ml_studio.gui.widgets.loading_overlay import LoadingOverlay
from ml_studio.gui.widgets.pill_tabs import PillTabs
from ml_studio.gui.widgets.search_bar import SearchBar
from ml_studio.gui.widgets.tag_chip import TagChip
from ml_studio.gui.widgets.data_table import DataFrameTableModel
from PyQt6.QtWidgets import QSizePolicy


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
        constrain_primary_button(self._import_btn)
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
        self._table_view.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._table_view.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Interactive
        )
        self._table_view.horizontalHeader().setStretchLastSection(True)

        filter_row = QHBoxLayout()
        self._search = SearchBar("Filter rows…")
        self._search.search_changed.connect(self._on_filter_changed)
        self._column_filter = QComboBox()
        self._column_filter.addItem("All columns")
        self._column_filter.currentIndexChanged.connect(self._on_filter_changed)
        self._row_count_label = QLabel("Showing 0 of 0 rows")
        self._row_count_label.setObjectName("Breadcrumb")
        filter_row.addWidget(self._search, 3)
        filter_row.addWidget(self._column_filter, 1)
        filter_row.addWidget(self._row_count_label)
        table_card = Card("Dataset table")
        table_layout = QVBoxLayout()
        table_layout.addLayout(filter_row)
        table_layout.addWidget(self._table_view, 1)
        table_card.add_layout(table_layout, stretch=1)

        self._profile_table = self._make_profile_table()
        self._issues_table = self._make_issues_table()
        self._plots_host = QWidget()
        self._plots_layout = QVBoxLayout(self._plots_host)
        self._plots_layout.setContentsMargins(0, 0, 0, 0)
        self._plots_placeholder = QLabel("Profile a dataset to see distributions and correlations.")
        self._plots_placeholder.setObjectName("Breadcrumb")
        self._plots_placeholder.setWordWrap(True)
        self._plots_layout.addWidget(self._plots_placeholder)
        self._tabs = PillTabs(
            [
                ("Table", "table", self._wrap(table_card)),
                ("Profiling", "chart", self._wrap(self._profile_table)),
                ("Plots", "chart", self._wrap(self._plots_host)),
                ("Quality Issues", "missing", self._wrap(self._issues_table)),
            ]
        )
        self._layout.addWidget(self._tabs, 1)

        self._empty = EmptyState(
            "No dataset loaded",
            "Import a CSV, Excel, Parquet, or JSON file to explore your data.",
            icon_name="import",
        )
        self._empty_import_btn = QPushButton("Import dataset")
        self._empty_import_btn.setObjectName("PrimaryButton")
        constrain_primary_button(self._empty_import_btn)
        self._empty.set_action(self._empty_import_btn)
        self._error_label = QLabel("")
        self._error_label.setObjectName("ErrorBanner")
        self._error_label.hide()
        self._layout.addWidget(self._error_label)
        self._layout.addWidget(self._empty, 1)
        self._overlay = LoadingOverlay(parent=self)

    def _wrap(self, widget: QWidget) -> QWidget:
        box = QWidget()
        layout = QVBoxLayout(box)
        layout.setContentsMargins(0, 0, 0, 0)
        widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(widget, 1)
        return box

    def _make_profile_table(self) -> QTableWidget:
        table = QTableWidget()
        table.setColumnCount(10)
        table.setHorizontalHeaderLabels(
            ["Column", "Type", "Missing", "Unique", "Min", "Max", "Mean", "Median", "Std", "Top values"]
        )
        configure_table_header(
            table.horizontalHeader(),
            contents_cols=(0, 1, 2, 3),
            stretch_cols=(9,),
        )
        table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        return table

    def _make_issues_table(self) -> QTableWidget:
        table = QTableWidget()
        table.setColumnCount(5)
        table.setHorizontalHeaderLabels(["Severity", "Type", "Column", "Details", "Action"])
        configure_table_header(
            table.horizontalHeader(),
            contents_cols=(0, 1, 2, 4),
            stretch_cols=(3,),
        )
        table.setAlternatingRowColors(True)
        table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
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
            self._clear_profile_tables()
            self._update_row_count_label()
            return
        self._empty.hide()
        self._tabs.show()
        self._model.set_dataframe(dataset.dataframe)
        self._column_filter.blockSignals(True)
        self._column_filter.clear()
        self._column_filter.addItem("All columns")
        self._column_filter.addItems([str(c) for c in dataset.dataframe.columns])
        self._column_filter.blockSignals(False)
        self._search.blockSignals(True)
        self._search.clear()
        self._search.blockSignals(False)
        self._update_row_count_label()
        self._clear_profile_tables()

    def _filter_column_name(self) -> str | None:
        if self._column_filter.currentIndex() <= 0:
            return None
        return self._column_filter.currentText()

    def _on_filter_changed(self, *_args) -> None:
        self._model.apply_filter(self._search.text(), column=self._filter_column_name())
        self._update_row_count_label()

    def _update_row_count_label(self) -> None:
        shown = self._model.visible_rows
        total = self._model.total_rows
        self._row_count_label.setText(f"Showing {shown:,} of {total:,} rows")

    def _clear_profile_tables(self) -> None:
        self._profile_table.setRowCount(0)
        self._issues_table.setRowCount(0)
        self._refresh_plots()

    def _refresh_plots(self) -> None:
        while self._plots_layout.count():
            item = self._plots_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        if self._dataset is None:
            ph = QLabel("Import a dataset to see plots.")
            ph.setObjectName("Breadcrumb")
            self._plots_layout.addWidget(ph)
            return
        try:
            from ml_studio.gui.charts import (
                chart_panel,
                correlation_heatmap,
                histogram_widget,
                missingness_widget,
            )

            df = self._dataset.dataframe
            num_cols = list(df.select_dtypes(include="number").columns)
            widgets = [missingness_widget(df)]
            if num_cols:
                widgets.append(histogram_widget(df[num_cols[0]], title=f"Histogram · {num_cols[0]}"))
            widgets.append(correlation_heatmap(df))
            self._plots_layout.addWidget(chart_panel(*widgets), 1)
        except Exception as exc:
            self._plots_layout.addWidget(QLabel(f"Plots unavailable: {exc}"))

    def apply_profile_result(self, result) -> None:
        """Apply async profiling worker output to the Profiling / Quality tabs."""
        if isinstance(result, dict):
            profile = result.get("profile")
            issues = result.get("issues") or []
        else:
            # Legacy: worker returned profile only
            profile = result
            issues = []
            if self._dataset is not None:
                from ml_studio.core.profiling import detect_quality_issues

                issues = detect_quality_issues(self._dataset)

        if profile is None:
            return

        self._profile_table.setRowCount(len(profile.columns))
        for i, col in enumerate(profile.columns):
            self._profile_table.setItem(i, 0, QTableWidgetItem(col.name))
            self._profile_table.setItem(i, 1, QTableWidgetItem(col.dtype))
            self._profile_table.setItem(i, 2, QTableWidgetItem(f"{col.missing_pct:.1f}%"))
            self._profile_table.setItem(i, 3, QTableWidgetItem(str(col.unique)))
            self._profile_table.setItem(
                i, 4, QTableWidgetItem("" if col.min is None else str(col.min))
            )
            self._profile_table.setItem(
                i, 5, QTableWidgetItem("" if col.max is None else str(col.max))
            )
            self._profile_table.setItem(
                i,
                6,
                QTableWidgetItem("" if col.mean is None else f"{col.mean:.4g}"),
            )
            self._profile_table.setItem(
                i,
                7,
                QTableWidgetItem("" if col.median is None else f"{col.median:.4g}"),
            )
            self._profile_table.setItem(
                i,
                8,
                QTableWidgetItem("" if col.std is None else f"{col.std:.4g}"),
            )
            top = getattr(col, "top_values", None) or []
            top_text = ", ".join(str(v) for v in top) if top else "—"
            self._profile_table.setItem(i, 9, QTableWidgetItem(top_text))

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
            fix_btn.clicked.connect(self._goto_prepare)
            self._issues_table.setCellWidget(i, 4, fix_btn)
        self._refresh_plots()

    def _goto_prepare(self) -> None:
        win = self.window()
        if hasattr(win, "_navigate"):
            win._navigate("prepare")

    def _populate_profile(self) -> None:
        """Deprecated sync path — kept for tests; prefer apply_profile_result."""
        if not self._dataset:
            return
        from ml_studio.core.profiling import detect_quality_issues, profile_dataset

        profile = profile_dataset(self._dataset)
        issues = detect_quality_issues(self._dataset)
        self.apply_profile_result({"profile": profile, "issues": issues})
