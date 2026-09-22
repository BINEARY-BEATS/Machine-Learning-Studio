"""Editable column schema table with role pickers."""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ml_studio.core.schema import ColumnRole
from ml_studio.gui.layout_utils import configure_table_header, fill_widget


class SchemaEditor(QWidget):
    role_changed = pyqtSignal(str, str)  # col_name, new_role

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)

        self._table = QTableWidget(0, 4)
        self._table.setHorizontalHeaderLabels(["Column", "Kind", "Sample Values", "Role"])
        configure_table_header(
            self._table.horizontalHeader(),
            contents_cols=(0, 1, 3),
            stretch_cols=(2,),
        )
        self._table.setAlternatingRowColors(True)
        self._table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self._table.setShowGrid(False)
        self._table.verticalHeader().hide()
        self._table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        fill_widget(self._layout, self._table)

    def set_schema(self, columns: list, overrides: dict = None):
        overrides = overrides or {}
        self._table.setRowCount(len(columns))

        for i, col in enumerate(columns):
            name_item = QTableWidgetItem(col.name)
            name_item.setFlags(Qt.ItemFlag.ItemIsEnabled)

            kind_item = QTableWidgetItem(col.kind.value)
            kind_item.setFlags(Qt.ItemFlag.ItemIsEnabled)

            samples = ", ".join(str(s) for s in col.sample_values[:3])
            sample_item = QTableWidgetItem(samples)
            sample_item.setFlags(Qt.ItemFlag.ItemIsEnabled)

            role_combo = QComboBox()
            for role in ColumnRole:
                role_combo.addItem(role.value.capitalize(), role.value)

            current_role = overrides.get(col.name, col.role.value)
            idx = role_combo.findData(current_role)
            if idx >= 0:
                role_combo.setCurrentIndex(idx)

            role_combo.currentIndexChanged.connect(
                lambda _i, c=col.name, combo=role_combo: self._on_role_changed(
                    c, combo.currentData() or ""
                )
            )

            self._table.setItem(i, 0, name_item)
            self._table.setItem(i, 1, kind_item)
            self._table.setItem(i, 2, sample_item)
            self._table.setCellWidget(i, 3, role_combo)

    def _on_role_changed(self, col_name: str, new_role: str) -> None:
        self.role_changed.emit(col_name, new_role)
