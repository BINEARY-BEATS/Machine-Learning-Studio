from PyQt6.QtWidgets import QTableWidget, QTableWidgetItem, QComboBox, QVBoxLayout, QWidget, QHeaderView
from PyQt6.QtCore import Qt, pyqtSignal

from ml_studio.core.schema import ColumnRole

class SchemaEditor(QWidget):
    role_changed = pyqtSignal(str, str) # col_name, new_role

    def __init__(self, parent=None):
        super().__init__(parent)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        
        self._table = QTableWidget(0, 4)
        self._table.setHorizontalHeaderLabels(["Column", "Kind", "Sample Values", "Role"])
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self._table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self._table.setAlternatingRowColors(True)
        self._table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self._table.setShowGrid(False)
        self._table.verticalHeader().hide()
        
        self._layout.addWidget(self._table)

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
                
            # Connect the combo box
            role_combo.currentTextChanged.connect(
                lambda text, c=col.name: self._on_role_changed(c, text.lower())
            )
            
            self._table.setItem(i, 0, name_item)
            self._table.setItem(i, 1, kind_item)
            self._table.setItem(i, 2, sample_item)
            self._table.setCellWidget(i, 3, role_combo)

    def _on_role_changed(self, col_name: str, new_role: str):
        self.role_changed.emit(col_name, new_role)
