from PyQt6.QtWidgets import QDialog, QVBoxLayout, QListWidget, QLineEdit, QPushButton, QHBoxLayout, QListWidgetItem
from PyQt6.QtCore import Qt
from ml_studio.transforms.registry import list_all

class TransformPickerDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add Step")
        self.setMinimumSize(400, 500)
        self.selected_transform = None
        self._all_transforms = list_all()
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        
        self._search = QLineEdit()
        self._search.setPlaceholderText("Search transforms...")
        self._search.textChanged.connect(self._filter_list)
        layout.addWidget(self._search)
        
        self._list = QListWidget()
        self._list.itemDoubleClicked.connect(self._on_accept)
        layout.addWidget(self._list)
        
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)
        
        self._add_btn = QPushButton("Add")
        self._add_btn.setObjectName("PrimaryButton")
        self._add_btn.clicked.connect(self._on_accept)
        btn_layout.addWidget(self._add_btn)
        
        layout.addLayout(btn_layout)
        
        self._populate_list()
        
        if self._list.count() > 0:
            self._list.setCurrentRow(0)

    def _populate_list(self):
        for t in self._all_transforms:
            item = QListWidgetItem(f"{t['name']} ({t['category']})")
            item.setData(Qt.ItemDataRole.UserRole, t['name'])
            item.setToolTip(t['doc'] or "")
            self._list.addItem(item)

    def _filter_list(self, text: str):
        text = text.lower()
        for i in range(self._list.count()):
            item = self._list.item(i)
            item.setHidden(text not in item.text().lower())

    def _on_accept(self):
        selected = self._list.selectedItems()
        if selected:
            self.selected_transform = selected[0].data(Qt.ItemDataRole.UserRole)
            self.accept()
