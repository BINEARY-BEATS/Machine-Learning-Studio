"""Command palette (Ctrl+K) with fuzzy search."""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QDialog, QLineEdit, QListWidget, QListWidgetItem, QVBoxLayout

from ml_studio.app.theme import ThemeMode, apply_theme


class CommandPalette(QDialog):
    command_selected = pyqtSignal(str)

    COMMANDS = [
        ("New Project", "new_project"),
        ("Open Project", "open_project"),
        ("Save", "save"),
        ("Import Dataset", "import_dataset"),
        ("Profile Dataset", "profile_dataset"),
        ("Prepare Dataset", "prepare_dataset"),
        ("Train Model", "train_model"),
        ("Run AutoML", "run_automl"),
        ("Evaluate Model", "evaluate_model"),
        ("Predict", "predict"),
        ("Open Model Registry", "model_registry"),
        ("Settings", "settings"),
    ]

    def __init__(self, parent=None, mode: ThemeMode = ThemeMode.LIGHT):
        super().__init__(parent)
        self.setWindowTitle("Command Palette")
        self.setModal(True)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        self._search = QLineEdit()
        self._search.setPlaceholderText("Type a command…")
        self._list = QListWidget()
        self._populate("")
        self._search.textChanged.connect(self._populate)
        self._list.itemActivated.connect(self._on_select)
        self._search.returnPressed.connect(self._on_search_return)
        layout.addWidget(self._search)
        layout.addWidget(self._list)
        self.resize(520, 360)
        self._search.setFocus()
        if parent and hasattr(parent, "styleSheet"):
            self.setStyleSheet(parent.styleSheet())
        else:
            from PyQt6.QtWidgets import QApplication

            app_widget = QApplication.instance()
            if app_widget:
                apply_theme(app_widget, mode)

    def _populate(self, query: str) -> None:
        self._list.clear()
        q = query.lower().strip()
        for label, cmd_id in self.COMMANDS:
            if self.fuzzy_match(q, label):
                item = QListWidgetItem(label)
                item.setData(Qt.ItemDataRole.UserRole, cmd_id)
                self._list.addItem(item)
        if self._list.count():
            self._list.setCurrentRow(0)

    def _on_select(self, item: QListWidgetItem) -> None:
        self.command_selected.emit(item.data(Qt.ItemDataRole.UserRole))
        self.accept()
        
    def _on_search_return(self) -> None:
        if self._list.count() > 0:
            self._on_select(self._list.item(0))

    @staticmethod
    def fuzzy_match(query: str, text: str) -> bool:
        if not query:
            return True
        ti = 0
        for ch in query:
            idx = text.lower().find(ch, ti)
            if idx < 0:
                return False
            ti = idx + 1
        return True
