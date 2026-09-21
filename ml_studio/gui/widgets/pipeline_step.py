from PyQt6.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QLabel, QCheckBox
from PyQt6.QtCore import pyqtSignal, Qt

from ml_studio.gui.widgets.icon_button import IconButton
from ml_studio.gui.widgets.card import Card


class PipelineStepWidget(QWidget):
    move_up_requested = pyqtSignal()
    move_down_requested = pyqtSignal()
    edit_requested = pyqtSignal()
    delete_requested = pyqtSignal()
    toggle_requested = pyqtSignal(bool)

    def __init__(self, step, index: int, total_steps: int, parent=None):
        super().__init__(parent)
        self.step = step
        self.index = index
        self._build_ui(total_steps)

    def _build_ui(self, total_steps: int):
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 4)

        self._card = Card()
        card_layout = QHBoxLayout()
        card_layout.setContentsMargins(8, 8, 8, 8)

        handle = QLabel("::")
        handle.setObjectName("TextMuted")
        handle.setCursor(Qt.CursorShape.OpenHandCursor)
        card_layout.addWidget(handle)

        self._enabled = QCheckBox()
        self._enabled.setChecked(getattr(self.step, "enabled", True))
        self._enabled.toggled.connect(self.toggle_requested.emit)
        card_layout.addWidget(self._enabled)

        text_layout = QVBoxLayout()
        name_label = QLabel(self.step.__class__.__name__)
        name_label.setObjectName("BaseText")
        name_label.setStyleSheet("font-weight: bold;")
        text_layout.addWidget(name_label)

        params_str = ", ".join(f"{k}={v}" for k, v in self.step.params.items())
        if not params_str:
            params_str = "No parameters"
        param_label = QLabel(params_str)
        param_label.setObjectName("TextMuted")
        text_layout.addWidget(param_label)

        card_layout.addLayout(text_layout)
        card_layout.addStretch()

        self._up_btn = IconButton("↑", tooltip="Move Up")
        self._up_btn.button().clicked.connect(self.move_up_requested.emit)
        self._up_btn.set_enabled(self.index > 0)
        card_layout.addWidget(self._up_btn)

        self._down_btn = IconButton("↓", tooltip="Move Down")
        self._down_btn.button().clicked.connect(self.move_down_requested.emit)
        self._down_btn.set_enabled(self.index < total_steps - 1)
        card_layout.addWidget(self._down_btn)

        self._edit_btn = IconButton("✎", tooltip="Edit")
        self._edit_btn.button().clicked.connect(self.edit_requested.emit)
        card_layout.addWidget(self._edit_btn)

        self._delete_btn = IconButton("✕", tooltip="Delete")
        self._delete_btn.button().clicked.connect(self.delete_requested.emit)
        card_layout.addWidget(self._delete_btn)

        self._card.add_layout(card_layout)
        self._layout.addWidget(self._card)
