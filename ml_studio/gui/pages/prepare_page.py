"""Visual preprocessing pipeline builder."""

from __future__ import annotations

from PyQt6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QListWidget, QPushButton, QVBoxLayout

from ml_studio.core.pipeline import Pipeline
from ml_studio.transforms.registry import get as get_transform, list_all as list_transforms
from ml_studio.gui.pages.base_page import BasePage
from ml_studio.gui.widgets.card import Card
from ml_studio.gui.widgets.empty_state import EmptyState
from ml_studio.gui.widgets.tag_chip import TagChip


class PreparePage(BasePage):
    def __init__(self, container, parent=None):
        self.pipeline = Pipeline()
        super().__init__(container, parent)

    def _build_ui(self) -> None:
        title = QLabel("Prepare Data")
        title.setObjectName("PageTitle")
        self._layout.addWidget(title)
        self._layout.addWidget(
            QLabel("Build preprocessing steps applied before training. Order matters — top to bottom.")
        )

        card = Card("Pipeline steps")
        self._pipeline_list = QListWidget()
        self._pipeline_list.setMinimumHeight(240)
        card.add_widget(self._pipeline_list)
        self._layout.addWidget(card)

        row = QHBoxLayout()
        self._transform_combo = QComboBox()
        self._transform_combo.addItems([t['name'] for t in list_transforms()])
        self._add_btn = QPushButton("Add Step")
        self._add_btn.setObjectName("PrimaryButton")
        self._clear_btn = QPushButton("Clear Pipeline")
        self._clear_btn.setObjectName("GhostButton")
        self._add_btn.clicked.connect(self._add_step)
        self._clear_btn.clicked.connect(self._clear_pipeline)
        row.addWidget(self._transform_combo, 1)
        row.addWidget(self._add_btn)
        row.addWidget(self._clear_btn)
        self._layout.addLayout(row)

        self._status = TagChip("0 steps", "info")
        self._layout.addWidget(self._status)
        self._empty = EmptyState(
            "Empty pipeline",
            "Add imputation, scaling, encoding, or feature engineering steps.",
            icon_name="clean",
        )
        self._layout.addWidget(self._empty)
        self._layout.addStretch()

    def _add_step(self) -> None:
        name = self._transform_combo.currentText()
        step = get_transform(name)
        self.pipeline.add(step)
        self._pipeline_list.addItem(f"{len(self.pipeline.nodes)}. {step.name} — {step.description}")
        self._status.setText(f"{len(self.pipeline.nodes)} steps")
        self._status.set_variant("success")
        self._empty.hide()

    def _clear_pipeline(self) -> None:
        self.pipeline = Pipeline()
        self._pipeline_list.clear()
        self._status.setText("0 steps")
        self._status.set_variant("info")
        self._empty.show()
