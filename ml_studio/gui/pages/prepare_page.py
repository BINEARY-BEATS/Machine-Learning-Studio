"""Visual preprocessing pipeline builder."""

from __future__ import annotations

import yaml

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ml_studio.core.pipeline import Pipeline
from ml_studio.core.recipes import RECIPES_DIR, apply_recipe, get_available_recipes
from ml_studio.core.schema import infer_schema
from ml_studio.gui.dialogs.step_config import StepConfigDialog
from ml_studio.gui.dialogs.transform_picker import TransformPickerDialog
from ml_studio.gui.pages.base_page import BasePage
from ml_studio.gui.widgets.empty_state import EmptyState
from ml_studio.gui.widgets.pipeline_step import PipelineStepWidget
from ml_studio.gui.widgets.schema_editor import SchemaEditor
from ml_studio.gui.widgets.tag_chip import TagChip
from ml_studio.app.theme import ThemeMode


class PreparePage(BasePage):
    """Build schema roles + preprocessing pipeline for the current session."""

    preview_requested = pyqtSignal()

    def __init__(self, container, parent=None):
        self.dataset = None
        self.pipeline = Pipeline()
        self.schema_overrides: dict[str, str] = {}
        self._schema_columns = []
        self._mode = ThemeMode.LIGHT
        super().__init__(container, parent)

    def _build_ui(self) -> None:
        title = QLabel("Prepare Data")
        title.setObjectName("PageTitle")
        self._layout.addWidget(title)

        desc = QLabel(
            "Configure column roles and build preprocessing steps applied before training."
        )
        desc.setObjectName("TextMuted")
        self._layout.addWidget(desc)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)

        schema_panel = QWidget()
        schema_layout = QVBoxLayout(schema_panel)
        schema_layout.setContentsMargins(0, 0, 0, 0)

        schema_header = QLabel("Dataset Schema")
        schema_header.setStyleSheet("font-weight: bold; font-size: 14px;")
        schema_layout.addWidget(schema_header)

        self.schema_editor = SchemaEditor()
        self.schema_editor.role_changed.connect(self._on_role_changed)
        schema_layout.addWidget(self.schema_editor, 1)

        splitter.addWidget(schema_panel)

        pipeline_panel = QWidget()
        pipeline_layout = QVBoxLayout(pipeline_panel)
        pipeline_layout.setContentsMargins(0, 0, 0, 0)

        pipe_header_row = QHBoxLayout()
        pipe_header = QLabel("Pipeline Steps")
        pipe_header.setStyleSheet("font-weight: bold; font-size: 14px;")
        pipe_header_row.addWidget(pipe_header)

        self._status = TagChip("0 steps", "info")
        pipe_header_row.addWidget(self._status)
        pipe_header_row.addStretch()

        self._preview_btn = QPushButton("Preview")
        self._preview_btn.clicked.connect(self._on_preview)
        pipe_header_row.addWidget(self._preview_btn)

        self._save_btn = QPushButton("Save Pipeline")
        self._save_btn.setObjectName("PrimaryButton")
        self._save_btn.clicked.connect(self._on_save_pipeline)
        from ml_studio.gui.layout_utils import constrain_primary_button

        constrain_primary_button(self._save_btn)
        pipe_header_row.addWidget(self._save_btn)

        pipeline_layout.addLayout(pipe_header_row)

        self._pipeline_list = QListWidget()
        self._pipeline_list.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self._pipeline_list.model().rowsMoved.connect(self._on_rows_moved)
        pipeline_layout.addWidget(self._pipeline_list, 1)

        self._empty = EmptyState(
            "Build your first pipeline step",
            "Add imputation, scaling, encoding, or feature engineering steps.",
            icon_name="clean",
        )
        pipeline_layout.addWidget(self._empty, 1)

        action_row = QHBoxLayout()

        self._add_btn = QPushButton("Add Step")
        self._add_btn.setObjectName("PrimaryButton")
        constrain_primary_button(self._add_btn)
        self._add_btn.clicked.connect(self._add_step)
        action_row.addWidget(self._add_btn)

        self._recipe_combo = QComboBox()
        self._recipe_combo.addItem("Load Recipe...")
        self._recipe_combo.addItems(get_available_recipes())
        self._recipe_combo.currentIndexChanged.connect(self._on_recipe_selected)
        action_row.addWidget(self._recipe_combo)

        self._clear_btn = QPushButton("Clear")
        self._clear_btn.setObjectName("GhostButton")
        self._clear_btn.clicked.connect(self._clear_pipeline)
        action_row.addWidget(self._clear_btn)
        action_row.addStretch()

        pipeline_layout.addLayout(action_row)

        splitter.addWidget(pipeline_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([480, 520])

        self._layout.addWidget(splitter, 1)
        self._refresh_ui()

    def set_theme_mode(self, mode: ThemeMode) -> None:
        self._mode = mode
        self._empty.set_theme_mode(mode)

    def set_dataset(self, dataset, schema_overrides: dict | None = None) -> None:
        """Hydrate schema editor from the session dataset (called after import)."""
        self.dataset = dataset
        if schema_overrides is not None:
            self.schema_overrides = dict(schema_overrides)
        if dataset is None:
            self._schema_columns = []
            self.schema_editor.set_schema([])
            self._refresh_ui()
            return
        schema = infer_schema(dataset.dataframe)
        self._schema_columns = schema.columns
        self.schema_editor.set_schema(schema.columns, self.schema_overrides)
        self._refresh_ui()

    def set_project(self, project) -> None:
        """Backward-compatible entry used by tests; prefers project.dataset."""
        overrides = getattr(project, "schema_overrides", None) or {}
        self.schema_overrides = dict(overrides)
        pipeline = getattr(project, "pipeline", None)
        if pipeline is not None:
            self.pipeline = pipeline
        dataset = getattr(project, "dataset", None)
        self.set_dataset(dataset, self.schema_overrides)

    def _column_names(self) -> list[str]:
        if self.dataset is None:
            return []
        return list(self.dataset.dataframe.columns)

    def _schema_columns_dict(self) -> dict:
        """Build {col: {role, kind}} for recipe expansion."""
        result = {}
        for col in self._schema_columns:
            role = self.schema_overrides.get(col.name, col.role.value)
            result[col.name] = {"role": role, "kind": col.kind.value}
        if not result and self.dataset is not None:
            for name in self.dataset.dataframe.columns:
                result[str(name)] = {
                    "role": self.schema_overrides.get(str(name), "feature"),
                    "kind": "numeric",
                }
        return result

    def _show_toast(self, message: str, variant: str = "default") -> None:
        win = self.window()
        toast = getattr(win, "_toast", None)
        if toast is not None and hasattr(toast, "show_message"):
            toast.show_message(message, variant=variant)

    def _on_role_changed(self, col_name: str, new_role: str) -> None:
        self.schema_overrides[col_name] = new_role
        if self.dataset is not None:
            if new_role == "target":
                self.dataset.target_column = col_name
            elif getattr(self.dataset, "target_column", None) == col_name:
                self.dataset.target_column = None
        self._mark_dirty()
        self._show_toast(f"Column '{col_name}' role updated to {new_role}", "success")

    def _mark_dirty(self) -> None:
        win = self.window()
        controller = getattr(win, "controller", None)
        if controller is not None and hasattr(controller, "mark_project_dirty"):
            controller.mark_project_dirty()

    def _add_step(self) -> None:
        picker = TransformPickerDialog(self)
        if picker.exec() != TransformPickerDialog.DialogCode.Accepted:
            return
        if not picker.selected_transform:
            return
        columns = self._column_names()
        config = StepConfigDialog(picker.selected_transform, {}, columns, self)
        if config.exec() != StepConfigDialog.DialogCode.Accepted:
            return
        step_obj = config.transform_class(**config.final_params)
        step_obj.enabled = True
        self.pipeline.add(step_obj)
        self._mark_dirty()
        self._refresh_ui()

    def _edit_step(self, index: int) -> None:
        step = self.pipeline.steps[index]
        columns = self._column_names()
        config = StepConfigDialog(step.__class__.__name__, step.params, columns, self)
        if config.exec() != StepConfigDialog.DialogCode.Accepted:
            return
        enabled = getattr(step, "enabled", True)
        new_step = config.transform_class(**config.final_params)
        new_step.enabled = enabled
        self.pipeline.steps[index] = new_step
        self._mark_dirty()
        self._refresh_ui()

    def _toggle_step(self, index: int, enabled: bool) -> None:
        if 0 <= index < len(self.pipeline.steps):
            self.pipeline.steps[index].enabled = enabled
            self._mark_dirty()

    def _move_up(self, index: int) -> None:
        if index > 0:
            step = self.pipeline.steps.pop(index)
            self.pipeline.steps.insert(index - 1, step)
            self._mark_dirty()
            self._refresh_ui()

    def _move_down(self, index: int) -> None:
        if index < len(self.pipeline.steps) - 1:
            step = self.pipeline.steps.pop(index)
            self.pipeline.steps.insert(index + 1, step)
            self._mark_dirty()
            self._refresh_ui()

    def _delete_step(self, index: int) -> None:
        self.pipeline.steps.pop(index)
        self._mark_dirty()
        self._refresh_ui()

    def _on_rows_moved(
        self, sourceParent, sourceStart, sourceEnd, destinationParent, destinationRow
    ) -> None:
        step = self.pipeline.steps.pop(sourceStart)
        if destinationRow > sourceStart:
            destinationRow -= 1
        self.pipeline.steps.insert(destinationRow, step)
        self._mark_dirty()
        self._refresh_ui()

    def _on_preview(self) -> None:
        if self.dataset is None:
            self._show_toast("Load a dataset first to preview.", "warning")
            return
        self.preview_requested.emit()

    def _on_save_pipeline(self) -> None:
        self._mark_dirty()
        n = len(self.pipeline.steps)
        self._show_toast(
            f"Pipeline ready for training ({n} step{'s' if n != 1 else ''}). "
            "Use File → Save (Ctrl+S) to write the project.",
            "success",
        )

    def _on_recipe_selected(self, index: int) -> None:
        if index == 0:
            return

        recipe_name = self._recipe_combo.currentText()
        self._recipe_combo.setCurrentIndex(0)

        recipe_path = RECIPES_DIR / f"{recipe_name}.yaml"
        try:
            desc = "Apply this recipe?"
            if recipe_path.exists():
                with open(recipe_path, encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
                desc = data.get("metadata", {}).get("description", desc)

            reply = QMessageBox.question(
                self,
                "Apply Recipe",
                f"Apply the '{recipe_name}' recipe?\n\n{desc}\n\n"
                "This will replace your current pipeline.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

            if self.dataset is None and not self._schema_columns:
                self._show_toast("Load a dataset before applying a recipe.", "warning")
                return

            self.pipeline = apply_recipe(recipe_name, self._schema_columns_dict())
            for step in self.pipeline.steps:
                if not hasattr(step, "enabled"):
                    step.enabled = True
            self._mark_dirty()
            self._refresh_ui()
            self._show_toast(f"Applied recipe: {recipe_name}", "success")
        except Exception as exc:
            self._show_toast(f"Error applying recipe: {exc}", "danger")

    def _clear_pipeline(self) -> None:
        self.pipeline = Pipeline()
        self._mark_dirty()
        self._refresh_ui()

    def _refresh_ui(self) -> None:
        self._pipeline_list.clear()

        num_steps = len(self.pipeline.steps)
        self._status.setText(f"{num_steps} steps")

        if num_steps == 0:
            self._pipeline_list.hide()
            self._empty.show()
            return

        self._pipeline_list.show()
        self._empty.hide()

        for i, step in enumerate(self.pipeline.steps):
            item = QListWidgetItem(self._pipeline_list)
            widget = PipelineStepWidget(step, i, len(self.pipeline.steps))
            widget.move_up_requested.connect(lambda idx=i: self._move_up(idx))
            widget.move_down_requested.connect(lambda idx=i: self._move_down(idx))
            widget.edit_requested.connect(lambda idx=i: self._edit_step(idx))
            widget.delete_requested.connect(lambda idx=i: self._delete_step(idx))
            widget.toggle_requested.connect(
                lambda enabled, idx=i: self._toggle_step(idx, enabled)
            )
            item.setSizeHint(widget.sizeHint())
            self._pipeline_list.setItemWidget(item, widget)
