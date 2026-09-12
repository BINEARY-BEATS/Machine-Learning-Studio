"""Full training wizard with stepper and configuration panels."""

from __future__ import annotations

from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ml_studio.core.training.registry import get_models_for_task
from ml_studio.core.training.task import TaskType
from ml_studio.core.training.trainer import TrainingConfig
from ml_studio.gui.pages.base_page import BasePage
from ml_studio.gui.widgets.card import Card
from ml_studio.gui.widgets.stepper import Stepper


class TrainPage(BasePage):
    STEPS = [
        "Task",
        "Dataset",
        "Features",
        "Preprocess",
        "Split",
        "Models",
        "Tune",
        "Train",
        "Eval",
        "Save",
    ]

    def __init__(self, container, parent=None):
        self._columns: list[str] = []
        self._dataset_name = "No dataset"
        super().__init__(container, parent)

    def _build_ui(self) -> None:
        title = QLabel("Train Model")
        title.setObjectName("PageTitle")
        self._layout.addWidget(title)

        self._stepper = Stepper(self.STEPS)
        self._stepper.step_clicked.connect(self._goto_step)
        self._layout.addWidget(self._stepper)

        nav = QHBoxLayout()
        self._back_btn = QPushButton("Back")
        self._back_btn.setObjectName("GhostButton")
        self._next_btn = QPushButton("Next")
        self._next_btn.setObjectName("PrimaryButton")
        self._back_btn.clicked.connect(self._prev_step)
        self._next_btn.clicked.connect(self._next_step)
        nav.addWidget(self._back_btn)
        nav.addWidget(self._next_btn)
        nav.addStretch()
        self._layout.addLayout(nav)

        self._stack = QStackedWidget()
        self._task_combo = QComboBox()
        self._task_combo.addItems([t.value for t in TaskType])
        self._dataset_label = QLabel("Import a dataset on the Data page.")
        self._target_combo = QComboBox()
        self._feature_list = QListWidget()
        self._feature_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        self._prepare_label = QLabel("Configure preprocessing on the Prepare page.")
        self._test_spin = QDoubleSpinBox()
        self._test_spin.setRange(0.05, 0.5)
        self._test_spin.setSingleStep(0.05)
        self._test_spin.setValue(0.2)
        self._cv_spin = QSpinBox()
        self._cv_spin.setRange(2, 10)
        self._cv_spin.setValue(5)
        self._model_combo = QComboBox()
        self._tune_combo = QComboBox()
        self._tune_combo.addItems(["None", "Grid search", "Optuna"])
        self._summary = QLabel("")
        self._summary.setWordWrap(True)
        self._train_btn = QPushButton("Start Training")
        self._train_btn.setObjectName("PrimaryButton")
        self._eval_label = QLabel("Evaluation runs automatically after training.")
        self._save_label = QLabel("Models are saved to the registry after training.")

        panels = [
            self._form_panel("Task type", [("Task", self._task_combo)]),
            self._wrap(self._dataset_label),
            self._form_panel("Target & features", [("Target", self._target_combo)]),
            self._wrap(self._prepare_label),
            self._form_panel("Split & CV", [("Test size", self._test_spin), ("CV folds", self._cv_spin)]),
            self._form_panel("Model", [("Algorithm", self._model_combo)]),
            self._form_panel("Tuning", [("Method", self._tune_combo)]),
            self._train_panel(),
            self._wrap(self._eval_label),
            self._wrap(self._save_label),
        ]
        for panel in panels:
            self._stack.addWidget(panel)
        self._layout.addWidget(self._stack, 1)
        self._task_combo.currentTextChanged.connect(self._update_models)
        self._model_combo.currentTextChanged.connect(self._refresh_summary)
        self._target_combo.currentTextChanged.connect(self._refresh_summary)
        self._feature_list.itemSelectionChanged.connect(self._refresh_summary)
        self._update_models()

    def _form_panel(self, title: str, rows: list[tuple[str, QWidget]]) -> QWidget:
        card = Card(title)
        form = QFormLayout()
        for label, widget in rows:
            form.addRow(f"{label}:", widget)
        card.add_layout(form)
        box = QWidget()
        layout = QVBoxLayout(box)
        layout.addWidget(card)
        if title == "Target & features":
            layout.addWidget(self._feature_list)
        layout.addStretch()
        return box

    def _wrap(self, widget: QWidget) -> QWidget:
        box = QWidget()
        layout = QVBoxLayout(box)
        layout.addWidget(widget)
        layout.addStretch()
        return box

    def _train_panel(self) -> QWidget:
        card = Card("Training summary")
        card.add_widget(self._summary)
        card.add_widget(self._train_btn)
        box = QWidget()
        layout = QVBoxLayout(box)
        layout.addWidget(card)
        layout.addStretch()
        return box

    def set_dataset(self, name: str, columns: list[str], suggested_target: str, suggested_task: TaskType) -> None:
        self._columns = columns
        self._dataset_name = name
        self._dataset_label.setText(f"{name} — {len(columns):,} columns loaded")
        self._target_combo.clear()
        self._target_combo.addItems(columns)
        self._feature_list.clear()
        for col in columns:
            item = QListWidgetItem(col)
            self._feature_list.addItem(item)
            item.setSelected(col != suggested_target)
        if suggested_target in columns:
            self._target_combo.setCurrentText(suggested_target)
        idx = self._task_combo.findText(suggested_task.value)
        if idx >= 0:
            self._task_combo.setCurrentIndex(idx)
        self._update_models()
        self._refresh_summary()

    def build_config(self) -> TrainingConfig | None:
        target = self._target_combo.currentText()
        features = [
            self._feature_list.item(i).text()
            for i in range(self._feature_list.count())
            if self._feature_list.item(i).isSelected() and self._feature_list.item(i).text() != target
        ]
        if not target or not features:
            return None
        from ml_studio.core.training.registry import MODEL_REGISTRY

        model_name = self._model_combo.currentText()
        model_id = next((k for k, v in MODEL_REGISTRY.items() if v.name == model_name), "linear_regression")
        return TrainingConfig(
            task=TaskType(self._task_combo.currentText()),
            model_id=model_id,
            target_column=target,
            feature_columns=features,
            test_size=float(self._test_spin.value()),
            cv_splits=int(self._cv_spin.value()),
        )

    def _update_models(self) -> None:
        self._model_combo.clear()
        task = TaskType(self._task_combo.currentText())
        for meta in get_models_for_task(task):
            self._model_combo.addItem(meta.name)
        self._refresh_summary()

    def _refresh_summary(self) -> None:
        config = self.build_config()
        if not config:
            self._summary.setText("Select target and at least one feature column.")
            return
        self._summary.setText(
            f"Task: {config.task.value}\n"
            f"Dataset: {self._dataset_name}\n"
            f"Target: {config.target_column}\n"
            f"Features: {len(config.feature_columns)}\n"
            f"Model: {self._model_combo.currentText()}\n"
            f"Test split: {config.test_size:.0%}  ·  CV: {config.cv_splits}-fold"
        )

    def _goto_step(self, index: int) -> None:
        self._stack.setCurrentIndex(index)
        self._stepper.set_current(index)

    def _next_step(self) -> None:
        idx = min(self._stepper.current_index() + 1, self._stepper.step_count() - 1)
        self._goto_step(idx)

    def _prev_step(self) -> None:
        idx = max(self._stepper.current_index() - 1, 0)
        self._goto_step(idx)

    def get_target_column(self) -> str:
        return self._target_combo.currentText()

    def get_task(self) -> TaskType:
        return TaskType(self._task_combo.currentText())

    def get_model_name(self) -> str:
        return self._model_combo.currentText()
