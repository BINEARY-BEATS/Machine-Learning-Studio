"""Full training wizard with stepper and configuration panels."""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ml_studio.core.training.registry import MODEL_REGISTRY, get_models_for_task
from ml_studio.core.training.task import TaskType
from ml_studio.core.training.trainer import TrainingConfig
from ml_studio.gui.layout_utils import constrain_primary_button, configure_table_header
from ml_studio.gui.pages.base_page import BasePage
from ml_studio.gui.widgets.card import Card
from ml_studio.gui.widgets.stepper import Stepper


class TrainPage(BasePage):
    STEPS = [
        "Task",
        "Data",
        "Model",
        "Tune",
        "Run",
    ]
    train_requested = pyqtSignal()
    automl_requested = pyqtSignal()

    def __init__(self, container, parent=None):
        self._columns: list[str] = []
        self._dataset_name = "No dataset"
        super().__init__(container, parent)

    def _build_ui(self) -> None:
        title = QLabel("Train Model")
        title.setObjectName("PageTitle")
        self._layout.addWidget(title)

        self._stepper = Stepper(self.STEPS)
        self._stepper.step_clicked.connect(self._on_step_clicked)
        self._layout.addWidget(self._stepper)

        nav = QHBoxLayout()
        self._back_btn = QPushButton("Back")
        self._back_btn.setObjectName("GhostButton")
        self._next_btn = QPushButton("Next")
        self._next_btn.setObjectName("PrimaryButton")
        constrain_primary_button(self._next_btn)
        self._back_btn.clicked.connect(self._prev_step)
        self._next_btn.clicked.connect(self._next_step)
        nav.addWidget(self._back_btn)
        nav.addWidget(self._next_btn)
        nav.addStretch()
        self._layout.addLayout(nav)

        # Hidden compatibility hook for older connections/tests
        self._train_btn = QPushButton("Start Training")
        self._train_btn.setObjectName("PrimaryButton")
        self._train_btn.hide()
        self._train_btn.clicked.connect(self.train_requested.emit)

        self._stack = QStackedWidget()
        self._task_combo = QComboBox()
        task_choices = [
            (TaskType.CLASSIFICATION, "CLASSIFICATION"),
            (TaskType.REGRESSION, "REGRESSION"),
            (TaskType.CLUSTERING, "CLUSTERING"),
            (TaskType.ANOMALY_DETECTION, "ANOMALY_DETECTION"),
            (TaskType.TIME_SERIES, "TIME_SERIES (regression models + time CV)"),
        ]
        for task, label in task_choices:
            self._task_combo.addItem(label, task.value)
        self._dataset_label = QLabel("Import a dataset on the Data page.")
        self._target_combo = QComboBox()
        self._feature_list = QListWidget()
        self._feature_list.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._prepare_label = QLabel(
            "Optional: build a preprocessing pipeline on the Prepare page. "
            "Enabled steps will run before training (fit on train split only)."
        )
        self._prepare_label.setWordWrap(True)
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
        self._tune_trials = QSpinBox()
        self._tune_trials.setRange(5, 100)
        self._tune_trials.setValue(20)
        self._tune_hint = QLabel("")
        self._tune_hint.setObjectName("TextMuted")
        self._tune_hint.setWordWrap(True)

        self._sum_task = QLabel("—")
        self._sum_dataset = QLabel("—")
        self._sum_target = QLabel("—")
        self._sum_features = QLabel("—")
        self._sum_model = QLabel("—")
        self._sum_tune = QLabel("—")
        self._sum_split = QLabel("—")
        self._sum_prep = QLabel("—")
        self._summary_form = QFormLayout()
        self._summary_form.setContentsMargins(0, 0, 0, 0)
        self._summary_form.setHorizontalSpacing(20)
        self._summary_form.setVerticalSpacing(10)
        self._summary_form.setLabelAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        self._summary_form.setFormAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
        )
        self._summary_form.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )
        self._summary_form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.DontWrapRows)
        for label, widget in (
            ("Task", self._sum_task),
            ("Dataset", self._sum_dataset),
            ("Target", self._sum_target),
            ("Features", self._sum_features),
            ("Prepare", self._sum_prep),
            ("Model", self._sum_model),
            ("Tuning", self._sum_tune),
            ("Split / CV", self._sum_split),
        ):
            widget.setWordWrap(False)
            widget.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            widget.setMinimumHeight(0)
            widget.setMaximumHeight(16777215)
            widget.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
            )
            key = QLabel(label)
            key.setObjectName("TextMuted")
            self._summary_form.addRow(key, widget)

        self._automl_table = QTableWidget()
        self._automl_table.setColumnCount(5)
        self._automl_table.setHorizontalHeaderLabels(
            ["Rank", "Model", "CV score", "Validation", "Time"]
        )
        configure_table_header(
            self._automl_table.horizontalHeader(),
            contents_cols=(0, 2, 3, 4),
            stretch_cols=(1,),
        )
        self._automl_table.setAlternatingRowColors(True)
        self._automl_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._automl_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._automl_table.verticalHeader().setVisible(False)
        self._automl_table.setMinimumHeight(160)
        self._automl_table.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._automl_table.setRowCount(0)

        panels = [
            self._form_panel("Task type", [("Task", self._task_combo)]),
            self._data_panel(),
            self._model_panel(),
            self._tune_panel(),
            self._train_panel(),
        ]
        for panel in panels:
            self._stack.addWidget(panel)
        self._layout.addWidget(self._stack, 1)
        self._task_combo.currentTextChanged.connect(self._update_models)
        self._model_combo.currentTextChanged.connect(self._on_model_or_tune_changed)
        self._tune_combo.currentTextChanged.connect(self._on_model_or_tune_changed)
        self._target_combo.currentTextChanged.connect(self._refresh_summary)
        self._feature_list.itemChanged.connect(self._refresh_summary)
        self._update_models()
        self._on_model_or_tune_changed()

    def _data_panel(self) -> QWidget:
        box = QWidget()
        layout = QVBoxLayout(box)
        layout.addWidget(self._dataset_label)
        card = Card("Target & features")
        form = QFormLayout()
        form.addRow("Target:", self._target_combo)
        card.add_layout(form)
        card.add_widget(self._feature_list, stretch=1)
        layout.addWidget(card, 1)
        layout.addWidget(self._prepare_label)
        return box

    def _model_panel(self) -> QWidget:
        box = QWidget()
        layout = QVBoxLayout(box)
        card = Card("Model & split")
        form = QFormLayout()
        form.addRow("Algorithm:", self._model_combo)
        form.addRow("Test size:", self._test_spin)
        form.addRow("CV folds:", self._cv_spin)
        card.add_layout(form)
        layout.addWidget(card)
        layout.addStretch(1)
        return box

    def _tune_panel(self) -> QWidget:
        card = Card("Hyperparameter tuning")
        form = QFormLayout()
        form.addRow("Method:", self._tune_combo)
        form.addRow("Optuna trials:", self._tune_trials)
        card.add_layout(form)
        card.add_widget(self._tune_hint)
        box = QWidget()
        layout = QVBoxLayout(box)
        layout.addWidget(card)
        layout.addStretch(1)
        return box

    def _form_panel(self, title: str, rows: list[tuple[str, QWidget]]) -> QWidget:
        card = Card(title)
        form = QFormLayout()
        for label, widget in rows:
            form.addRow(f"{label}:", widget)
        card.add_layout(form)
        box = QWidget()
        layout = QVBoxLayout(box)
        layout.addWidget(card)
        layout.addStretch(1)
        return box

    def _wrap(self, widget: QWidget) -> QWidget:
        box = QWidget()
        layout = QVBoxLayout(box)
        layout.addWidget(widget)
        layout.addStretch(1)
        return box

    def _train_panel(self) -> QWidget:
        summary = Card("Training summary")
        summary.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        summary.add_layout(self._summary_form)
        hint = QLabel("Click Start Training in the wizard bar above to begin.")
        hint.setObjectName("TextMuted")
        hint.setWordWrap(True)
        summary.add_widget(hint)

        automl = Card("AutoML")
        automl_btn = QPushButton("Run AutoML leaderboard")
        automl_btn.setObjectName("GhostButton")
        automl_btn.setToolTip(
            "Runs in the background (UI stays responsive). "
            "Compares models with CV, then selects the best for Start Training."
        )
        automl_btn.clicked.connect(self.automl_requested.emit)
        self._automl_btn = automl_btn
        automl.add_widget(automl_btn)
        self._automl_status = QLabel(
            "Run AutoML to compare models. Best model is selected automatically."
        )
        self._automl_status.setObjectName("TextMuted")
        self._automl_status.setWordWrap(True)
        automl.add_widget(self._automl_status)
        automl.add_widget(self._automl_table, stretch=1)

        inner = QWidget()
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        layout.addWidget(summary)
        layout.addWidget(automl, 1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        scroll.setWidget(inner)
        return scroll

    def set_automl_busy(self, busy: bool, message: str = "") -> None:
        self._automl_btn.setEnabled(not busy)
        self._automl_btn.setText("Running AutoML…" if busy else "Run AutoML leaderboard")
        if busy:
            self._automl_status.setText(message or "Comparing models…")
            self._automl_table.setRowCount(0)
        elif message:
            self._automl_status.setText(message)

    def show_automl_leaderboard(self, entries: list) -> None:
        self._automl_table.setRowCount(0)
        if not entries:
            self._automl_status.setText("No models completed successfully.")
            return
        best = entries[0]
        self._automl_status.setText(
            f"Best: {best.model_name}  ·  CV {best.cv_score:.4f}  ·  ready to Start Training"
        )
        self._automl_table.setRowCount(len(entries))
        for i, e in enumerate(entries):
            val = f"{e.validation_score:.4f}" if e.validation_score is not None else "—"
            cells = [
                f"#{e.rank}",
                e.model_name,
                f"{e.cv_score:.4f}",
                val,
                f"{e.training_time:.1f}s",
            ]
            for col, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if col != 1:
                    item.setTextAlignment(
                        int(Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter)
                    )
                self._automl_table.setItem(i, col, item)
        self._automl_table.selectRow(0)

    def on_show(self) -> None:
        self._refresh_prepare_label()
        self._refresh_summary()

    def _refresh_prepare_label(self) -> None:
        win = self.window()
        pages = getattr(win, "_pages", None) or {}
        prepare = pages.get("prepare")
        if prepare is None:
            return
        n = len(getattr(prepare, "pipeline", None).steps) if getattr(prepare, "pipeline", None) else 0
        active = 0
        if n:
            active = sum(
                1 for s in prepare.pipeline.steps if getattr(s, "enabled", True)
            )
        if n == 0:
            self._prepare_label.setText(
                "No preprocessing steps yet. Optional: open Prepare to add imputation, "
                "encoding, or scaling. Training will use raw prepared features."
            )
        else:
            self._prepare_label.setText(
                f"Prepare pipeline: {active} enabled / {n} total steps will run before the model."
            )

    def set_dataset(
        self, name: str, columns: list[str], suggested_target: str, suggested_task: TaskType
    ) -> None:
        self._columns = columns
        self._dataset_name = name
        self._dataset_label.setText(f"{name} — {len(columns):,} columns loaded")
        self._target_combo.clear()
        self._target_combo.addItems(columns)
        self._feature_list.clear()
        for col in columns:
            item = QListWidgetItem(col)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(
                Qt.CheckState.Checked if col != suggested_target else Qt.CheckState.Unchecked
            )
            self._feature_list.addItem(item)
        if suggested_target in columns:
            self._target_combo.setCurrentText(suggested_target)
        idx = self._task_combo.findData(suggested_task.value)
        if idx >= 0:
            self._task_combo.setCurrentIndex(idx)
        self._update_models()
        self._refresh_summary()

    def build_config(self) -> TrainingConfig | None:
        task = self.get_task()
        unsupervised = task in (TaskType.CLUSTERING, TaskType.ANOMALY_DETECTION)
        target = self._target_combo.currentText()
        features = [
            self._feature_list.item(i).text()
            for i in range(self._feature_list.count())
            if self._feature_list.item(i).checkState() == Qt.CheckState.Checked
            and (unsupervised or self._feature_list.item(i).text() != target)
        ]
        if not features:
            return None
        if not unsupervised and not target:
            return None
        if not self._model_combo.currentText():
            return None

        model_name = self._model_combo.currentText()
        model_id = next(
            (k for k, v in MODEL_REGISTRY.items() if v.name == model_name),
            "linear_regression",
        )
        tune_label = self._tune_combo.currentText()
        tune_method = {
            "None": "none",
            "Grid search": "grid",
            "Optuna": "optuna",
        }.get(tune_label, "none")
        if unsupervised:
            tune_method = "none"

        return TrainingConfig(
            task=task,
            model_id=model_id,
            target_column=target or (features[0] if features else ""),
            feature_columns=features,
            test_size=float(self._test_spin.value()),
            cv_splits=int(self._cv_spin.value()),
            tune_method=tune_method,
            tune_trials=int(self._tune_trials.value()),
            is_time_series=task == TaskType.TIME_SERIES,
        )

    def validate_step(self, index: int) -> str | None:
        if index <= 0:
            return None
        if index >= 1 and not self._columns:
            return "Import a dataset on the Data page first."
        if index >= 1:
            target = self._target_combo.currentText()
            features = [
                self._feature_list.item(i).text()
                for i in range(self._feature_list.count())
                if self._feature_list.item(i).checkState() == Qt.CheckState.Checked
                and self._feature_list.item(i).text() != target
            ]
            unsupervised = self._current_task_value() in (
                TaskType.CLUSTERING.value,
                TaskType.ANOMALY_DETECTION.value,
            )
            if index >= 1 and self._columns:
                if not unsupervised and (not target or not features):
                    # Allow being on Data step while still selecting
                    if index >= 2:
                        return "Select a target column and at least one feature."
                if unsupervised and not features and index >= 2:
                    return "Select at least one feature column."
        if index >= 2 and not self._model_combo.currentText():
            return "Select a model algorithm."
        if index >= 3 and self._tune_combo.currentText() == "Optuna":
            try:
                import optuna  # noqa: F401
            except ImportError:
                return "Optuna is not installed. Choose None/Grid, or: pip install optuna"
        if index >= 4 and self.build_config() is None:
            return "Finish Task, Data, and Model before training."
        return None

    def _update_models(self) -> None:
        self._model_combo.clear()
        task = self.get_task()
        models = get_models_for_task(task)
        for meta in models:
            self._model_combo.addItem(meta.name)
        if not models and task == TaskType.TIME_SERIES:
            for meta in get_models_for_task(TaskType.REGRESSION):
                self._model_combo.addItem(meta.name)
        self._on_model_or_tune_changed()

    def _on_model_or_tune_changed(self, *_args) -> None:
        self._tune_trials.setEnabled(self._tune_combo.currentText() == "Optuna")
        model_name = self._model_combo.currentText()
        model_id = next((k for k, v in MODEL_REGISTRY.items() if v.name == model_name), None)
        space = MODEL_REGISTRY[model_id].hyperparameters if model_id else {}
        method = self._tune_combo.currentText()
        if method == "None":
            self._tune_hint.setText("Uses model defaults (fastest).")
        elif not space:
            self._tune_hint.setText(
                "This model has no registered search space — tuning will be skipped."
            )
        elif method == "Optuna":
            self._tune_hint.setText(
                f"Optuna will search: {', '.join(space.keys())} "
                f"({self._tune_trials.value()} trials)."
            )
        else:
            self._tune_hint.setText(f"Grid search over: {', '.join(space.keys())}.")
        self._refresh_summary()

    def _summary_prepare_text(self) -> str:
        """Short prepare line for the summary grid (not the long Data-step hint)."""
        win = self.window()
        pages = getattr(win, "_pages", None) or {}
        prepare = pages.get("prepare")
        if prepare is None:
            return "—"
        pipe = getattr(prepare, "pipeline", None)
        steps = getattr(pipe, "steps", None) or []
        if not steps:
            return "None"
        active = sum(1 for s in steps if getattr(s, "enabled", True))
        return f"{active} of {len(steps)} steps"

    def _refresh_summary(self) -> None:
        config = self.build_config()
        self._refresh_prepare_label()
        prep_txt = self._summary_prepare_text()
        if not config:
            self._sum_task.setText("—")
            self._sum_dataset.setText(self._dataset_name)
            self._sum_target.setText("Select target and features")
            self._sum_features.setText("—")
            self._sum_model.setText("—")
            self._sum_tune.setText("—")
            self._sum_split.setText("—")
            self._sum_prep.setText(prep_txt)
            return
        tune_txt = {
            "none": "None",
            "grid": "Grid search",
            "optuna": f"Optuna ({config.tune_trials} trials)",
        }.get(config.tune_method, config.tune_method)
        self._sum_task.setText(config.task.value)
        self._sum_dataset.setText(self._dataset_name)
        self._sum_target.setText(config.target_column)
        self._sum_features.setText(str(len(config.feature_columns)))
        self._sum_model.setText(self._model_combo.currentText())
        self._sum_tune.setText(tune_txt)
        self._sum_prep.setText(prep_txt)
        self._sum_split.setText(
            f"{config.test_size:.0%} test · {config.cv_splits}-fold CV"
        )

    def _on_step_clicked(self, index: int) -> None:
        current = self._stepper.current_index()
        if index > current:
            err = self.validate_step(index)
            if err:
                QMessageBox.warning(self, "Complete this step", err)
                return
        self._goto_step(index)

    def _goto_step(self, index: int) -> None:
        if index in (1, 4):
            self._refresh_prepare_label()
        self._stack.setCurrentIndex(index)
        self._stepper.set_current(index)
        if index >= len(self.STEPS) - 1:
            self._next_btn.setText("Start Training")
        else:
            self._next_btn.setText("Next")

    def _next_step(self) -> None:
        idx = self._stepper.current_index()
        if idx >= len(self.STEPS) - 1:
            self.train_requested.emit()
            return
        nxt = idx + 1
        err = self.validate_step(nxt)
        if err:
            QMessageBox.warning(self, "Complete this step", err)
            return
        self._goto_step(nxt)

    def _prev_step(self) -> None:
        idx = max(self._stepper.current_index() - 1, 0)
        self._goto_step(idx)

    def get_target_column(self) -> str:
        return self._target_combo.currentText()

    def get_task(self) -> TaskType:
        data = self._task_combo.currentData()
        if data:
            return TaskType(data)
        text = self._task_combo.currentText().split(" ")[0]
        return TaskType(text)

    def _current_task_value(self) -> str:
        data = self._task_combo.currentData()
        if data:
            return data
        return self._task_combo.currentText().split(" ")[0]

    def get_model_name(self) -> str:
        return self._model_combo.currentText()
