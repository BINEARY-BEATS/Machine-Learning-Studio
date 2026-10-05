"""Home — pipeline readiness console for the ML lab workflow."""

from __future__ import annotations

from PyQt6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from ml_studio.gui.layout_utils import constrain_primary_button
from ml_studio.gui.pages.base_page import BasePage
from ml_studio.gui.widgets.card import Card
from ml_studio.gui.widgets.empty_state import EmptyState


class HomePage(BasePage):
    def _build_ui(self) -> None:
        title = QLabel("ML Studio")
        title.setObjectName("PageTitle")
        self._layout.addWidget(title)

        subtitle = QLabel("Local tabular lab — import → prepare → train → evaluate → predict")
        subtitle.setObjectName("Breadcrumb")
        self._layout.addWidget(subtitle)

        self._ready_card = Card("Pipeline readiness")
        ready_body = QWidget()
        self._ready_layout = QVBoxLayout(ready_body)
        self._ready_layout.setContentsMargins(0, 0, 0, 0)
        self._ready_layout.setSpacing(4)

        self._row_project = self._make_row("Project")
        self._row_data = self._make_row("Dataset")
        self._row_target = self._make_row("Target")
        self._row_pipeline = self._make_row("Prepare pipeline")
        self._row_model = self._make_row("Last model")
        self._row_next = self._make_row("Next action")
        for row in (
            self._row_project,
            self._row_data,
            self._row_target,
            self._row_pipeline,
            self._row_model,
            self._row_next,
        ):
            self._ready_layout.addWidget(row[0])

        self._ready_card.add_widget(ready_body)
        self._layout.addWidget(self._ready_card)

        actions = QHBoxLayout()
        self._new_btn = QPushButton("New Project")
        self._new_btn.setObjectName("PrimaryButton")
        constrain_primary_button(self._new_btn)
        self._open_btn = QPushButton("Open Project")
        self._open_btn.setObjectName("GhostButton")
        self._import_btn = QPushButton("Import Dataset")
        self._import_btn.setObjectName("GhostButton")
        actions.addWidget(self._new_btn)
        actions.addWidget(self._open_btn)
        actions.addWidget(self._import_btn)
        actions.addStretch()
        self._layout.addLayout(actions)

        self._empty = EmptyState(
            "Welcome to ML Studio",
            "Create a project, import tabular data, and train models offline.",
            icon_name="home",
        )
        self._layout.addWidget(self._empty, 1)
        self._layout.addStretch(1)

    def _make_row(self, label: str) -> tuple[QWidget, QLabel, QLabel]:
        row = QWidget()
        row.setObjectName("ReadinessRow")
        lay = QHBoxLayout(row)
        lay.setContentsMargins(0, 6, 0, 6)
        key = QLabel(label)
        key.setFixedWidth(140)
        val = QLabel("—")
        val.setObjectName("MonoMetric")
        val.setWordWrap(True)
        lay.addWidget(key)
        lay.addWidget(val, 1)
        return row, key, val

    def wire_empty_actions(self, new_cb, open_cb, import_cb) -> None:
        from PyQt6.QtWidgets import QPushButton

        btn = QPushButton("Import dataset")
        btn.setObjectName("PrimaryButton")
        constrain_primary_button(btn)
        btn.clicked.connect(import_cb)
        self._empty.set_action(btn)

    def refresh_stats(self, controller, pages=None) -> None:
        pm = self.container.project_manager
        if pm and pm.current:
            self._row_project[2].setText(pm.current.name)
            self._empty.hide()
            self._ready_card.show()
        else:
            self._row_project[2].setText("None — create or open a project")
            self._empty.show()

        ds = controller.current_dataset
        if ds:
            self._row_data[2].setText(f"{ds.name}  ·  {ds.row_count:,} × {ds.column_count}")
        else:
            self._row_data[2].setText("No dataset — import CSV / Parquet")

        target = "—"
        if ds and getattr(ds, "target_column", None):
            target = str(ds.target_column)
        elif pages and "prepare" in pages:
            prepare = pages["prepare"]
            schema = getattr(prepare, "schema_overrides", None) or {}
            for col, role in schema.items():
                if role == "target":
                    target = col
                    break
            if target == "—" and hasattr(prepare, "get_target_column"):
                try:
                    t = prepare.get_target_column()
                    if t:
                        target = t
                except Exception:
                    pass
        if pages and "train" in pages:
            train = pages["train"]
            cfg_target = getattr(train, "_target_combo", None)
            if cfg_target is not None and cfg_target.currentText():
                target = cfg_target.currentText()
        self._row_target[2].setText(target)

        n_steps = 0
        if pages and "prepare" in pages:
            pipe = getattr(pages["prepare"], "pipeline", None)
            if pipe is not None:
                n_steps = len([s for s in pipe.steps if getattr(s, "enabled", True)])
        self._row_pipeline[2].setText(
            f"{n_steps} active step(s)" if n_steps else "Empty (optional)"
        )

        models = controller.registry.list_models()
        if models:
            m = models[0]
            primary = (
                m.metrics.get("r2")
                or m.metrics.get("f1")
                or m.metrics.get("accuracy")
                or "—"
            )
            score = f"{primary:.3f}" if isinstance(primary, float) else str(primary)
            self._row_model[2].setText(f"{m.name}  ·  score {score}")
        else:
            self._row_model[2].setText("None trained yet")

        # Next action
        if not (pm and pm.current):
            nxt = "Create or open a project"
        elif not ds:
            nxt = "Import a dataset (Data)"
        elif target in ("—", "", None):
            nxt = "Set a target column (Prepare)"
        elif not models:
            nxt = "Train a model (Train)"
        else:
            nxt = "Evaluate runs or Predict on new data"
        self._row_next[2].setText(nxt)

    def on_show(self) -> None:
        win = self.window()
        controller = getattr(win, "controller", None)
        pages = getattr(win, "_pages", None)
        if controller is not None:
            self.refresh_stats(controller, pages)
