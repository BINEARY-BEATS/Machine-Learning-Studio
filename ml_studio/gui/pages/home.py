"""Home dashboard with KPIs, quick actions, and recent activity."""

from __future__ import annotations

from PyQt6.QtWidgets import QGridLayout, QHBoxLayout, QLabel, QPushButton

from ml_studio.gui.layout_utils import constrain_primary_button
from ml_studio.gui.pages.base_page import BasePage
from ml_studio.gui.widgets.card import Card
from ml_studio.gui.widgets.empty_state import EmptyState
from ml_studio.gui.widgets.stat_card import StatCard


class HomePage(BasePage):
    def _build_ui(self) -> None:
        title = QLabel("Machine Learning Studio")
        title.setObjectName("PageTitle")
        self._layout.addWidget(title)

        self._stats = QGridLayout()
        self._stats.setColumnStretch(0, 1)
        self._stats.setColumnStretch(1, 1)
        self._project_card = StatCard("Project", "None")
        self._datasets_card = StatCard("Datasets", "0")
        self._models_card = StatCard("Models", "0")
        self._experiments_card = StatCard("Experiments", "0")
        self._stats.addWidget(self._project_card, 0, 0)
        self._stats.addWidget(self._datasets_card, 0, 1)
        self._stats.addWidget(self._models_card, 1, 0)
        self._stats.addWidget(self._experiments_card, 1, 1)
        self._layout.addLayout(self._stats)

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

        recent = Card("Recent activity")
        self._recent_label = QLabel("No recent activity yet.")
        self._recent_label.setWordWrap(True)
        recent.add_widget(self._recent_label)
        self._layout.addWidget(recent, 1)

        self._empty = EmptyState(
            "Welcome to ML Studio",
            "Create a project, import data, and start training models.",
            icon_name="home",
        )
        self._layout.addWidget(self._empty, 1)

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
            self._project_card.set_value(pm.current.name)
            self._empty.hide()
        else:
            self._project_card.set_value("None")
            self._empty.show()
        datasets = 1 if controller.current_dataset else 0
        models = len(controller.registry.list_models())
        experiments = 0
        if pages and "evaluate" in pages:
            experiments = len(pages["evaluate"]._runs)
        self._datasets_card.set_value(str(datasets))
        self._models_card.set_value(str(models))
        self._experiments_card.set_value(str(experiments))
        if controller.current_dataset:
            self._recent_label.setText(
                f"Latest dataset: {controller.current_dataset.name} "
                f"({controller.current_dataset.row_count:,} rows)"
            )

    def on_show(self) -> None:
        win = self.window()
        controller = getattr(win, "controller", None)
        pages = getattr(win, "_pages", None)
        if controller is not None:
            self.refresh_stats(controller, pages)
