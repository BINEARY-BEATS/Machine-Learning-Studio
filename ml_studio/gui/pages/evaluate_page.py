"""Evaluation page with semantic metrics and experiment history."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ml_studio.app.metric_color import metric_color
from ml_studio.app.theme import ThemeMode
from ml_studio.gui.layout_utils import constrain_primary_button, configure_table_header
from ml_studio.gui.pages.base_page import BasePage
from ml_studio.gui.pages.evaluate_clusters import build_clusters_table, fill_clusters_table
from ml_studio.gui.widgets.card import Card
from ml_studio.gui.widgets.empty_state import EmptyState
from ml_studio.gui.widgets.stat_card import StatCard

_SKIP_METRIC_KEYS = frozenset({"confusion_matrix", "cluster_profile"})


@dataclass
class ExperimentRun:
    run_id: str
    model_name: str
    model_id: str
    task: str
    dataset_name: str
    target: str
    metrics: dict
    cv_mean: float | None
    cv_std: float | None
    duration_sec: float
    train_rows: int
    test_rows: int
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    registry_id: str = ""
    y_true: object | None = None
    y_pred: object | None = None


class EvaluatePage(BasePage):
    def __init__(self, container, parent=None):
        self._runs: list[ExperimentRun] = []
        self._mode = ThemeMode.LIGHT
        super().__init__(container, parent)

    def _build_ui(self) -> None:
        header = QHBoxLayout()
        title = QLabel("Model Evaluation")
        title.setObjectName("PageTitle")
        header.addWidget(title)
        header.addStretch()
        self._count_label = QLabel("0 experiments")
        header.addWidget(self._count_label)
        self._layout.addLayout(header)

        self._metric_cards = QGridLayout()
        self._metric_cards.setColumnStretch(0, 1)
        self._metric_cards.setColumnStretch(1, 1)
        self._primary_card = StatCard("Primary metric", metric_key="r2")
        self._cv_card = StatCard("CV mean")
        self._duration_card = StatCard("Duration")
        self._rows_card = StatCard("Train rows")
        self._metric_cards.addWidget(self._primary_card, 0, 0)
        self._metric_cards.addWidget(self._cv_card, 0, 1)
        self._metric_cards.addWidget(self._duration_card, 0, 2)
        self._metric_cards.addWidget(self._rows_card, 0, 3)
        for col in range(4):
            self._metric_cards.setColumnStretch(col, 1)
        self._layout.addLayout(self._metric_cards)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(QLabel("Experiment history"))
        self._leaderboard = QTableWidget()
        self._leaderboard.setColumnCount(7)
        self._leaderboard.setHorizontalHeaderLabels(
            ["Model", "Task", "Score", "CV", "Duration", "Dataset", "Time"]
        )
        configure_table_header(
            self._leaderboard.horizontalHeader(),
            contents_cols=(1, 2, 3, 4, 6),
            stretch_cols=(0, 5),
        )
        self._leaderboard.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._leaderboard.itemSelectionChanged.connect(self._on_run_selected)
        left_layout.addWidget(self._leaderboard, 1)
        splitter.addWidget(left)

        right = Card("Metric details")
        self._detail_tabs = QTabWidget()
        metrics_tab = QWidget()
        metrics_layout = QVBoxLayout(metrics_tab)
        metrics_layout.setContentsMargins(0, 0, 0, 0)
        self._metrics_table = QTableWidget()
        self._metrics_table.setColumnCount(2)
        self._metrics_table.setHorizontalHeaderLabels(["Metric", "Value"])
        configure_table_header(
            self._metrics_table.horizontalHeader(),
            contents_cols=(0,),
            stretch_cols=(1,),
        )
        self._metrics_table.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        metrics_layout.addWidget(self._metrics_table, 1)
        self._plot_host = QWidget()
        self._plot_layout = QVBoxLayout(self._plot_host)
        self._plot_layout.setContentsMargins(0, 0, 0, 0)
        metrics_layout.addWidget(self._plot_host, 1)
        self._detail_tabs.addTab(metrics_tab, "Metrics")
        self._clusters_table = build_clusters_table()
        self._detail_tabs.addTab(self._clusters_table, "Clusters")
        right.add_widget(self._detail_tabs, stretch=1)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([640, 360])
        self._content_splitter = splitter
        self._layout.addWidget(splitter, 1)

        self._empty = EmptyState(
            "No experiments yet",
            "Train a model to see metrics, semantic warnings, and run history here.",
            icon_name="chart",
        )
        go_train = QPushButton("Go to Train")
        go_train.setObjectName("PrimaryButton")
        constrain_primary_button(go_train)
        go_train.clicked.connect(self._goto_train)
        self._empty.set_action(go_train)
        self._layout.addWidget(self._empty, 1)
        self._refresh_leaderboard()

    def _goto_train(self) -> None:
        win = self.window()
        if hasattr(win, "_navigate"):
            win._navigate("train")

    def set_theme_mode(self, mode: ThemeMode) -> None:
        self._mode = mode
        self._primary_card.set_theme_mode(mode)
        self._empty.set_theme_mode(mode)

    def add_run(self, result, model_display_name: str, dataset_name: str, registry_id: str = "") -> None:
        cv_mean = result.cv_scores.get("mean") if result.cv_scores else None
        cv_std = result.cv_scores.get("std") if result.cv_scores else None
        run = ExperimentRun(
            run_id=result.experiment_id,
            model_name=model_display_name,
            model_id=result.model_id,
            task=result.task.value,
            dataset_name=dataset_name,
            target=result.target_column,
            metrics=dict(result.metrics),
            cv_mean=cv_mean,
            cv_std=cv_std,
            duration_sec=result.training_duration,
            train_rows=result.train_size,
            test_rows=result.test_size,
            registry_id=registry_id,
            y_true=getattr(result, "y_true_holdout", None),
            y_pred=getattr(result, "y_pred_holdout", None),
        )
        self._runs.insert(0, run)
        self._persist_runs()
        self._refresh_leaderboard()
        self._leaderboard.selectRow(0)
        self._show_run_detail(run)

    def _persist_runs(self) -> None:
        """Best-effort durable experiment store via project manager."""
        try:
            from ml_studio.core.persistence.experiments import ExperimentStore

            pm = self.container.project_manager
            if not pm or not pm.current or not pm.current.path:
                return
            store = ExperimentStore(pm.current.path)
            store.save_runs(self._runs)
        except Exception:
            pass

    def load_persisted_runs(self) -> None:
        try:
            from ml_studio.core.persistence.experiments import ExperimentStore

            pm = self.container.project_manager
            if not pm or not pm.current or not pm.current.path:
                return
            store = ExperimentStore(pm.current.path)
            loaded = store.load_runs()
            if loaded:
                self._runs = loaded
                self._refresh_leaderboard()
        except Exception:
            pass

    def run_count(self) -> int:
        return len(self._runs)

    def _primary_score(self, task: str, metrics: dict):
        if task == "REGRESSION":
            return metrics.get("r2")
        if task == "CLASSIFICATION":
            return metrics.get("f1") or metrics.get("accuracy")
        if task == "CLUSTERING":
            return metrics.get("silhouette")
        return None

    def _score_label(self, task: str) -> str:
        return {"REGRESSION": "R²", "CLASSIFICATION": "F1", "CLUSTERING": "Silhouette"}.get(task, "Score")

    def _refresh_leaderboard(self) -> None:
        empty = len(self._runs) == 0
        self._empty.setVisible(empty)
        self._content_splitter.setVisible(not empty)
        # Hide metric cards row when empty by hiding widgets
        for card in (
            self._primary_card,
            self._cv_card,
            self._duration_card,
            self._rows_card,
        ):
            card.setVisible(not empty)
        self._count_label.setText(f"{len(self._runs)} experiment{'s' if len(self._runs) != 1 else ''}")
        self._leaderboard.setRowCount(len(self._runs))
        for i, run in enumerate(self._runs):
            score = self._primary_score(run.task, run.metrics)
            score_text = f"{score:.4f}" if isinstance(score, float) else "—"
            cv_text = f"{run.cv_mean:.4f}" if run.cv_mean is not None else "—"
            self._leaderboard.setItem(i, 0, QTableWidgetItem(run.model_name))
            self._leaderboard.setItem(i, 1, QTableWidgetItem(run.task))
            self._leaderboard.setItem(i, 2, QTableWidgetItem(score_text))
            self._leaderboard.setItem(i, 3, QTableWidgetItem(cv_text))
            self._leaderboard.setItem(i, 4, QTableWidgetItem(f"{run.duration_sec:.1f}s"))
            self._leaderboard.setItem(i, 5, QTableWidgetItem(run.dataset_name))
            self._leaderboard.setItem(i, 6, QTableWidgetItem(run.timestamp.strftime("%H:%M:%S")))

    def _on_run_selected(self) -> None:
        rows = self._leaderboard.selectionModel().selectedRows()
        if rows and 0 <= rows[0].row() < len(self._runs):
            self._show_run_detail(self._runs[rows[0].row()])

    def _show_run_detail(self, run: ExperimentRun) -> None:
        score_name = self._score_label(run.task)
        primary = self._primary_score(run.task, run.metrics)
        metric_key = {
            "REGRESSION": "r2",
            "TIME_SERIES": "r2",
            "CLASSIFICATION": "f1",
            "CLUSTERING": "silhouette",
        }.get(run.task, "r2")
        self._primary_card.set_metric_key(metric_key)
        self._primary_card.set_value(
            f"{primary:.4f}" if isinstance(primary, float) else "—",
            raw=primary,
            task=run.task,
        )
        self._primary_card.set_title(score_name)
        self._cv_card.set_value(f"{run.cv_mean:.4f}" if run.cv_mean is not None else "—")
        self._duration_card.set_value(f"{run.duration_sec:.1f}s")
        self._rows_card.set_value(f"{run.train_rows:,}")

        self._fill_metrics_table(run.metrics, run.task)
        fill_clusters_table(self._clusters_table, run.metrics.get("cluster_profile"))
        has_clusters = bool(run.metrics.get("cluster_profile"))
        self._detail_tabs.setTabVisible(1, has_clusters)
        if has_clusters:
            self._detail_tabs.setCurrentIndex(1 if run.task == "CLUSTERING" else 0)
        self._update_plots(run)

    def _fill_metrics_table(self, metrics: dict, task: str) -> None:
        rows: list[tuple[str, str, object]] = []
        for key, val in metrics.items():
            if key in _SKIP_METRIC_KEYS:
                continue
            if isinstance(val, float):
                rows.append((key, f"{val:.4f}", val))
            else:
                rows.append((key, str(val), val))
        self._metrics_table.setRowCount(len(rows))
        for i, (key, text, raw) in enumerate(rows):
            self._metrics_table.setItem(i, 0, QTableWidgetItem(key))
            item = QTableWidgetItem(text)
            if isinstance(raw, float):
                color = metric_color(self._mode, key, raw, task)
                item.setForeground(__import__("PyQt6.QtGui", fromlist=["QColor"]).QColor(color))
            self._metrics_table.setItem(i, 1, item)

    def _update_plots(self, run: ExperimentRun) -> None:
        while self._plot_layout.count():
            item = self._plot_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        try:
            from ml_studio.gui.charts import confusion_matrix_widget, residual_parity_widget

            cm = run.metrics.get("confusion_matrix")
            if cm is not None:
                self._plot_layout.addWidget(confusion_matrix_widget(cm))
            elif (
                run.task in ("REGRESSION", "TIME_SERIES")
                and run.y_true is not None
                and run.y_pred is not None
            ):
                self._plot_layout.addWidget(residual_parity_widget(run.y_true, run.y_pred))
        except Exception as exc:
            self._plot_layout.addWidget(QLabel(f"Plot unavailable: {exc}"))

    def on_show(self) -> None:
        self._refresh_leaderboard()
        if self._runs and not self._leaderboard.selectedItems():
            self._leaderboard.selectRow(0)

    def show_metrics(self, metrics: dict) -> None:
        """Show a one-off metrics dict (tests / quick introspection)."""
        if not metrics:
            return
        self._empty.hide()
        self._content_splitter.show()
        for card in (
            self._primary_card,
            self._cv_card,
            self._duration_card,
            self._rows_card,
        ):
            card.show()
        self._fill_metrics_table(metrics, "REGRESSION")
        fill_clusters_table(self._clusters_table, metrics.get("cluster_profile"))
        self._detail_tabs.setTabVisible(1, bool(metrics.get("cluster_profile")))
        primary = metrics.get("r2") or metrics.get("f1") or metrics.get("accuracy")
        self._primary_card.set_value(
            f"{primary:.4f}" if isinstance(primary, float) else "—",
            raw=primary if isinstance(primary, float) else None,
        )
        self._primary_card.set_title("Primary metric")
