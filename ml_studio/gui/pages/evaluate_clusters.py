"""Clusters tab helpers for EvaluatePage."""

from __future__ import annotations

from PyQt6.QtWidgets import QTableWidget, QTableWidgetItem, QWidget

from ml_studio.gui.layout_utils import configure_table_header


def build_clusters_table(parent: QWidget | None = None) -> QTableWidget:
    table = QTableWidget(parent)
    table.setColumnCount(3)
    table.setHorizontalHeaderLabels(["Cluster", "Size", "Feature means"])
    configure_table_header(
        table.horizontalHeader(),
        contents_cols=(0, 1),
        stretch_cols=(2,),
    )
    return table


def fill_clusters_table(table: QTableWidget, profile: list | None) -> None:
    if not profile:
        table.setRowCount(0)
        return
    table.setRowCount(len(profile))
    for i, entry in enumerate(profile):
        means = entry.get("means") or {}
        means_text = ", ".join(f"{k}={v:.3f}" for k, v in means.items()) if means else "—"
        table.setItem(i, 0, QTableWidgetItem(str(entry.get("cluster", ""))))
        table.setItem(i, 1, QTableWidgetItem(str(entry.get("size", ""))))
        table.setItem(i, 2, QTableWidgetItem(means_text))
