"""Import preview dialog before committing a dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from ml_studio.app.theme_tokens import SPACE
from ml_studio.core.schema import infer_schema
from ml_studio.gui.widgets.card import Card


def preview_dataframe(path: Path, nrows: int = 100) -> pd.DataFrame:
    ext = path.suffix.lower()
    if ext == ".csv":
        return pd.read_csv(path, nrows=nrows)
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(path, nrows=nrows)
    if ext == ".json":
        return pd.read_json(path, nrows=nrows)
    if ext == ".parquet":
        return pd.read_parquet(path).head(nrows)
    if ext in (".feather", ".arrow"):
        return pd.read_feather(path).head(nrows)
    return pd.read_csv(path, nrows=nrows)


class ImportPreviewDialog(QDialog):
    """Show first rows and inferred schema before import."""

    def __init__(self, path: Path, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Import Preview — {path.name}")
        self.resize(760, 520)
        layout = QVBoxLayout(self)
        layout.setSpacing(SPACE[4])

        preview_df = preview_dataframe(path)
        schema = infer_schema(preview_df)
        layout.addWidget(
            QLabel(
                f"{path.name}  ·  preview {len(preview_df)} rows  ·  "
                f"{len(schema.columns)} columns"
            )
        )

        card = Card("Preview")
        table = QTableWidget()
        table.setRowCount(min(20, len(preview_df)))
        table.setColumnCount(len(preview_df.columns))
        table.setHorizontalHeaderLabels([str(c) for c in preview_df.columns])
        for r in range(table.rowCount()):
            for c, col in enumerate(preview_df.columns):
                val = preview_df.iloc[r, c]
                table.setItem(r, c, QTableWidgetItem("" if pd.isna(val) else str(val)))
        card.add_widget(table)
        layout.addWidget(card)

        schema_card = Card("Inferred schema")
        schema_text = ", ".join(f"{col.name} ({col.kind.value})" for col in schema.columns[:12])
        schema_card.add_widget(QLabel(schema_text))
        layout.addWidget(schema_card)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
