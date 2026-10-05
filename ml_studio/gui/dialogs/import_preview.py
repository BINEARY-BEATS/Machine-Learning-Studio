"""Import preview dialog before committing a dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QMessageBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from ml_studio.app.theme_tokens import SPACE
from ml_studio.core.schema import infer_schema
from ml_studio.gui.widgets.card import Card


def _looks_like_jsonl(path: Path) -> bool:
    """Heuristic: first non-empty line is a JSON object and another line follows."""
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            first = ""
            for line in f:
                s = line.strip()
                if s:
                    first = s
                    break
            if not first.startswith("{"):
                return False
            for line in f:
                if line.strip():
                    return True
    except OSError:
        return False
    return path.suffix.lower() == ".jsonl"


def preview_dataframe(path: Path, nrows: int = 100) -> pd.DataFrame:
    """Load a small preview of a tabular file for the import dialog."""
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".csv":
        return pd.read_csv(path, nrows=nrows)
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(path, nrows=nrows)
    if ext == ".jsonl" or (ext == ".json" and _looks_like_jsonl(path)):
        # nrows requires lines=True for JSON Lines
        return pd.read_json(path, lines=True, nrows=nrows)
    if ext == ".json":
        # Regular JSON array/object — cannot use nrows; load then head
        return pd.read_json(path).head(nrows)
    if ext == ".parquet":
        return pd.read_parquet(path).head(nrows)
    if ext in (".feather", ".arrow"):
        return pd.read_feather(path).head(nrows)
    if ext == ".orc":
        return pd.read_orc(path).head(nrows)
    try:
        return pd.read_csv(path, nrows=nrows)
    except Exception:
        return pd.read_json(path, lines=True, nrows=nrows)


class ImportPreviewDialog(QDialog):
    """Show first rows and inferred schema before import."""

    def __init__(self, path: Path, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Import Preview — {path.name}")
        self.resize(760, 520)
        layout = QVBoxLayout(self)
        layout.setSpacing(SPACE[4])
        self._preview_failed = False

        try:
            preview_df = preview_dataframe(path)
        except Exception as exc:
            self._preview_failed = True
            QMessageBox.critical(
                parent,
                "Import preview failed",
                f"Could not preview {path.name}:\n{exc}",
            )
            layout.addWidget(QLabel(f"Preview failed for {path.name}."))
            buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
            buttons.rejected.connect(self.reject)
            layout.addWidget(buttons)
            return

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
