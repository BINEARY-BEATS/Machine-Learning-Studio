"""Import preview dialog before committing a dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ml_studio.app.theme_tokens import SPACE
from ml_studio.core.ingestion import LocalFileSource
from ml_studio.core.schema import infer_schema
from ml_studio.gui.widgets.card import Card


def preview_dataframe(path: Path, nrows: int = 100, **opts: Any) -> pd.DataFrame:
    """Load a small preview via LocalFileSource (no pandas logic in GUI)."""
    df, _meta = LocalFileSource(path, **{
        k: opts[k] for k in ("table", "sheet", "sep", "encoding") if k in opts
    }).preview(nrows=nrows, **opts)
    return df


class ImportPreviewDialog(QDialog):
    """Show first rows and inferred schema before import."""

    def __init__(self, path: Path, parent=None) -> None:
        super().__init__(parent)
        self._path = Path(path)
        self.setWindowTitle(f"Import Preview — {self._path.name}")
        self.resize(780, 560)
        self._preview_failed = False
        self._options: dict[str, Any] = {}
        self._meta_delimiter: str | None = None
        self._meta_encoding: str | None = None

        self._layout = QVBoxLayout(self)
        self._layout.setSpacing(SPACE[4])
        self._summary = QLabel("")
        self._layout.addWidget(self._summary)

        opts_row = QHBoxLayout()
        self._sheet_combo = QComboBox()
        self._sheet_combo.hide()
        self._table_combo = QComboBox()
        self._table_combo.hide()
        self._sep_edit = QLineEdit()
        self._sep_edit.setPlaceholderText("Delimiter")
        self._sep_edit.setMaximumWidth(80)
        self._enc_label = QLabel("")
        self._enc_label.setObjectName("Breadcrumb")
        form = QFormLayout()
        form.addRow("Sheet", self._sheet_combo)
        form.addRow("Table", self._table_combo)
        form.addRow("Delimiter", self._sep_edit)
        form.addRow("Encoding", self._enc_label)
        opts_host = QWidget()
        opts_host.setLayout(form)
        opts_row.addWidget(opts_host)
        self._layout.addLayout(opts_row)

        self._table = QTableWidget()
        preview_card = Card("Preview")
        preview_card.add_widget(self._table)
        self._layout.addWidget(preview_card)

        self._schema_label = QLabel("")
        schema_card = Card("Inferred schema")
        schema_card.add_widget(self._schema_label)
        self._layout.addWidget(schema_card)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        self._layout.addWidget(buttons)

        self._sheet_combo.currentIndexChanged.connect(self._reload_preview)
        self._table_combo.currentIndexChanged.connect(self._reload_preview)
        self._sep_edit.editingFinished.connect(self._reload_preview)

        if not self._reload_preview(initial=True):
            self._preview_failed = True

    def chosen_options(self) -> dict[str, Any]:
        opts: dict[str, Any] = dict(self._options)
        sep = self._sep_edit.text()
        if sep:
            opts["sep"] = sep.replace("\\t", "\t")
        if self._sheet_combo.isVisible() and self._sheet_combo.currentText():
            opts["sheet"] = self._sheet_combo.currentText()
        if self._table_combo.isVisible() and self._table_combo.currentText():
            opts["table"] = self._table_combo.currentText()
        if self._meta_encoding:
            opts.setdefault("encoding", self._meta_encoding)
        return opts

    def _current_opts(self) -> dict[str, Any]:
        opts: dict[str, Any] = {}
        sep = self._sep_edit.text()
        if sep:
            opts["sep"] = sep.replace("\\t", "\t")
        if self._sheet_combo.isVisible() and self._sheet_combo.currentText():
            opts["sheet"] = self._sheet_combo.currentText()
        if self._table_combo.isVisible() and self._table_combo.currentText():
            opts["table"] = self._table_combo.currentText()
        return opts

    def _reload_preview(self, *_args, initial: bool = False) -> bool:
        try:
            source = LocalFileSource(self._path, **self._current_opts())
            df, meta = source.preview(nrows=100)
        except Exception as exc:
            if initial:
                QMessageBox.critical(
                    self.parent(),
                    "Import preview failed",
                    f"Could not preview {self._path.name}:\n{exc}",
                )
                self._summary.setText(f"Preview failed for {self._path.name}.")
            return False

        self._meta_delimiter = meta.delimiter
        self._meta_encoding = meta.encoding
        self._options = {
            k: v
            for k, v in {
                "sep": meta.delimiter,
                "encoding": meta.encoding,
                "sheet": meta.sheet,
                "table": meta.table,
            }.items()
            if v is not None
        }
        self._fill_combos(meta, initial=initial)
        if meta.delimiter and (initial or not self._sep_edit.text()):
            shown = "\\t" if meta.delimiter == "\t" else meta.delimiter
            self._sep_edit.blockSignals(True)
            self._sep_edit.setText(shown)
            self._sep_edit.blockSignals(False)
        self._enc_label.setText(meta.encoding or "—")
        self._summary.setText(
            f"{self._path.name}  ·  preview {len(df)} rows  ·  {len(df.columns)} columns"
            + (f"  ·  delimiter {meta.delimiter!r}" if meta.delimiter else "")
        )
        self._fill_table(df)
        schema = infer_schema(df)
        self._schema_label.setText(
            ", ".join(f"{col.name} ({col.kind.value})" for col in schema.columns[:12])
        )
        return True

    def _fill_combos(self, meta, *, initial: bool) -> None:
        if meta.sheets:
            self._sheet_combo.show()
            cur = self._sheet_combo.currentText()
            self._sheet_combo.blockSignals(True)
            self._sheet_combo.clear()
            self._sheet_combo.addItems(meta.sheets)
            if cur in meta.sheets:
                self._sheet_combo.setCurrentText(cur)
            elif meta.sheet:
                self._sheet_combo.setCurrentText(str(meta.sheet))
            self._sheet_combo.blockSignals(False)
        else:
            self._sheet_combo.hide()
        if meta.tables:
            self._table_combo.show()
            cur = self._table_combo.currentText()
            self._table_combo.blockSignals(True)
            self._table_combo.clear()
            self._table_combo.addItems(meta.tables)
            if cur in meta.tables:
                self._table_combo.setCurrentText(cur)
            elif meta.table:
                self._table_combo.setCurrentText(str(meta.table))
            self._table_combo.blockSignals(False)
        else:
            self._table_combo.hide()

    def _fill_table(self, preview_df: pd.DataFrame) -> None:
        self._table.setRowCount(min(20, len(preview_df)))
        self._table.setColumnCount(len(preview_df.columns))
        self._table.setHorizontalHeaderLabels([str(c) for c in preview_df.columns])
        for r in range(self._table.rowCount()):
            for c, col in enumerate(preview_df.columns):
                val = preview_df.iloc[r, c]
                self._table.setItem(
                    r, c, QTableWidgetItem("" if pd.isna(val) else str(val))
                )
