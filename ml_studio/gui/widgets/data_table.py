"""Virtualized pandas DataFrame table model."""

from __future__ import annotations

import re

import pandas as pd
import numpy as np
from PyQt6.QtCore import QAbstractTableModel, QModelIndex, Qt, QVariant


class DataFrameTableModel(QAbstractTableModel):
    """Lazy QAbstractTableModel for large DataFrames."""

    def __init__(self, dataframe: pd.DataFrame | None = None, parent=None):
        super().__init__(parent)
        self._df = dataframe if dataframe is not None else pd.DataFrame()
        self._filter_text = ""
        self._filtered_indices: list[int] | None = None

    def set_dataframe(self, df: pd.DataFrame) -> None:
        self.beginResetModel()
        self._df = df
        self._filtered_indices = None
        self._filter_text = ""
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()) -> int:
        if self._filtered_indices is not None:
            return len(self._filtered_indices)
        return len(self._df)

    def columnCount(self, parent=QModelIndex()) -> int:
        return len(self._df.columns)

    def _row_index(self, row: int) -> int:
        if self._filtered_indices is not None:
            return self._filtered_indices[row]
        return row

    def data(self, index: QModelIndex, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or self._df.empty:
            return QVariant()
        row, col = index.row(), index.column()
        if role == Qt.ItemDataRole.DisplayRole:
            try:
                value = self._df.iloc[self._row_index(row), col]
                if pd.isna(value):
                    return "∅"
                if isinstance(value, (float, np.floating)):
                    return f"{float(value):.6g}"
                if isinstance(value, (int, np.integer)):
                    return str(int(value))
                return str(value)
            except IndexError:
                return QVariant()
        if role == Qt.ItemDataRole.TextAlignmentRole:
            col_name = self._df.columns[col]
            if pd.api.types.is_numeric_dtype(self._df[col_name]):
                return int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        return QVariant()

    def headerData(self, section: int, orientation: Qt.Orientation, role=Qt.ItemDataRole.DisplayRole):
        if role != Qt.ItemDataRole.DisplayRole:
            return QVariant()
        if orientation == Qt.Orientation.Horizontal:
            if 0 <= section < len(self._df.columns):
                return str(self._df.columns[section])
        else:
            return str(self._row_index(section) + 1)
        return QVariant()

    def sort(self, column: int, order: Qt.SortOrder) -> None:
        if self._df.empty or column >= len(self._df.columns):
            return
        col_name = self._df.columns[column]
        self.layoutAboutToBeChanged.emit()
        ascending = order == Qt.SortOrder.AscendingOrder
        self._df = self._df.sort_values(by=col_name, ascending=ascending, kind="mergesort")
        self._filtered_indices = None
        self.layoutChanged.emit()

    def apply_filter(self, text: str) -> None:
        self.beginResetModel()
        try:
            self._filter_text = text.strip().lower()
            if not self._filter_text or self._df.empty:
                self._filtered_indices = None
            else:
                # Literal substring match — never treat user input as regex
                str_df = self._df.astype(str)
                mask = str_df.apply(
                    lambda row: row.str.lower().str.contains(
                        re.escape(self._filter_text),
                        na=False,
                        regex=True,
                    ).any(),
                    axis=1,
                )
                self._filtered_indices = list(self._df.index[mask])
        except Exception:
            self._filtered_indices = None
        finally:
            self.endResetModel()

    @property
    def dataframe(self) -> pd.DataFrame:
        return self._df
