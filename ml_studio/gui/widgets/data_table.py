"""Virtualized pandas DataFrame table model."""

from __future__ import annotations

import numpy as np
import pandas as pd
from PyQt6.QtCore import QAbstractTableModel, QModelIndex, Qt, QVariant


def _is_numeric_looking(text: str) -> bool:
    return any(ch.isdigit() for ch in text)


def _argsort_nan_last(series: pd.Series, ascending: bool) -> np.ndarray:
    """Stable argsort positions with NaN placed last."""
    s = series.reset_index(drop=True)
    na = s.isna().to_numpy()
    if not na.any():
        return s.sort_values(ascending=ascending, kind="mergesort").index.to_numpy(
            dtype=np.intp
        )
    valid = s.dropna()
    ordered = valid.sort_values(ascending=ascending, kind="mergesort").index.to_numpy(
        dtype=np.intp
    )
    return np.concatenate([ordered, np.flatnonzero(na).astype(np.intp)])


def _series_contains_literal(series: pd.Series, needle: str) -> np.ndarray:
    """Case-insensitive literal substring match; avoids slow full-frame astype."""
    needle_l = needle.lower()
    if pd.api.types.is_numeric_dtype(series):
        arr = series.to_numpy()
        na = pd.isna(arr)
        out = np.zeros(len(arr), dtype=bool)
        valid = ~na
        if not bool(valid.any()):
            return out
        # Numeric strings have no case — skip np.char.lower
        as_str = np.asarray(arr[valid], dtype=str)
        out[valid] = np.char.find(as_str, needle_l) >= 0
        return out
    as_str = series.fillna("").astype(str).to_numpy(dtype=str, copy=False)
    return np.char.find(np.char.lower(as_str), needle_l) >= 0


def _filter_mask(df: pd.DataFrame, text: str, column: str | None) -> np.ndarray:
    """Vectorized literal substring mask (case-insensitive)."""
    needle = text.strip()
    if not needle:
        return np.ones(len(df), dtype=bool)
    numeric_ok = _is_numeric_looking(needle)
    if column is not None:
        cols = [column] if column in df.columns else []
    else:
        cols = list(df.columns)
    masks: list[np.ndarray] = []
    for col in cols:
        series = df[col]
        if pd.api.types.is_numeric_dtype(series) and not numeric_ok:
            continue
        masks.append(_series_contains_literal(series, needle))
    if not masks:
        return np.zeros(len(df), dtype=bool)
    out = masks[0]
    for m in masks[1:]:
        out = out | m
    return out


class DataFrameTableModel(QAbstractTableModel):
    """Lazy QAbstractTableModel for large DataFrames."""

    def __init__(self, dataframe: pd.DataFrame | None = None, parent=None):
        super().__init__(parent)
        self._df = dataframe if dataframe is not None else pd.DataFrame()
        self._filter_text = ""
        self._filter_column: str | None = None
        n = len(self._df)
        self._order = np.arange(n, dtype=np.intp)
        self._visible = self._order.copy()

    def set_dataframe(self, df: pd.DataFrame) -> None:
        self.beginResetModel()
        self._df = df
        self._filter_text = ""
        self._filter_column = None
        n = len(df)
        self._order = np.arange(n, dtype=np.intp)
        self._visible = self._order.copy()
        self.endResetModel()

    @property
    def dataframe(self) -> pd.DataFrame:
        return self._df

    @property
    def total_rows(self) -> int:
        return len(self._df)

    @property
    def visible_rows(self) -> int:
        return len(self._visible)

    def rowCount(self, parent=QModelIndex()) -> int:
        return len(self._visible)

    def columnCount(self, parent=QModelIndex()) -> int:
        return len(self._df.columns)

    def _pos(self, row: int) -> int:
        return int(self._visible[row])

    def data(self, index: QModelIndex, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or self._df.empty:
            return QVariant()
        row, col = index.row(), index.column()
        if role == Qt.ItemDataRole.DisplayRole:
            try:
                value = self._df.iloc[self._pos(row), col]
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
        elif 0 <= section < len(self._visible):
            return str(self._pos(section) + 1)
        return QVariant()

    def sort(self, column: int, order: Qt.SortOrder) -> None:
        if self._df.empty or column >= len(self._df.columns) or len(self._visible) == 0:
            return
        col_name = self._df.columns[column]
        self.layoutAboutToBeChanged.emit()
        ascending = order == Qt.SortOrder.AscendingOrder
        series = self._df.iloc[self._visible][col_name]
        sorter = _argsort_nan_last(series, ascending=ascending)
        new_visible = self._visible[sorter]
        self._sync_order_after_visible_sort(new_visible)
        self._visible = new_visible
        self.layoutChanged.emit()

    def _sync_order_after_visible_sort(self, new_visible: np.ndarray) -> None:
        if len(new_visible) == len(self._order):
            self._order = new_visible.copy()
            return
        visible_set = set(new_visible.tolist())
        out = self._order.copy()
        vi = 0
        for i, pos in enumerate(self._order):
            if int(pos) in visible_set:
                out[i] = new_visible[vi]
                vi += 1
        self._order = out

    def apply_filter(self, text: str, column: str | None = None) -> None:
        self.beginResetModel()
        try:
            self._filter_text = (text or "").strip()
            self._filter_column = column
            if not self._filter_text or self._df.empty:
                self._visible = self._order.copy()
            else:
                mask = _filter_mask(self._df, self._filter_text, column)
                self._visible = self._order[mask[self._order]]
        except Exception:
            self._visible = self._order.copy()
        finally:
            self.endResetModel()
