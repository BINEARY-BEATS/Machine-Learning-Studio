import pytest
import pandas as pd
import numpy as np
import time
from PyQt6.QtCore import Qt

from ml_studio.gui.widgets.data_table import DataFrameTableModel


def test_row_count_matches_dataframe():
    df = pd.DataFrame({"a": range(100), "b": range(100)})
    model = DataFrameTableModel(df)
    assert model.rowCount() == 100
    assert model.columnCount() == 2


def test_data_returns_correct_value():
    df = pd.DataFrame({"a": [1.2345678, np.nan], "b": ["test", "foo"]})
    model = DataFrameTableModel(df)

    idx00 = model.index(0, 0)
    assert model.data(idx00, Qt.ItemDataRole.DisplayRole) == "1.23457"

    idx10 = model.index(1, 0)
    assert model.data(idx10, Qt.ItemDataRole.DisplayRole) == "∅"

    idx01 = model.index(0, 1)
    assert model.data(idx01, Qt.ItemDataRole.DisplayRole) == "test"

    align_numeric = model.data(idx00, Qt.ItemDataRole.TextAlignmentRole)
    assert align_numeric == int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

    align_text = model.data(idx01, Qt.ItemDataRole.TextAlignmentRole)
    if hasattr(align_text, "isNull"):
        assert align_text.isNull()
    else:
        assert align_text is None


def test_header_returns_correct_label():
    df = pd.DataFrame({"colA": [1], "colB": [2]})
    model = DataFrameTableModel(df)

    assert model.headerData(0, Qt.Orientation.Horizontal, Qt.ItemDataRole.DisplayRole) == "colA"
    assert model.headerData(1, Qt.Orientation.Horizontal, Qt.ItemDataRole.DisplayRole) == "colB"
    assert model.headerData(0, Qt.Orientation.Vertical, Qt.ItemDataRole.DisplayRole) == "1"


def test_sort_by_column_reorders_rows():
    df = pd.DataFrame({"a": [3, 1, 2], "b": ["Z", "X", "Y"]})
    model = DataFrameTableModel(df)

    model.sort(0, Qt.SortOrder.AscendingOrder)
    assert model.data(model.index(0, 0), Qt.ItemDataRole.DisplayRole) == "1"
    assert model.data(model.index(1, 0), Qt.ItemDataRole.DisplayRole) == "2"

    model.sort(0, Qt.SortOrder.DescendingOrder)
    assert model.data(model.index(0, 0), Qt.ItemDataRole.DisplayRole) == "3"
    assert model.data(model.index(2, 0), Qt.ItemDataRole.DisplayRole) == "1"


def test_filter_reduces_visible_rows():
    df = pd.DataFrame({"name": ["Alice", "Bob", "Charlie", "David"]})
    model = DataFrameTableModel(df)

    model.apply_filter("a")
    assert model.rowCount() == 3  # Alice, Charlie, David

    assert model.data(model.index(0, 0), Qt.ItemDataRole.DisplayRole) == "Alice"
    assert model.data(model.index(1, 0), Qt.ItemDataRole.DisplayRole) == "Charlie"

    model.apply_filter("")
    assert model.rowCount() == 4


def test_sort_filter_non_range_index_correct_values():
    """Shuffled non-RangeIndex: sort + filter must use iloc positions, not labels."""
    df = pd.DataFrame(
        {"val": [30, 10, 20, 40], "tag": ["d", "b", "c", "a"]},
        index=[100, 200, 300, 400],
    )
    df = df.iloc[[2, 0, 3, 1]].copy()
    assert list(df.index) == [300, 100, 400, 200]
    model = DataFrameTableModel(df)

    model.sort(0, Qt.SortOrder.AscendingOrder)
    assert model.data(model.index(0, 0), Qt.ItemDataRole.DisplayRole) == "10"
    assert model.headerData(0, Qt.Orientation.Vertical, Qt.ItemDataRole.DisplayRole) == "4"

    model.apply_filter("a", column="tag")
    assert model.rowCount() == 1
    assert model.data(model.index(0, 0), Qt.ItemDataRole.DisplayRole) == "40"
    assert model.data(model.index(0, 1), Qt.ItemDataRole.DisplayRole) == "a"
    assert model.headerData(0, Qt.Orientation.Vertical, Qt.ItemDataRole.DisplayRole) == "3"

    assert list(model.dataframe["val"]) == [20, 30, 40, 10]


def test_column_restricted_filter():
    df = pd.DataFrame({"name": ["Alice", "Bob"], "city": ["Austin", "Boston"]})
    model = DataFrameTableModel(df)
    model.apply_filter("a", column="city")
    assert model.rowCount() == 1
    assert model.data(model.index(0, 0), Qt.ItemDataRole.DisplayRole) == "Alice"
    model.apply_filter("a", column="name")
    assert model.rowCount() == 1
    assert model.data(model.index(0, 0), Qt.ItemDataRole.DisplayRole) == "Alice"


def test_sort_preserves_filter():
    df = pd.DataFrame({"a": [3, 1, 2, 4], "b": ["x", "y", "x", "z"]})
    model = DataFrameTableModel(df)
    model.apply_filter("x", column="b")
    assert model.rowCount() == 2
    model.sort(0, Qt.SortOrder.AscendingOrder)
    assert model.rowCount() == 2
    assert model.data(model.index(0, 0), Qt.ItemDataRole.DisplayRole) == "2"
    assert model.data(model.index(1, 0), Qt.ItemDataRole.DisplayRole) == "3"


def test_nan_sorts_last():
    df = pd.DataFrame({"a": [2.0, np.nan, 1.0]})
    model = DataFrameTableModel(df)
    model.sort(0, Qt.SortOrder.AscendingOrder)
    assert model.data(model.index(0, 0), Qt.ItemDataRole.DisplayRole) == "1"
    assert model.data(model.index(1, 0), Qt.ItemDataRole.DisplayRole) == "2"
    assert model.data(model.index(2, 0), Qt.ItemDataRole.DisplayRole) == "∅"


def test_large_dataframe_performance():
    df = pd.DataFrame({"a": np.random.randn(1000000)})
    model = DataFrameTableModel(df)

    start_time = time.time()
    count = model.rowCount()
    val = model.data(model.index(999999, 0), Qt.ItemDataRole.DisplayRole)
    elapsed = time.time() - start_time
    assert count == 1000000
    assert elapsed < 0.1
    assert val is not None


@pytest.mark.slow
def test_filter_1m_rows_under_3s():
    rng = np.random.RandomState(0)
    df = pd.DataFrame({f"c{i}": rng.randn(1_000_000) for i in range(10)})
    df["label"] = rng.choice(["alpha", "beta", "gamma"], size=1_000_000)
    model = DataFrameTableModel(df)
    start = time.perf_counter()
    model.apply_filter("alpha", column="label")
    elapsed = time.perf_counter() - start
    assert model.rowCount() > 0
    assert elapsed < 3.0, f"filter took {elapsed:.2f}s"
