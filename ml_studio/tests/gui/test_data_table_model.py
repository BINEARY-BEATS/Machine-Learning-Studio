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
    
    # Check float formatting
    idx00 = model.index(0, 0)
    assert model.data(idx00, Qt.ItemDataRole.DisplayRole) == "1.23457"
    
    # Check NaN handling
    idx10 = model.index(1, 0)
    assert model.data(idx10, Qt.ItemDataRole.DisplayRole) == "∅"
    
    # Check string formatting
    idx01 = model.index(0, 1)
    assert model.data(idx01, Qt.ItemDataRole.DisplayRole) == "test"
    
    # Check text alignment (numeric is right aligned)
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
    
    # Horizontal headers
    assert model.headerData(0, Qt.Orientation.Horizontal, Qt.ItemDataRole.DisplayRole) == "colA"
    assert model.headerData(1, Qt.Orientation.Horizontal, Qt.ItemDataRole.DisplayRole) == "colB"
    
    # Vertical headers (1-indexed row numbers)
    assert model.headerData(0, Qt.Orientation.Vertical, Qt.ItemDataRole.DisplayRole) == "1"


def test_sort_by_column_reorders_rows():
    df = pd.DataFrame({"a": [3, 1, 2], "b": ["Z", "X", "Y"]})
    model = DataFrameTableModel(df)
    
    # Sort ascending by col 'a'
    model.sort(0, Qt.SortOrder.AscendingOrder)
    assert model.data(model.index(0, 0), Qt.ItemDataRole.DisplayRole) == "1"
    assert model.data(model.index(1, 0), Qt.ItemDataRole.DisplayRole) == "2"
    
    # Sort descending by col 'a'
    model.sort(0, Qt.SortOrder.DescendingOrder)
    assert model.data(model.index(0, 0), Qt.ItemDataRole.DisplayRole) == "3"
    assert model.data(model.index(2, 0), Qt.ItemDataRole.DisplayRole) == "1"


def test_filter_reduces_visible_rows():
    df = pd.DataFrame({"name": ["Alice", "Bob", "Charlie", "David"]})
    model = DataFrameTableModel(df)
    
    # Filter for 'a'
    model.apply_filter("a")
    assert model.rowCount() == 3  # Alice, Charlie, David
    
    # Verify values in filtered view
    assert model.data(model.index(0, 0), Qt.ItemDataRole.DisplayRole) == "Alice"
    assert model.data(model.index(1, 0), Qt.ItemDataRole.DisplayRole) == "Charlie"
    
    # Clear filter
    model.apply_filter("")
    assert model.rowCount() == 4


def test_large_dataframe_performance():
    df = pd.DataFrame({"a": np.random.randn(1000000)})
    model = DataFrameTableModel(df)
    
    # Measuring time to initialize and read a value (view time)
    start_time = time.time()
    
    # Data count
    count = model.rowCount()
    val = model.data(model.index(999999, 0), Qt.ItemDataRole.DisplayRole)
    
    elapsed = time.time() - start_time
    assert count == 1000000
    assert elapsed < 0.1  # Should be < 100ms
