"""Tests for import preview helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ml_studio.gui.dialogs.import_preview import preview_dataframe


def test_preview_jsonl(tmp_path: Path):
    path = tmp_path / "rows.jsonl"
    path.write_text(
        '{"a": 1, "b": "x"}\n{"a": 2, "b": "y"}\n{"a": 3, "b": "z"}\n',
        encoding="utf-8",
    )
    df = preview_dataframe(path, nrows=2)
    assert len(df) == 2
    assert list(df.columns) == ["a", "b"]


def test_preview_json_array(tmp_path: Path):
    path = tmp_path / "rows.json"
    path.write_text('[{"a": 1}, {"a": 2}, {"a": 3}]', encoding="utf-8")
    df = preview_dataframe(path, nrows=2)
    assert len(df) == 2
    assert "a" in df.columns


def test_preview_json_misnamed_jsonl(tmp_path: Path):
    """A .json file that is actually JSON Lines must not crash on nrows."""
    path = tmp_path / "fake.json"
    path.write_text('{"a": 1}\n{"a": 2}\n{"a": 3}\n', encoding="utf-8")
    df = preview_dataframe(path, nrows=2)
    assert len(df) == 2


def test_preview_csv(tmp_path: Path):
    path = tmp_path / "t.csv"
    pd.DataFrame({"x": range(50)}).to_csv(path, index=False)
    df = preview_dataframe(path, nrows=10)
    assert len(df) == 10
