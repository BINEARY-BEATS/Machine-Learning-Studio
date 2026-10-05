"""B6 ingestion acceptance: TSV, semicolon CSV, sheets, tables, remote ext."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from ml_studio.core.ingestion import (
    LocalFileSource,
    load_dataset_from_path,
    remote_extension,
)


def test_tsv_loads_multiple_columns(tmp_path: Path):
    path = tmp_path / "data.tsv"
    path.write_text("a\tb\tc\n1\t2\t3\n4\t5\t6\n", encoding="utf-8")
    ds = load_dataset_from_path(path)
    assert list(ds.dataframe.columns) == ["a", "b", "c"]
    assert len(ds.dataframe) == 2


def test_semicolon_csv_sniffed(tmp_path: Path):
    path = tmp_path / "eu.csv"
    path.write_text("a;b;c\n1;2;3\n4;5;6\n", encoding="utf-8")
    ds = load_dataset_from_path(path)
    assert len(ds.dataframe.columns) == 3
    assert list(ds.dataframe.columns) == ["a", "b", "c"]


def test_json_and_jsonl_preview(tmp_path: Path):
    j = tmp_path / "arr.json"
    j.write_text('[{"x": 1}, {"x": 2}, {"x": 3}]', encoding="utf-8")
    df, _ = LocalFileSource(j).preview(nrows=2)
    assert len(df) == 2
    assert "x" in df.columns

    jl = tmp_path / "rows.jsonl"
    jl.write_text('{"x": 1}\n{"x": 2}\n{"x": 3}\n', encoding="utf-8")
    df2, _ = LocalFileSource(jl).preview(nrows=2)
    assert len(df2) == 2


def test_multi_sheet_xlsx(tmp_path: Path):
    pytest.importorskip("openpyxl")
    from ml_studio.core.ingestion import list_sheets

    path = tmp_path / "multi.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame({"a": [1, 2]}).to_excel(writer, sheet_name="one", index=False)
        pd.DataFrame({"b": [9, 8, 7]}).to_excel(writer, sheet_name="two", index=False)
    assert list_sheets(path) == ["one", "two"]
    df, meta = LocalFileSource(path, sheet="two").preview(nrows=10)
    assert list(df.columns) == ["b"]
    assert meta.sheet == "two"
    assert "two" in meta.sheets
    ds = load_dataset_from_path(path, sheet="two")
    assert list(ds.dataframe.columns) == ["b"]
    assert len(ds.dataframe) == 3


def test_multi_table_sqlite(tmp_path: Path):
    path = tmp_path / "db.sqlite"
    conn = sqlite3.connect(path)
    try:
        pd.DataFrame({"a": [1, 2]}).to_sql("t1", conn, index=False)
        pd.DataFrame({"b": [3, 4, 5]}).to_sql("t2", conn, index=False)
    finally:
        conn.close()
    from ml_studio.core.ingestion import list_tables

    assert set(list_tables(path)) == {"t1", "t2"}
    df, meta = LocalFileSource(path, table="t2").preview(nrows=10)
    assert list(df.columns) == ["b"]
    assert meta.table == "t2"
    ds = load_dataset_from_path(path, table="t2")
    assert len(ds.dataframe) == 3


@pytest.mark.parametrize("ext,writer", [
    (".csv", lambda p, df: df.to_csv(p, index=False)),
    (".tsv", lambda p, df: df.to_csv(p, sep="\t", index=False)),
    (".json", lambda p, df: df.to_json(p, orient="records")),
    (".jsonl", lambda p, df: df.to_json(p, orient="records", lines=True)),
    (".parquet", lambda p, df: df.to_parquet(p)),
    (".feather", lambda p, df: df.to_feather(p)),
])
def test_preview_every_common_extension(tmp_path: Path, ext: str, writer):
    if ext in (".parquet", ".feather"):
        pytest.importorskip("pyarrow")
    path = tmp_path / f"sample{ext}"
    writer(path, pd.DataFrame({"n": [1, 2, 3], "s": ["a", "b", "c"]}))
    df, _meta = LocalFileSource(path).preview(nrows=2)
    assert len(df) <= 2
    assert "n" in df.columns


def test_preview_excel_and_sqlite_extensions(tmp_path: Path):
    pytest.importorskip("openpyxl")
    xlsx = tmp_path / "s.xlsx"
    pd.DataFrame({"a": [1]}).to_excel(xlsx, index=False)
    df, _ = LocalFileSource(xlsx).preview(nrows=5)
    assert "a" in df.columns

    db = tmp_path / "s.db"
    conn = sqlite3.connect(db)
    pd.DataFrame({"a": [1]}).to_sql("t", conn, index=False)
    conn.close()
    df2, _ = LocalFileSource(db).preview(nrows=5)
    assert "a" in df2.columns


def test_preview_orc_if_available(tmp_path: Path):
    pytest.importorskip("pyarrow")
    try:
        path = tmp_path / "s.orc"
        pd.DataFrame({"a": [1, 2, 3]}).to_orc(path)
    except Exception as exc:
        pytest.skip(f"orc write unavailable: {exc}")
    df, _ = LocalFileSource(path).preview(nrows=2)
    assert "a" in df.columns


def test_remote_extension_strips_query():
    assert remote_extension("https://cdn.example.com/data/file.csv?token=abc") == ".csv"
    assert remote_extension("s3://bucket/path/data.parquet") == ".parquet"


def test_remote_load_uses_urlparse_extension():
    from ml_studio.core.ingestion import RemoteFileSource

    csv_bytes = b"a,b\n1,2\n"
    mock_f = MagicMock()
    mock_f.__enter__ = MagicMock(return_value=MagicMock(**{
        "read": MagicMock(),
        "seek": MagicMock(),
    }))
    # Use BytesIO-like via pandas reading from StringIO through our handle path
    import io

    handle = io.BytesIO(csv_bytes)
    with patch("fsspec.open", return_value=MagicMock(__enter__=lambda s: handle, __exit__=MagicMock())):
        src = RemoteFileSource("https://example.com/files/data.csv?token=xyz")
        df = src.load()
    assert list(df.columns) == ["a", "b"]
    assert len(df) == 1
