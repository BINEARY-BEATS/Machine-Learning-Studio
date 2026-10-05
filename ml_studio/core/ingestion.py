"""Data ingestion from multiple formats."""

from __future__ import annotations

import csv
import sqlite3
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd

from ml_studio.app.logger import get_logger
from ml_studio.core.dataset import DataSourceType, Dataset

logger = get_logger("ingestion")
_ROW_LIMIT = 1_000_000


@dataclass
class PreviewMeta:
    delimiter: str | None = None
    encoding: str | None = None
    sheets: list[str] = field(default_factory=list)
    tables: list[str] = field(default_factory=list)
    sheet: str | None = None
    table: str | None = None


class DataSource(ABC):
    @abstractmethod
    def load(self) -> pd.DataFrame:
        ...


def detect_encoding(path: Path) -> str:
    try:
        import chardet

        with path.open("rb") as f:
            raw = f.read(min(100_000, path.stat().st_size))
        if not raw:
            return "utf-8"
        result = chardet.detect(raw)
        enc = result.get("encoding") or "utf-8"
        if result.get("confidence", 0) > 0.7:
            return enc
    except ImportError:
        logger.debug("chardet not installed; using utf-8")
    return "utf-8"


def sniff_delimiter(path: Path, encoding: str = "utf-8") -> str:
    if path.suffix.lower() == ".tsv":
        return "\t"
    try:
        with path.open("r", encoding=encoding, errors="replace") as f:
            sample = f.read(65_536)
        if not sample.strip():
            return ","
        return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except (csv.Error, OSError):
        return ","


def list_sheets(path: Path) -> list[str]:
    xl = pd.ExcelFile(path)
    return list(xl.sheet_names)


def list_tables(path: Path) -> list[str]:
    conn = sqlite3.connect(path)
    try:
        names = pd.read_sql(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name", conn
        )
        return [str(n) for n in names["name"].tolist()] if not names.empty else []
    finally:
        conn.close()


def _cap(df: pd.DataFrame, sample: bool) -> pd.DataFrame:
    if sample and len(df) > _ROW_LIMIT:
        logger.warning("Dataset > %s rows. Truncating.", _ROW_LIMIT)
        return df.iloc[:_ROW_LIMIT]
    return df


class LocalFileSource(DataSource):
    SUPPORTED = {
        ".csv": "csv",
        ".tsv": "csv",
        ".xlsx": "excel",
        ".xls": "excel",
        ".json": "json",
        ".jsonl": "jsonl",
        ".parquet": "parquet",
        ".feather": "feather",
        ".arrow": "feather",
        ".orc": "orc",
        ".sqlite": "sqlite",
        ".db": "sqlite",
    }

    def __init__(
        self,
        path: Path,
        *,
        table: str | None = None,
        sheet: str | int | None = None,
        sep: str | None = None,
        encoding: str | None = None,
    ) -> None:
        self.path = Path(path)
        self.table = table
        self.sheet = sheet
        self.sep = sep
        self.encoding = encoding

    def _fmt(self) -> str:
        ext = self.path.suffix.lower()
        fmt = self.SUPPORTED.get(ext)
        if fmt is None:
            raise ValueError(f"Unsupported file type: {ext}")
        return fmt

    def _merge_opts(self, opts: dict[str, Any]) -> None:
        if "table" in opts and opts["table"] is not None:
            self.table = opts["table"]
        if "sheet" in opts and opts["sheet"] is not None:
            self.sheet = opts["sheet"]
        if "sep" in opts and opts["sep"] is not None:
            self.sep = opts["sep"]
        if "encoding" in opts and opts["encoding"] is not None:
            self.encoding = opts["encoding"]

    def load(self, sample: bool = True, **opts: Any) -> pd.DataFrame:
        if not self.path.exists():
            raise FileNotFoundError(str(self.path))
        self._merge_opts(opts)
        return getattr(self, f"_load_{self._fmt()}")(sample=sample)

    def preview(self, nrows: int = 100, **opts: Any) -> tuple[pd.DataFrame, PreviewMeta]:
        if not self.path.exists():
            raise FileNotFoundError(str(self.path))
        self._merge_opts(opts)
        fmt = self._fmt()
        meta = PreviewMeta()
        df = getattr(self, f"_preview_{fmt}")(nrows=nrows, meta=meta)
        return df, meta

    def _csv_kwargs(self, meta: PreviewMeta | None = None) -> dict[str, Any]:
        enc = self.encoding or detect_encoding(self.path)
        sep = self.sep or sniff_delimiter(self.path, enc)
        if meta is not None:
            meta.encoding = enc
            meta.delimiter = sep
        return {"encoding": enc, "sep": sep, "low_memory": False}

    def _load_csv(self, sample: bool = True) -> pd.DataFrame:
        kw = self._csv_kwargs()
        chunks: list[pd.DataFrame] = []
        total = 0
        for chunk in pd.read_csv(self.path, chunksize=100_000, **kw):
            if sample and total + len(chunk) > _ROW_LIMIT:
                chunks.append(chunk.iloc[: _ROW_LIMIT - total])
                break
            chunks.append(chunk)
            total += len(chunk)
        if not chunks:
            return pd.read_csv(self.path, nrows=0, encoding=kw["encoding"], sep=kw["sep"])
        return pd.concat(chunks, ignore_index=True)

    def _preview_csv(self, nrows: int, meta: PreviewMeta) -> pd.DataFrame:
        return pd.read_csv(self.path, nrows=nrows, **self._csv_kwargs(meta))

    def _load_parquet(self, sample: bool = True) -> pd.DataFrame:
        return _cap(pd.read_parquet(self.path), sample)

    def _preview_parquet(self, nrows: int, meta: PreviewMeta) -> pd.DataFrame:
        try:
            import pyarrow.parquet as pq

            pf = pq.ParquetFile(self.path)
            batches = []
            taken = 0
            for batch in pf.iter_batches(batch_size=min(nrows, 10_000)):
                batches.append(batch.to_pandas())
                taken += len(batches[-1])
                if taken >= nrows:
                    break
            if batches:
                return pd.concat(batches, ignore_index=True).head(nrows)
        except Exception:
            pass
        return pd.read_parquet(self.path).head(nrows)

    def _load_json(self, sample: bool = True) -> pd.DataFrame:
        return _cap(self._read_json_any(), sample)

    def _preview_json(self, nrows: int, meta: PreviewMeta) -> pd.DataFrame:
        return self._read_json_any(nrows=nrows)

    def _read_json_any(self, nrows: int | None = None) -> pd.DataFrame:
        try:
            df = pd.read_json(self.path)
            return df if nrows is None else df.head(nrows)
        except ValueError:
            return pd.read_json(self.path, lines=True, nrows=nrows)

    def _load_jsonl(self, sample: bool = True) -> pd.DataFrame:
        return _cap(pd.read_json(self.path, lines=True), sample)

    def _preview_jsonl(self, nrows: int, meta: PreviewMeta) -> pd.DataFrame:
        return pd.read_json(self.path, lines=True, nrows=nrows)

    def _load_feather(self, sample: bool = True) -> pd.DataFrame:
        return _cap(pd.read_feather(self.path), sample)

    def _preview_feather(self, nrows: int, meta: PreviewMeta) -> pd.DataFrame:
        return pd.read_feather(self.path).head(nrows)

    def _load_orc(self, sample: bool = True) -> pd.DataFrame:
        return _cap(pd.read_orc(self.path), sample)

    def _preview_orc(self, nrows: int, meta: PreviewMeta) -> pd.DataFrame:
        return pd.read_orc(self.path).head(nrows)

    def _load_excel(self, sample: bool = True) -> pd.DataFrame:
        return _cap(pd.read_excel(self.path, sheet_name=self.sheet or 0), sample)

    def _preview_excel(self, nrows: int, meta: PreviewMeta) -> pd.DataFrame:
        meta.sheets = list_sheets(self.path)
        sheet = self.sheet if self.sheet is not None else (meta.sheets[0] if meta.sheets else 0)
        meta.sheet = str(sheet)
        return pd.read_excel(self.path, sheet_name=sheet, nrows=nrows)

    def _resolve_table(self, meta: PreviewMeta | None = None) -> str:
        tables = list_tables(self.path)
        if meta is not None:
            meta.tables = tables
        if self.table:
            if meta is not None:
                meta.table = self.table
            return self.table
        if not tables:
            raise ValueError("SQLite database has no tables")
        if meta is not None:
            meta.table = tables[0]
        return tables[0]

    def _load_sqlite(self, sample: bool = True) -> pd.DataFrame:
        conn = sqlite3.connect(self.path)
        try:
            table = self._resolve_table()
            df = pd.read_sql(f'SELECT * FROM "{table}"', conn)
            return _cap(df, sample)
        finally:
            conn.close()

    def _preview_sqlite(self, nrows: int, meta: PreviewMeta) -> pd.DataFrame:
        conn = sqlite3.connect(self.path)
        try:
            table = self._resolve_table(meta)
            return pd.read_sql(f'SELECT * FROM "{table}" LIMIT {int(nrows)}', conn)
        finally:
            conn.close()


def remote_extension(url: str) -> str:
    return Path(urlparse(url).path).suffix.lower()


class RemoteFileSource(DataSource):
    def __init__(self, url: str) -> None:
        self.url = url

    def load(self, sample: bool = True) -> pd.DataFrame:
        import fsspec

        ext = remote_extension(self.url)
        storage_options = {"anon": True} if self.url.startswith("s3://") else {}
        with fsspec.open(self.url, "rb", **storage_options) as f:
            df = self._read_handle(f, ext, sample)
        return _cap(df, sample)

    def _read_handle(self, f, ext: str, sample: bool) -> pd.DataFrame:
        nrows = _ROW_LIMIT if sample else None
        if ext in (".csv", ".tsv"):
            sep = "\t" if ext == ".tsv" else ","
            return pd.read_csv(f, sep=sep, nrows=nrows)
        if ext == ".parquet":
            return pd.read_parquet(f)
        if ext == ".json":
            try:
                return pd.read_json(f)
            except ValueError:
                f.seek(0)
                return pd.read_json(f, lines=True, nrows=nrows)
        if ext == ".jsonl":
            return pd.read_json(f, lines=True, nrows=nrows)
        if ext in (".feather", ".arrow"):
            return pd.read_feather(f)
        if ext == ".orc":
            return pd.read_orc(f)
        if ext in (".xls", ".xlsx"):
            return pd.read_excel(f)
        raise ValueError(f"Unsupported remote format: {ext}")


class SqlSource(DataSource):
    def __init__(self, uri: str, query: str = "SELECT * FROM mytable") -> None:
        self.uri = uri
        self.query = query

    def load(self, sample: bool = True) -> pd.DataFrame:
        from sqlalchemy import create_engine

        engine = create_engine(self.uri)
        return _cap(pd.read_sql(self.query, engine), sample)


def load_dataset_from_path(
    path: Path,
    name: str | None = None,
    sample: bool = True,
    **opts: Any,
) -> Dataset:
    source = LocalFileSource(
        path,
        table=opts.get("table"),
        sheet=opts.get("sheet"),
        sep=opts.get("sep"),
        encoding=opts.get("encoding"),
    )
    df = source.load(sample=sample, **opts)
    ds = Dataset(
        name=name or path.stem,
        source=str(path.resolve()),
        source_type=DataSourceType.LOCAL_FILE,
    )
    ds.set_dataframe(df, reason="import")
    if sample and len(df) == _ROW_LIMIT:
        ds.is_sampled = True
    logger.info("Loaded dataset '%s' shape=%s", ds.name, df.shape)
    return ds


def load_dataset_from_url(
    url: str,
    name: str | None = None,
    sample: bool = True,
    query: str = "SELECT * FROM mytable",
) -> Dataset:
    if url.startswith(("sqlite://", "postgresql://", "mysql://", "snowflake://")):
        df = SqlSource(url, query).load(sample=sample)
        ds = Dataset(name=name or "sql_query", source=url, source_type=DataSourceType.SQL)
        ds.set_dataframe(df, reason="import_sql")
        if sample and len(df) == _ROW_LIMIT:
            ds.is_sampled = True
        return ds
    df = RemoteFileSource(url).load(sample=sample)
    ds = Dataset(
        name=name or Path(urlparse(url).path).stem or "remote",
        source=url,
        source_type=DataSourceType.REMOTE,
    )
    ds.set_dataframe(df, reason="import_remote")
    if sample and len(df) == _ROW_LIMIT:
        ds.is_sampled = True
    return ds
