"""Data ingestion from multiple formats."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

from ml_studio.app.logger import get_logger
from ml_studio.core.dataset import DataSourceType, Dataset

logger = get_logger("ingestion")


class DataSource(ABC):
    @abstractmethod
    def load(self) -> pd.DataFrame:
        ...


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

    def __init__(self, path: Path, *, table: str | None = None) -> None:
        self.path = Path(path)
        self.table = table

    def load(self, sample: bool = True) -> pd.DataFrame:
        if not self.path.exists():
            raise FileNotFoundError(str(self.path))
        ext = self.path.suffix.lower()
        fmt = self.SUPPORTED.get(ext)
        if fmt is None:
            raise ValueError(f"Unsupported file type: {ext}")

        loader = getattr(self, f"_load_{fmt}")
        return loader(sample=sample)

    def _load_csv(self, sample: bool = True) -> pd.DataFrame:
        encoding = self._detect_encoding()
        chunks: list[pd.DataFrame] = []
        total_rows = 0
        limit = 1_000_000

        for chunk in pd.read_csv(
            self.path, chunksize=100_000, encoding=encoding, low_memory=False
        ):
            if sample and total_rows + len(chunk) > limit:
                # Truncate the chunk to fit the limit
                chunk = chunk.iloc[: limit - total_rows]
                chunks.append(chunk)
                total_rows += len(chunk)
                logger.warning(f"Dataset > {limit} rows. Truncating to 1M rows for interactive session.")
                break
            chunks.append(chunk)
            total_rows += len(chunk)
            
        if not chunks:
            return pd.read_csv(self.path, nrows=0, encoding=encoding)
        return pd.concat(chunks, ignore_index=True)

    def _detect_encoding(self) -> str:
        try:
            import chardet

            with self.path.open("rb") as f:
                raw = f.read(min(100_000, self.path.stat().st_size))
            if not raw:
                return "utf-8"
            result = chardet.detect(raw)
            enc = result.get("encoding") or "utf-8"
            if result.get("confidence", 0) > 0.7:
                return enc
        except ImportError:
            logger.debug("chardet not installed; using utf-8")
        return "utf-8"

    def _load_parquet(self, sample: bool = True) -> pd.DataFrame:
        df = pd.read_parquet(self.path)
        if sample and len(df) > 1_000_000:
            logger.warning("Dataset > 1M rows. Truncating.")
            return df.iloc[:1_000_000]
        return df

    def _load_json(self, sample: bool = True) -> pd.DataFrame:
        df = pd.read_json(self.path)
        if sample and len(df) > 1_000_000:
            return df.iloc[:1_000_000]
        return df

    def _load_jsonl(self, sample: bool = True) -> pd.DataFrame:
        df = pd.read_json(self.path, lines=True)
        if sample and len(df) > 1_000_000:
            return df.iloc[:1_000_000]
        return df

    def _load_feather(self, sample: bool = True) -> pd.DataFrame:
        df = pd.read_feather(self.path)
        if sample and len(df) > 1_000_000:
            return df.iloc[:1_000_000]
        return df

    def _load_orc(self, sample: bool = True) -> pd.DataFrame:
        df = pd.read_orc(self.path)
        if sample and len(df) > 1_000_000:
            return df.iloc[:1_000_000]
        return df

    def _load_excel(self, sample: bool = True) -> pd.DataFrame:
        df = pd.read_excel(self.path)
        if sample and len(df) > 1_000_000:
            return df.iloc[:1_000_000]
        return df

    def _load_sqlite(self, sample: bool = True) -> pd.DataFrame:
        import sqlite3

        conn = sqlite3.connect(self.path)
        try:
            if self.table:
                query = f'SELECT * FROM "{self.table}"'
            else:
                tables = pd.read_sql(
                    "SELECT name FROM sqlite_master WHERE type='table'", conn
                )
                if tables.empty:
                    raise ValueError("SQLite database has no tables")
                table = tables.iloc[0]["name"]
                query = f'SELECT * FROM "{table}"'
            df = pd.read_sql(query, conn)
            if sample and len(df) > 1_000_000:
                return df.iloc[:1_000_000]
            return df
        finally:
            conn.close()


class RemoteFileSource(DataSource):
    def __init__(self, url: str) -> None:
        self.url = url

    def load(self, sample: bool = True) -> pd.DataFrame:
        import fsspec

        with fsspec.open(self.url, "rb") as f:
            path = Path(self.url)
            ext = path.suffix.lower()
            storage_options = {"anon": True} if self.url.startswith("s3://") else None
            
            if ext == ".csv":
                df = pd.read_csv(self.url, storage_options=storage_options) if not sample else pd.read_csv(self.url, nrows=1_000_000, storage_options=storage_options)
                return df
            elif ext == ".parquet":
                df = pd.read_parquet(self.url, storage_options=storage_options)
            elif ext == ".json":
                df = pd.read_json(self.url, storage_options=storage_options)
            elif ext == ".jsonl":
                df = pd.read_json(self.url, lines=True, storage_options=storage_options)
            elif ext == ".feather":
                df = pd.read_feather(self.url, storage_options=storage_options)
            elif ext == ".orc":
                df = pd.read_orc(self.url, storage_options=storage_options)
            elif ext in [".xls", ".xlsx"]:
                df = pd.read_excel(self.url, storage_options=storage_options)
            else:
                raise ValueError(f"Unsupported remote format: {ext}")
                
            if sample and len(df) > 1_000_000:
                return df.iloc[:1_000_000]
            return df


class SqlSource(DataSource):
    """
    Load data via SQLAlchemy.
    IMPLEMENTED but untested for Postgres. (Phase 2 backlog: test via Docker).
    """
    
    def __init__(self, uri: str, query: str = "SELECT * FROM mytable") -> None:
        self.uri = uri
        self.query = query
        
    def load(self, sample: bool = True) -> pd.DataFrame:
        from sqlalchemy import create_engine
        engine = create_engine(self.uri)
        
        # Use query if passed in via env or string, default is SELECT * FROM mytable (for test)
        df = pd.read_sql(self.query, engine)
        if sample and len(df) > 1_000_000:
            return df.iloc[:1_000_000]
        return df


def load_dataset_from_path(path: Path, name: str | None = None, sample: bool = True) -> Dataset:
    source = LocalFileSource(path)
    df = source.load(sample=sample)
    ds = Dataset(
        name=name or path.stem,
        source=str(path.resolve()),
        source_type=DataSourceType.LOCAL_FILE,
    )
    ds.set_dataframe(df, reason="import")
    if sample and len(df) == 1_000_000:
        ds.is_sampled = True
    logger.info("Loaded dataset '%s' shape=%s", ds.name, df.shape)
    return ds


def load_dataset_from_url(url: str, name: str | None = None, sample: bool = True, query: str = "SELECT * FROM mytable") -> Dataset:
    if url.startswith(("sqlite://", "postgresql://", "mysql://", "snowflake://")):
        source = SqlSource(url, query)
        df = source.load(sample=sample)
        ds = Dataset(
            name=name or "sql_query",
            source=url,
            source_type=DataSourceType.SQL,
        )
        ds.set_dataframe(df, reason="import_sql")
        if sample and len(df) == 1_000_000:
            ds.is_sampled = True
        return ds
        
    source = RemoteFileSource(url)
    df = source.load(sample=sample)
    ds = Dataset(
        name=name or Path(url).stem,
        source=url,
        source_type=DataSourceType.REMOTE,
    )
    ds.set_dataframe(df, reason="import_remote")
    if sample and len(df) == 1_000_000:
        ds.is_sampled = True
    return ds
