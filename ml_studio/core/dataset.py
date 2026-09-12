"""Dataset abstraction with versioning and lineage."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import pandas as pd

from ml_studio.app.logger import get_logger

logger = get_logger("dataset")


class DataSourceType(str, Enum):
    LOCAL_FILE = "local_file"
    REMOTE = "remote"
    SQL = "sql"
    MEMORY = "memory"


@dataclass(frozen=True)
class DatasetVersion:
    version: int
    row_count: int
    column_count: int
    memory_bytes: int
    created_at: datetime
    transformations: tuple[str, ...] = ()


@dataclass
class Dataset:
    """First-class dataset with identity, schema reference, and lineage."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "Untitled Dataset"
    source: str = ""
    source_type: DataSourceType = DataSourceType.LOCAL_FILE
    version: int = 1
    _dataframe: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)
    schema_id: str = ""
    lineage: list[str] = field(default_factory=list)
    transformations: list[str] = field(default_factory=list)
    statistics: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    modified_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    target_column: str | None = None
    is_sampled: bool = False

    @property
    def dataframe(self) -> pd.DataFrame:
        return self._dataframe

    @property
    def row_count(self) -> int:
        return len(self._dataframe)

    @property
    def column_count(self) -> int:
        return len(self._dataframe.columns)

    @property
    def memory_bytes(self) -> int:
        return int(self._dataframe.memory_usage(deep=True).sum())

    def cache_key(self, column: str, statistic: str) -> str:
        return f"{self.id}:{self.version}:{column}:{statistic}"

    def set_dataframe(self, df: pd.DataFrame, *, reason: str = "load") -> None:
        """Assign dataframe; only copies when CoW cannot guarantee isolation."""
        self._dataframe = df
        self.version += 1
        self.modified_at = datetime.now(timezone.utc)
        self.transformations.append(reason)
        self.statistics.clear()
        logger.debug("Dataset %s updated to v%d (%s)", self.id[:8], self.version, reason)

    def fork(self, df: pd.DataFrame, reason: str) -> Dataset:
        """Create a new dataset version derived from this one."""
        child = Dataset(
            name=f"{self.name} ({reason})",
            source=self.source,
            source_type=self.source_type,
            _dataframe=df,
            schema_id=self.schema_id,
            lineage=self.lineage + [self.id],
            transformations=[reason],
        )
        return child

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "source": self.source,
            "source_type": self.source_type.value,
            "version": self.version,
            "schema_id": self.schema_id,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "memory_bytes": self.memory_bytes,
            "target_column": self.target_column,
            "created_at": self.created_at.isoformat(),
            "modified_at": self.modified_at.isoformat(),
        }
