"""Project metadata and session state."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ml_studio.core.dataset import Dataset
from ml_studio.core.pipeline import Pipeline


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class ProjectMetadata:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "Untitled Project"
    description: str = ""
    format_version: int = 2
    created_at: datetime = field(default_factory=_utcnow)
    modified_at: datetime = field(default_factory=_utcnow)
    settings: dict[str, Any] = field(default_factory=dict)

    def touch(self) -> None:
        self.modified_at = _utcnow()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "format_version": self.format_version,
            "created_at": self.created_at.isoformat(),
            "modified_at": self.modified_at.isoformat(),
            "settings": self.settings,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProjectMetadata:
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            name=data.get("name", "Untitled Project"),
            description=data.get("description", ""),
            format_version=data.get("format_version", 1),
            created_at=datetime.fromisoformat(data["created_at"])
            if "created_at" in data
            else _utcnow(),
            modified_at=datetime.fromisoformat(data["modified_at"])
            if "modified_at" in data
            else _utcnow(),
            settings=data.get("settings", {}),
        )


@dataclass
class Project:
    metadata: ProjectMetadata
    path: Path | None = None
    dirty: bool = False
    dataset: Dataset | None = None
    pipeline: Pipeline | None = None
    schema_overrides: dict[str, str] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.metadata.name

    @name.setter
    def name(self, value: str) -> None:
        self.metadata.name = value
        self.mark_dirty()

    def mark_dirty(self) -> None:
        self.dirty = True
        self.metadata.touch()

    def mark_clean(self) -> None:
        self.dirty = False

    def clear_session(self) -> None:
        self.dataset = None
        self.pipeline = None
        self.schema_overrides = {}
