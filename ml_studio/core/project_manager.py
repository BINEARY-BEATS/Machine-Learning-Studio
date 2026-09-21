"""Project lifecycle: create, open, save, autosave, recent projects."""

from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path
from typing import Callable

from ml_studio.app.config import AppConfig
from ml_studio.app.logger import get_logger
from ml_studio.core.persistence.session_io import read_session, write_session
from ml_studio.core.pipeline import Pipeline
from ml_studio.core.project import Project, ProjectMetadata

logger = get_logger("project_manager")

RECENT_FILE = "recent_projects.json"
MANIFEST = "manifest.json"


class ProjectManager:
    """Manages project files (.mlstudio versioned archives)."""

    SUPPORTED_FORMAT_VERSION = 2

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.current: Project | None = None
        self._recent_path = Path.home() / ".mlstudio" / RECENT_FILE
        self._listeners: list[Callable[[Project | None], None]] = []

    def subscribe(self, callback: Callable[[Project | None], None]) -> None:
        self._listeners.append(callback)

    def _notify(self) -> None:
        for cb in self._listeners:
            cb(self.current)

    def new_project(self, name: str = "Untitled Project") -> Project:
        meta = ProjectMetadata(name=name, format_version=self.SUPPORTED_FORMAT_VERSION)
        project = Project(metadata=meta)
        self.current = project
        self._notify()
        logger.info("Created new project: %s", name)
        return project

    def open_project(self, path: Path) -> Project:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Project not found: {path}")
        if path.suffix != self.config.project_extension:
            raise ValueError(f"Expected {self.config.project_extension} file")

        extract_dir = path.parent / f".{path.stem}_cache"
        if extract_dir.exists():
            shutil.rmtree(extract_dir)
        extract_dir.mkdir(parents=True)

        with zipfile.ZipFile(path, "r") as zf:
            zf.extractall(extract_dir)

        manifest_path = extract_dir / MANIFEST
        if not manifest_path.exists():
            raise ValueError("Invalid project: missing manifest.json")

        with manifest_path.open(encoding="utf-8") as f:
            manifest = json.load(f)

        fmt = manifest.get("format_version", 0)
        if fmt > self.SUPPORTED_FORMAT_VERSION:
            raise ValueError(
                f"Project format v{fmt} is newer than supported "
                f"v{self.SUPPORTED_FORMAT_VERSION}. "
                "Please upgrade Machine Learning Studio."
            )

        meta = ProjectMetadata.from_dict(manifest)
        project = Project(metadata=meta, path=path)

        try:
            session = read_session(extract_dir)
            project.dataset = session.get("dataset")
            project.pipeline = session.get("pipeline") or Pipeline()
            project.schema_overrides = session.get("schema_overrides") or {}
        except Exception as exc:
            logger.warning("Could not fully restore session from %s: %s", path, exc)
            project.pipeline = Pipeline()

        project.mark_clean()
        self.current = project
        self._add_recent(path)
        self._notify()
        logger.info("Opened project: %s", path)
        return project

    def save_project(self, path: Path | None = None) -> Path:
        if self.current is None:
            raise RuntimeError("No project open")
        save_path = Path(path) if path else self.current.path
        if save_path is None:
            raise ValueError("No save path specified")

        save_path = Path(save_path)
        if save_path.suffix != self.config.project_extension:
            save_path = save_path.with_suffix(self.config.project_extension)

        staging = save_path.parent / f".{save_path.stem}_staging"
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)

        for sub in ("datasets", "models", "experiments", "pipelines"):
            (staging / sub).mkdir(exist_ok=True)

        self.current.metadata.format_version = self.SUPPORTED_FORMAT_VERSION
        manifest = self.current.metadata.to_dict()
        with (staging / MANIFEST).open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        settings = self.current.metadata.settings
        with (staging / "settings.json").open("w", encoding="utf-8") as f:
            json.dump(settings, f, indent=2)

        write_session(
            staging,
            dataset=self.current.dataset,
            pipeline=self.current.pipeline,
            schema_overrides=self.current.schema_overrides,
        )

        with zipfile.ZipFile(save_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for file_path in staging.rglob("*"):
                if file_path.is_file():
                    zf.write(file_path, file_path.relative_to(staging))

        shutil.rmtree(staging)
        self.current.path = save_path
        self.current.mark_clean()
        self._add_recent(save_path)
        self._notify()
        logger.info("Saved project: %s", save_path)
        return save_path

    def close_project(self) -> None:
        self.current = None
        self._notify()

    def get_recent_projects(self, limit: int = 10) -> list[Path]:
        if not self._recent_path.exists():
            return []
        with self._recent_path.open(encoding="utf-8") as f:
            data = json.load(f)
        valid = [Path(p) for p in data if Path(p).exists()]
        return valid[:limit]

    def _add_recent(self, path: Path) -> None:
        self._recent_path.parent.mkdir(parents=True, exist_ok=True)
        recent = self.get_recent_projects(limit=50)
        path = path.resolve()
        recent = [p for p in recent if p.resolve() != path]
        recent.insert(0, path)
        with self._recent_path.open("w", encoding="utf-8") as f:
            json.dump([str(p) for p in recent[:20]], f, indent=2)
