"""Dependency injection container wiring core services to the GUI."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ml_studio.app.config import AppConfig

if TYPE_CHECKING:
    from ml_studio.core.project_manager import ProjectManager


@dataclass
class AppContainer:
    """Central service locator / DI container."""

    config: AppConfig
    project_manager: ProjectManager | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        from ml_studio.core.project_manager import ProjectManager

        self.project_manager = ProjectManager(self.config)


def create_container(config: AppConfig) -> AppContainer:
    return AppContainer(config=config)
