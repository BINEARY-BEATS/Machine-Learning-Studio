"""ML task type definitions."""

from __future__ import annotations

from enum import Enum


class TaskType(str, Enum):
    CLASSIFICATION = "CLASSIFICATION"
    REGRESSION = "REGRESSION"
    CLUSTERING = "CLUSTERING"
    ANOMALY_DETECTION = "ANOMALY_DETECTION"
    TIME_SERIES = "TIME_SERIES"

    @property
    def display_name(self) -> str:
        return self.value.replace("_", " ").title()
