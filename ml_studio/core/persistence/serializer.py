"""Model and pipeline serialization."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from ml_studio.core.pipeline import PreprocessingPipeline
from ml_studio.core.training.task import TaskType


@dataclass
class InferencePipeline:
    """Complete inference artifact: preprocessing + estimator + schema."""

    estimator: Any
    preprocessing: PreprocessingPipeline | None
    feature_columns: list[str]
    target_column: str
    task: TaskType
    feature_schema: dict[str, Any] = field(default_factory=dict)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X[self.feature_columns] if self.feature_columns else X
        if self.preprocessing:
            return self.preprocessing.transform(X)
        return X

    def predict(self, X: pd.DataFrame):
        return self.estimator.predict(self.transform(X))

    def save(self, directory: Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        artifact_path = directory / "pipeline.joblib"
        joblib.dump(self, artifact_path)
        meta = {
            "feature_columns": self.feature_columns,
            "target_column": self.target_column,
            "task": self.task.value,
            "feature_schema": self.feature_schema,
        }
        with (directory / "metadata.json").open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        return artifact_path

    @classmethod
    def load(cls, directory: Path) -> InferencePipeline:
        return joblib.load(Path(directory) / "pipeline.joblib")


@dataclass
class ModelVersion:
    model_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    version: int = 1
    task: TaskType = TaskType.REGRESSION
    dataset_id: str = ""
    dataset_version: int = 1
    feature_schema: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    training_timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    training_duration: float = 0.0
    tags: list[str] = field(default_factory=list)
    notes: str = ""
    artifact_dir: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "name": self.name,
            "version": self.version,
            "task": self.task.value,
            "dataset_id": self.dataset_id,
            "dataset_version": self.dataset_version,
            "feature_schema": self.feature_schema,
            "metrics": self.metrics,
            "hyperparameters": self.hyperparameters,
            "training_timestamp": self.training_timestamp.isoformat(),
            "training_duration": self.training_duration,
            "tags": self.tags,
            "notes": self.notes,
            "artifact_dir": self.artifact_dir,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelVersion:
        return cls(
            model_id=data["model_id"],
            name=data.get("name", ""),
            version=data.get("version", 1),
            task=TaskType(data.get("task", "REGRESSION")),
            dataset_id=data.get("dataset_id", ""),
            dataset_version=data.get("dataset_version", 1),
            feature_schema=data.get("feature_schema", {}),
            metrics=data.get("metrics", {}),
            hyperparameters=data.get("hyperparameters", {}),
            training_timestamp=datetime.fromisoformat(data["training_timestamp"])
            if "training_timestamp" in data
            else datetime.now(timezone.utc),
            training_duration=data.get("training_duration", 0),
            tags=data.get("tags", []),
            notes=data.get("notes", ""),
            artifact_dir=data.get("artifact_dir", ""),
        )
