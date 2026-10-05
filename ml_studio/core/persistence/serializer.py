"""Model and pipeline serialization."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from ml_studio.core.pipeline import Pipeline
from ml_studio.core.training.data_prep import EncodingBundle
from ml_studio.core.training.task import TaskType


@dataclass
class InferencePipeline:
    """Complete inference artifact: preprocessing + encoders + estimator + schema."""

    estimator: Any
    preprocessing: Pipeline | None
    feature_columns: list[str]
    target_column: str
    task: TaskType
    feature_schema: dict[str, Any] = field(default_factory=dict)
    encoding: EncodingBundle | None = None
    input_feature_columns: list[str] = field(default_factory=list)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted prep + feature encoders (never re-fit)."""
        work = X.copy()
        raw_cols = self.input_feature_columns or self.feature_columns
        if self.preprocessing is not None:
            present = [c for c in raw_cols if c in work.columns]
            if present:
                # Keep extra columns prep might need; prefer declared input order
                ordered = [c for c in raw_cols if c in work.columns]
                work = work[ordered]
            work = self.preprocessing.transform(work)
        elif raw_cols:
            ordered = [c for c in raw_cols if c in work.columns]
            if ordered:
                work = work[ordered]

        cols = self.feature_columns
        for col in cols:
            if col not in work.columns:
                work[col] = np.nan
        work = work[cols]
        if self.encoding is not None:
            work = self.encoding.transform_features(work)
        return work

    def predict(self, X: pd.DataFrame):
        """Predict and inverse-transform classification labels when encoded."""
        Xt = self.transform(X)
        raw = self.estimator.predict(Xt)
        if self.encoding is not None and self.encoding.target_encoder is not None:
            return self.encoding.inverse_target(raw)
        return raw

    def predict_proba(self, X: pd.DataFrame):
        Xt = self.transform(X)
        if not hasattr(self.estimator, "predict_proba"):
            raise AttributeError("Estimator has no predict_proba")
        return self.estimator.predict_proba(Xt)

    def save(self, directory: Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        artifact_path = directory / "pipeline.joblib"
        joblib.dump(self, artifact_path)
        meta = {
            "feature_columns": self.feature_columns,
            "input_feature_columns": self.input_feature_columns,
            "target_column": self.target_column,
            "task": self.task.value,
            "feature_schema": self.feature_schema,
            "has_encoding": self.encoding is not None,
            "has_target_encoder": bool(
                self.encoding and self.encoding.target_encoder is not None
            ),
            "encoded_features": list(self.encoding.feature_encoders.keys())
            if self.encoding
            else [],
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
    pipeline_hash: str = ""

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
            "pipeline_hash": self.pipeline_hash,
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
            pipeline_hash=data.get("pipeline_hash", ""),
        )
