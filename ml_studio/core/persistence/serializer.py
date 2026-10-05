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

from ml_studio.core.inference.schema_coerce import (
    build_feature_schema,
    coerce_value,
    decode_predictions,
)
from ml_studio.core.pipeline import Pipeline
from ml_studio.core.training.data_prep import EncodingBundle
from ml_studio.core.training.task import TaskType


@dataclass
class InferencePipeline:
    """Complete inference artifact: preprocessing + schema + estimator."""

    estimator: Any
    preprocessing: Pipeline | None
    feature_columns: list[str]
    target_column: str
    task: TaskType
    feature_schema: dict[str, Any] = field(default_factory=dict)
    target_classes: list[str] | None = None
    encoding: EncodingBundle | None = None
    input_feature_columns: list[str] = field(default_factory=list)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted prep (incl. AutoEncode). Never re-fit."""
        work = X.copy()
        raw_cols = self.input_feature_columns or list(self.feature_schema.keys()) or self.feature_columns
        if self.preprocessing is not None:
            ordered = [c for c in raw_cols if c in work.columns]
            if ordered:
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
        Xt = self.transform(X)
        if self.task in (TaskType.CLUSTERING, TaskType.ANOMALY_DETECTION):
            from ml_studio.core.training.unsupervised import predict_new

            return predict_new(self.estimator, Xt)
        raw = self.estimator.predict(Xt)
        return self.decode(raw)

    def predict_proba(self, X: pd.DataFrame):
        Xt = self.transform(X)
        if not hasattr(self.estimator, "predict_proba"):
            raise AttributeError("Estimator has no predict_proba")
        return self.estimator.predict_proba(Xt)

    def coerce_row(self, row: dict[str, Any]) -> pd.DataFrame:
        """Cast a raw input dict using feature_schema. Raises ValueError({field: msg})."""
        schema = self.feature_schema or {}
        cols = list(schema.keys()) or (self.input_feature_columns or self.feature_columns)
        errors: dict[str, str] = {}
        out: dict[str, Any] = {}
        for col in cols:
            spec = schema.get(col, {})
            kind = spec.get("kind", "categorical")
            raw = row.get(col, None)
            try:
                out[col] = coerce_value(kind, col, raw)
            except ValueError as exc:
                payload = exc.args[0] if exc.args else {col: "invalid value"}
                if isinstance(payload, dict):
                    errors.update(payload)
                else:
                    errors[col] = str(payload)
        if errors:
            raise ValueError(errors)
        return pd.DataFrame([out])

    def decode(self, preds: Any) -> Any:
        classes = self.target_classes
        if not classes and self.encoding is not None:
            return self.encoding.inverse_target(preds)
        return decode_predictions(preds, classes)

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
            "target_classes": self.target_classes,
            "has_encoding": self.encoding is not None,
            "has_target_encoder": bool(
                self.encoding and self.encoding.target_encoder is not None
            ),
        }
        with (directory / "metadata.json").open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, default=str)
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
    target_classes: list[str] | None = None
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
            "target_classes": self.target_classes,
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
            target_classes=data.get("target_classes"),
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


# Re-export for callers that built schema at train time
__all__ = [
    "InferencePipeline",
    "ModelVersion",
    "build_feature_schema",
]
