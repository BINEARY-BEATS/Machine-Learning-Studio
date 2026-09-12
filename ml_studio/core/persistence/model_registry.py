"""Model registry for versioned model storage."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ml_studio.app.logger import get_logger
from ml_studio.core.persistence.serializer import InferencePipeline, ModelVersion
from ml_studio.core.training.trainer import TrainingResult

logger = get_logger("model_registry")


class ModelRegistry:
    def __init__(self, base_dir: Path) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self.base_dir / "registry.json"
        self._models: dict[str, ModelVersion] = {}
        self._load_index()

    def _load_index(self) -> None:
        if self._index_path.exists():
            with self._index_path.open(encoding="utf-8") as f:
                data = json.load(f)
            for item in data.get("models", []):
                mv = ModelVersion.from_dict(item)
                self._models[mv.model_id] = mv

    def _save_index(self) -> None:
        with self._index_path.open("w", encoding="utf-8") as f:
            json.dump({"models": [m.to_dict() for m in self._models.values()]}, f, indent=2)

    def register(
        self,
        result: TrainingResult,
        name: str,
        dataset_id: str = "",
        dataset_version: int = 1,
        tags: list[str] | None = None,
        notes: str = "",
    ) -> ModelVersion:
        artifact_dir = self.base_dir / result.experiment_id
        pipeline = InferencePipeline(
            estimator=result.estimator,
            preprocessing=result.preprocessing,
            feature_columns=result.feature_columns,
            target_column=result.target_column,
            task=result.task,
        )
        pipeline.save(artifact_dir)

        version = ModelVersion(
            model_id=result.experiment_id,
            name=name,
            task=result.task,
            dataset_id=dataset_id,
            dataset_version=dataset_version,
            metrics=result.metrics,
            hyperparameters={"model_id": result.model_id},
            training_duration=result.training_duration,
            tags=tags or [],
            notes=notes,
            artifact_dir=str(artifact_dir),
        )
        self._models[version.model_id] = version
        self._save_index()
        logger.info("Registered model %s v%d", name, version.version)
        return version

    def load_pipeline(self, model_id: str) -> InferencePipeline:
        mv = self._models.get(model_id)
        if not mv:
            raise KeyError(f"Model not found: {model_id}")
        return InferencePipeline.load(Path(mv.artifact_dir))

    def list_models(self) -> list[ModelVersion]:
        return sorted(self._models.values(), key=lambda m: m.training_timestamp, reverse=True)

    def get(self, model_id: str) -> ModelVersion | None:
        return self._models.get(model_id)

    def archive(self, model_id: str) -> None:
        if model_id in self._models:
            self._models[model_id].tags.append("archived")
            self._save_index()
