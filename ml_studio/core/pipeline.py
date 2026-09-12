"""Base transform and pipeline composition."""

from __future__ import annotations

import json
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline as SklearnPipeline


class TransformStep(ABC):
    name: str = "transform"
    description: str = ""

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> TransformStep:
        ...

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        ...

    def fit_transform(self, X: pd.DataFrame, y: pd.Series | None = None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    def to_sklearn(self) -> BaseEstimator:
        raise NotImplementedError(f"{self.name} does not support sklearn export")

    def get_params(self) -> dict[str, Any]:
        return {}

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "params": self.get_params()}


@dataclass
class PipelineNode:
    step: TransformStep
    status: str = "pending"
    node_id: str = field(default_factory=lambda: str(uuid.uuid4()))


class PreprocessingPipeline:
    """First-class serializable preprocessing pipeline."""

    def __init__(self, name: str = "default") -> None:
        self.id = str(uuid.uuid4())
        self.name = name
        self.nodes: list[PipelineNode] = []
        self._fitted = False

    def add(self, step: TransformStep) -> PreprocessingPipeline:
        self.nodes.append(PipelineNode(step=step))
        return self

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> PreprocessingPipeline:
        current = X
        for node in self.nodes:
            node.step.fit(current, y)
            node.status = "fitted"
            current = node.step.transform(current)
        self._fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("Pipeline must be fitted before transform")
        current = X
        for node in self.nodes:
            current = node.step.transform(current)
        return current

    def fit_transform(self, X: pd.DataFrame, y: pd.Series | None = None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    def to_sklearn_pipeline(self) -> SklearnPipeline:
        steps = [(n.step.name, n.step.to_sklearn()) for n in self.nodes]
        return SklearnPipeline(steps)

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> PreprocessingPipeline:
        return joblib.load(path)

    def describe(self) -> list[dict[str, Any]]:
        return [
            {
                "id": n.node_id,
                "name": n.step.name,
                "description": n.step.description,
                "status": n.status,
                "params": n.step.get_params(),
            }
            for n in self.nodes
        ]
