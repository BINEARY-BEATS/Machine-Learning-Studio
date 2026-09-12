"""Evaluation report generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ml_studio.core.evaluation.metrics import compute_metrics
from ml_studio.core.training.task import TaskType


@dataclass
class EvaluationReport:
    task: TaskType
    metrics: dict[str, Any]
    model_name: str
    dataset_name: str
    notes: str = ""

    def summary(self) -> str:
        lines = [f"Model: {self.model_name}", f"Dataset: {self.dataset_name}", f"Task: {self.task.value}", ""]
        for k, v in self.metrics.items():
            if k == "confusion_matrix":
                continue
            if isinstance(v, float):
                lines.append(f"  {k}: {v:.4f}")
            else:
                lines.append(f"  {k}: {v}")
        return "\n".join(lines)


def build_report(
    task: TaskType,
    model_name: str,
    dataset_name: str,
    y_true,
    y_pred,
    y_proba=None,
    X=None,
    model=None,
) -> EvaluationReport:
    metrics = compute_metrics(task, y_true, y_pred, y_proba, X, model)
    return EvaluationReport(task=task, metrics=metrics, model_name=model_name, dataset_name=dataset_name)
