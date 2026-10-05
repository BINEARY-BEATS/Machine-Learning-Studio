"""Durable experiment history stored alongside .mlstudio projects."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class ExperimentStore:
    """JSON experiment ledger next to a project file or inside a workspace dir."""

    def __init__(self, project_path: Path) -> None:
        self.project_path = Path(project_path)
        if self.project_path.suffix == ".mlstudio":
            self._path = self.project_path.with_suffix(".experiments.json")
        else:
            self._path = self.project_path / "experiments.json"

    def save_runs(self, runs: list[Any]) -> Path:
        payload = {"runs": [self._run_to_dict(r) for r in runs]}
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return self._path

    def load_runs(self) -> list[Any]:
        if not self._path.exists():
            return []
        from ml_studio.gui.pages.evaluate_page import ExperimentRun

        data = json.loads(self._path.read_text(encoding="utf-8"))
        runs = []
        for item in data.get("runs", []):
            ts = item.get("timestamp")
            if isinstance(ts, str):
                try:
                    ts = datetime.fromisoformat(ts)
                except ValueError:
                    ts = datetime.now(timezone.utc)
            else:
                ts = datetime.now(timezone.utc)
            runs.append(
                ExperimentRun(
                    run_id=item.get("run_id", ""),
                    model_name=item.get("model_name", ""),
                    model_id=item.get("model_id", ""),
                    task=item.get("task", ""),
                    dataset_name=item.get("dataset_name", ""),
                    target=item.get("target", ""),
                    metrics=item.get("metrics", {}),
                    cv_mean=item.get("cv_mean"),
                    cv_std=item.get("cv_std"),
                    duration_sec=float(item.get("duration_sec", 0)),
                    train_rows=int(item.get("train_rows", 0)),
                    test_rows=int(item.get("test_rows", 0)),
                    timestamp=ts,
                    registry_id=item.get("registry_id", ""),
                )
            )
        return runs

    @staticmethod
    def _run_to_dict(run: Any) -> dict[str, Any]:
        metrics = dict(getattr(run, "metrics", {}) or {})
        # Drop huge arrays from JSON ledger
        metrics.pop("confusion_matrix", None)
        return {
            "run_id": run.run_id,
            "model_name": run.model_name,
            "model_id": run.model_id,
            "task": run.task,
            "dataset_name": run.dataset_name,
            "target": run.target,
            "metrics": {
                k: (float(v) if hasattr(v, "item") else v)
                for k, v in metrics.items()
                if not hasattr(v, "shape") or getattr(v, "ndim", 0) == 0
            },
            "cv_mean": run.cv_mean,
            "cv_std": run.cv_std,
            "duration_sec": run.duration_sec,
            "train_rows": run.train_rows,
            "test_rows": run.test_rows,
            "timestamp": run.timestamp.isoformat()
            if hasattr(run.timestamp, "isoformat")
            else str(run.timestamp),
            "registry_id": run.registry_id,
        }
