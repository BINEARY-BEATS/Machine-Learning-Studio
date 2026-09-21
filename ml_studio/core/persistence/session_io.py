"""Serialize / deserialize project session artifacts (dataset + pipeline)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from ml_studio.app.logger import get_logger
from ml_studio.core.dataset import DataSourceType, Dataset
from ml_studio.core.pipeline import Pipeline

logger = get_logger("session_io")

SESSION_FILE = "session.json"
DATASET_META = "datasets/meta.json"
DATASET_PARQUET = "datasets/data.parquet"
DATASET_JOBLIB = "datasets/data.joblib"
PIPELINE_FILE = "pipelines/prepare.json"


def write_session(
    staging: Path,
    *,
    dataset: Dataset | None,
    pipeline: Pipeline | None,
    schema_overrides: dict[str, str] | None,
) -> None:
    """Write session files into a staging directory before zipping."""
    (staging / "datasets").mkdir(parents=True, exist_ok=True)
    (staging / "pipelines").mkdir(parents=True, exist_ok=True)

    dataset_format = None
    if dataset is not None and not dataset.dataframe.empty:
        dataset_format = _write_dataframe(staging, dataset.dataframe)
        meta = dataset.to_dict()
        meta["dataframe_format"] = dataset_format
        with (staging / DATASET_META).open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    if pipeline is not None and pipeline.steps:
        with (staging / PIPELINE_FILE).open("w", encoding="utf-8") as f:
            json.dump(pipeline.to_dict(), f, indent=2)

    session = {
        "schema_overrides": schema_overrides or {},
        "has_dataset": dataset_format is not None,
        "has_pipeline": bool(pipeline and pipeline.steps),
        "dataset_format": dataset_format,
    }
    with (staging / SESSION_FILE).open("w", encoding="utf-8") as f:
        json.dump(session, f, indent=2)


def read_session(extract_dir: Path) -> dict[str, Any]:
    """Load session payload from an extracted .mlstudio directory.

    Returns keys: dataset, pipeline, schema_overrides (any may be None/empty).
    """
    schema_overrides: dict[str, str] = {}
    dataset: Dataset | None = None
    pipeline: Pipeline | None = None

    session_path = extract_dir / SESSION_FILE
    session: dict[str, Any] = {}
    if session_path.exists():
        with session_path.open(encoding="utf-8") as f:
            session = json.load(f) or {}
        schema_overrides = dict(session.get("schema_overrides") or {})

    meta_path = extract_dir / DATASET_META
    if meta_path.exists():
        with meta_path.open(encoding="utf-8") as f:
            meta = json.load(f)
        fmt = meta.get("dataframe_format") or session.get("dataset_format")
        df = _read_dataframe(extract_dir, fmt)
        if df is not None:
            dataset = Dataset(
                id=meta.get("id") or __import__("uuid").uuid4().hex,
                name=meta.get("name", "Dataset"),
                source=meta.get("source", ""),
                source_type=DataSourceType(meta.get("source_type", "memory")),
                version=int(meta.get("version", 1)),
                _dataframe=df,
                schema_id=meta.get("schema_id", ""),
                target_column=meta.get("target_column"),
            )

    pipeline_path = extract_dir / PIPELINE_FILE
    if pipeline_path.exists():
        with pipeline_path.open(encoding="utf-8") as f:
            pipeline = Pipeline.from_dict(json.load(f))

    return {
        "dataset": dataset,
        "pipeline": pipeline,
        "schema_overrides": schema_overrides,
    }


def _write_dataframe(staging: Path, df: pd.DataFrame) -> str:
    parquet_path = staging / DATASET_PARQUET
    try:
        df.to_parquet(parquet_path, index=False)
        return "parquet"
    except Exception as exc:
        logger.info("Parquet save unavailable (%s); using joblib", exc)
        import joblib

        joblib.dump(df, staging / DATASET_JOBLIB)
        return "joblib"


def _read_dataframe(extract_dir: Path, fmt: str | None) -> pd.DataFrame | None:
    parquet_path = extract_dir / DATASET_PARQUET
    joblib_path = extract_dir / DATASET_JOBLIB

    if fmt == "parquet" or (fmt is None and parquet_path.exists()):
        if parquet_path.exists():
            return pd.read_parquet(parquet_path)
    if fmt == "joblib" or joblib_path.exists():
        if joblib_path.exists():
            import joblib

            return joblib.load(joblib_path)
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    return None
