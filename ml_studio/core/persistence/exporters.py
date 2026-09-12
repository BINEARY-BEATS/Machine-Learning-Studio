"""Model export to various formats."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib

from ml_studio.core.persistence.serializer import InferencePipeline


class ExportError(Exception):
    pass


def export_joblib(pipeline: InferencePipeline, path: Path) -> Path:
    joblib.dump(pipeline, path)
    return path


def export_pickle(pipeline: InferencePipeline, path: Path) -> Path:
    import pickle

    with path.open("wb") as f:
        pickle.dump(pipeline, f)
    return path


def export_onnx(pipeline: InferencePipeline, path: Path) -> Path:
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
    except ImportError as e:
        raise ExportError(
            "ONNX export requires skl2onnx and onnx. pip install skl2onnx onnx"
        ) from e

    n_features = len(pipeline.feature_columns)
    initial_type = [("float_input", FloatTensorType([None, n_features]))]
    try:
        onnx_model = convert_sklearn(pipeline.estimator, initial_types=initial_type)
        with path.open("wb") as f:
            f.write(onnx_model.SerializeToString())
        return path
    except Exception as e:
        raise ExportError(
            f"Cannot export {type(pipeline.estimator).__name__} to ONNX: {e}"
        ) from e


def export_requirements(path: Path, extras: list[str] | None = None) -> Path:
    lines = ["scikit-learn>=1.3", "pandas>=2.0", "numpy>=1.24", "joblib>=1.3"]
    if extras:
        lines.extend(extras)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def export_model(
    pipeline: InferencePipeline,
    path: Path,
    fmt: str,
) -> Path:
    exporters = {
        "joblib": export_joblib,
        "pickle": export_pickle,
        "onnx": export_onnx,
    }
    if fmt not in exporters:
        raise ExportError(f"Unsupported export format: {fmt}")
    return exporters[fmt](pipeline, path)
