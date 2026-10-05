"""Single and batch prediction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from ml_studio.core.evaluation.explain import explain_single_prediction
from ml_studio.core.persistence.serializer import InferencePipeline


@dataclass
class PredictionResult:
    prediction: Any
    probabilities: dict[str, float] | list[float] | None = None
    explanation: dict[str, Any] | None = None


class Predictor:
    def __init__(self, pipeline: InferencePipeline) -> None:
        self.pipeline = pipeline

    def predict_single(self, features: dict[str, Any], explain: bool = False) -> PredictionResult:
        row = self._row_frame(features)
        pred = self.pipeline.predict(row)
        if hasattr(pred, "__len__") and not isinstance(pred, (str, bytes)):
            try:
                pred = pred[0]
            except Exception:
                pass
        proba = self._class_proba_dict(row)
        explanation = None
        if explain:
            try:
                X = self.pipeline.transform(row)
                explanation = explain_single_prediction(self.pipeline.estimator, X, 0)
            except Exception:
                explanation = None
        return PredictionResult(prediction=pred, probabilities=proba, explanation=explanation)

    def _row_frame(self, features: dict[str, Any]) -> pd.DataFrame:
        if self.pipeline.feature_schema:
            return self.pipeline.coerce_row(features)
        return pd.DataFrame([features])

    def _class_proba_dict(self, row: pd.DataFrame) -> dict[str, float] | None:
        model = self.pipeline.estimator
        if not hasattr(model, "predict_proba"):
            return None
        try:
            raw = model.predict_proba(self.pipeline.transform(row))[0]
        except Exception:
            return None
        classes = self.pipeline.target_classes
        if classes and len(classes) == len(raw):
            return {str(c): float(p) for c, p in zip(classes, raw)}
        if hasattr(model, "classes_"):
            return {str(c): float(p) for c, p in zip(model.classes_, raw)}
        return {str(i): float(p) for i, p in enumerate(raw)}

    def predict_batch(
        self,
        path: Path,
        output_path: Path | None = None,
        chunk_size: int = 10_000,
        progress_callback: Callable[[int, str], None] | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ) -> Path:
        from ml_studio.core.ingestion import LocalFileSource

        source = LocalFileSource(path)
        ext = path.suffix.lower()
        output = output_path or path.with_name(f"{path.stem}_predictions.csv")

        if ext == ".csv":
            chunks_written = False
            for i, chunk in enumerate(pd.read_csv(path, chunksize=chunk_size)):
                if cancel_check and cancel_check():
                    raise RuntimeError("Batch prediction cancelled")
                chunk_out = self._score_frame(chunk)
                chunk_out.to_csv(output, mode="w" if i == 0 else "a", header=i == 0, index=False)
                chunks_written = True
                if progress_callback:
                    progress_callback(min(99, (i + 1) * 10), f"Processed chunk {i+1}")
            if not chunks_written:
                df = source.load()
                self._score_frame(df).to_csv(output, index=False)
        else:
            df = source.load()
            if cancel_check and cancel_check():
                raise RuntimeError("Batch prediction cancelled")
            self._score_frame(df).to_csv(output, index=False)

        if progress_callback:
            progress_callback(100, "Complete")
        return output

    def _score_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["prediction"] = self.pipeline.predict(df)
        classes = self.pipeline.target_classes
        model = self.pipeline.estimator
        if classes and hasattr(model, "predict_proba"):
            try:
                proba = self.pipeline.predict_proba(df)
                for i, label in enumerate(classes):
                    name = str(label)
                    col = name if name not in out.columns else f"proba_{name}"
                    out[col] = proba[:, i]
            except Exception:
                pass
        return out
