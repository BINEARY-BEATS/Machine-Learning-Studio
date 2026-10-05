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
    probabilities: list[float] | None = None
    explanation: dict[str, Any] | None = None


class Predictor:
    def __init__(self, pipeline: InferencePipeline) -> None:
        self.pipeline = pipeline

    def predict_single(self, features: dict[str, Any], explain: bool = False) -> PredictionResult:
        row = pd.DataFrame([features])
        # Use InferencePipeline.predict for encode + inverse target labels
        pred = self.pipeline.predict(row)
        if hasattr(pred, "__len__") and not isinstance(pred, (str, bytes)):
            try:
                pred = pred[0]
            except Exception:
                pass
        proba = None
        model = self.pipeline.estimator
        if hasattr(model, "predict_proba"):
            try:
                X = self.pipeline.transform(row)
                proba = model.predict_proba(X)[0].tolist()
            except Exception:
                pass
        explanation = None
        if explain:
            try:
                X = self.pipeline.transform(row)
                explanation = explain_single_prediction(model, X, 0)
            except Exception:
                explanation = None
        return PredictionResult(prediction=pred, probabilities=proba, explanation=explanation)

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
                preds = self.pipeline.predict(chunk)
                chunk_out = chunk.copy()
                chunk_out["prediction"] = preds
                chunk_out.to_csv(output, mode="w" if i == 0 else "a", header=i == 0, index=False)
                chunks_written = True
                if progress_callback:
                    progress_callback(min(99, (i + 1) * 10), f"Processed chunk {i+1}")
            if not chunks_written:
                df = source.load()
                df["prediction"] = self.pipeline.predict(df)
                df.to_csv(output, index=False)
        else:
            df = source.load()
            if cancel_check and cancel_check():
                raise RuntimeError("Batch prediction cancelled")
            df["prediction"] = self.pipeline.predict(df)
            df.to_csv(output, index=False)

        if progress_callback:
            progress_callback(100, "Complete")
        return output
