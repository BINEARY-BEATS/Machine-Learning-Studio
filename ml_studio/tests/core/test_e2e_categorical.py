"""E2E categorical correctness smoke: train → register → predict → reopen metrics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml_studio.core.inference.predictor import Predictor
from ml_studio.core.persistence.experiments import ExperimentStore
from ml_studio.core.persistence.model_registry import ModelRegistry
from ml_studio.core.training.task import TaskType
from ml_studio.core.training.trainer import Trainer, TrainingConfig
from ml_studio.gui.pages.evaluate_page import ExperimentRun


def test_categorical_e2e_train_predict_persist(tmp_path: Path):
    rng = np.random.RandomState(7)
    n = 120
    color = rng.choice(["red", "blue", "green"], n)
    label = np.where(color == "red", "yes", np.where(color == "blue", "no", "maybe"))
    df = pd.DataFrame(
        {
            "color": color,
            "size": rng.choice(["S", "M", "L"], n),
            "x": rng.randn(n),
            "label": label,
        }
    )
    config = TrainingConfig(
        task=TaskType.CLASSIFICATION,
        model_id="logistic_regression",
        target_column="label",
        feature_columns=["color", "size", "x"],
        test_size=0.25,
        random_state=7,
        cv_splits=3,
    )
    result = Trainer().train(df, config)
    assert result.target_classes is not None
    assert result.encoding is not None
    assert result.encoding.target_encoder is not None
    assert "accuracy" in result.metrics or "f1" in result.metrics

    registry = ModelRegistry(tmp_path / "models")
    mv = registry.register(result, name="cat-e2e")
    assert mv.pipeline_hash

    pipe = registry.load_pipeline(mv.model_id)
    pred = Predictor(pipe).predict_single({"color": "red", "size": "M", "x": 0.0})
    assert pred.prediction in ("yes", "no", "maybe")

    # Experiment store round-trip
    project = tmp_path / "proj.mlstudio"
    project.write_text("stub", encoding="utf-8")
    store = ExperimentStore(project)
    run = ExperimentRun(
        run_id=result.experiment_id,
        model_name="cat-e2e",
        model_id=result.model_id,
        task=result.task.value,
        dataset_name="cat",
        target="label",
        metrics=dict(result.metrics),
        cv_mean=result.cv_scores.get("mean"),
        cv_std=result.cv_scores.get("std"),
        duration_sec=result.training_duration,
        train_rows=result.train_size,
        test_rows=result.test_size,
        registry_id=mv.model_id,
    )
    store.save_runs([run])
    loaded = store.load_runs()
    assert len(loaded) == 1
    assert loaded[0].model_name == "cat-e2e"


def test_api_train_predict_roundtrip(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from ml_studio.api import Project

    csv = tmp_path / "t.csv"
    pd.DataFrame(
        {
            "a": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1] * 5,
            "b": list(range(50)),
            "y": [0, 1] * 25,
        }
    ).to_csv(csv, index=False)

    p = Project.create("cli_proj")
    p.load_data(str(csv))
    p.set_target("y", task="classification")
    result = p.train(model_id="logistic_regression")
    assert result.estimator is not None
    metrics = p.evaluate()
    assert metrics
    out = p.predict(str(csv), str(tmp_path / "out.csv"))
    assert Path(out).exists()
    scored = pd.read_csv(out)
    assert "prediction" in scored.columns
