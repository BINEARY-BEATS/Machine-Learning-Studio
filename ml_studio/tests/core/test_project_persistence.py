"""Project save/open session round-trip tests."""

from __future__ import annotations

import pandas as pd
import pytest

from ml_studio.app.config import AppConfig
from ml_studio.core.dataset import Dataset
from ml_studio.core.pipeline import Pipeline
from ml_studio.core.project_manager import ProjectManager
from ml_studio.transforms.registry import get as get_transform


@pytest.fixture
def pm(tmp_path, monkeypatch):
    config = AppConfig()
    manager = ProjectManager(config)
    # Keep recent list out of the real home dir during tests
    manager._recent_path = tmp_path / "recent_projects.json"
    return manager


def test_save_open_restores_dataset_and_pipeline(pm, tmp_path):
    project = pm.new_project("Roundtrip")
    df = pd.DataFrame({"a": [1.0, 2.0, None], "b": ["x", "y", "x"], "y": [0, 1, 0]})
    project.dataset = Dataset(_dataframe=df, name="demo", target_column="y")
    Impute = get_transform("Impute")
    pipe = Pipeline()
    step = Impute(strategy="mean", columns=["a"])
    step.enabled = False
    pipe.add(step)
    project.pipeline = pipe
    project.schema_overrides = {"y": "target", "a": "feature"}

    path = tmp_path / "demo.mlstudio"
    pm.save_project(path)
    assert path.exists()
    assert project.dirty is False

    pm.close_project()
    opened = pm.open_project(path)

    assert opened.metadata.name == "Roundtrip"
    assert opened.dataset is not None
    assert list(opened.dataset.dataframe.columns) == ["a", "b", "y"]
    assert len(opened.dataset.dataframe) == 3
    assert opened.dataset.target_column == "y"
    assert opened.schema_overrides.get("y") == "target"
    assert len(opened.pipeline.steps) == 1
    assert opened.pipeline.steps[0].__class__.__name__ == "Impute"
    assert opened.pipeline.steps[0].enabled is False


def test_open_legacy_empty_project_still_works(pm, tmp_path):
    """v1 archives with only manifest should open without crashing."""
    import json
    import zipfile

    path = tmp_path / "legacy.mlstudio"
    staging = tmp_path / "legacy_staging"
    staging.mkdir()
    (staging / "datasets").mkdir()
    manifest = {
        "id": "abc",
        "name": "Legacy",
        "format_version": 1,
        "created_at": "2024-01-01T00:00:00+00:00",
        "modified_at": "2024-01-01T00:00:00+00:00",
        "settings": {},
    }
    with (staging / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f)
    with zipfile.ZipFile(path, "w") as zf:
        zf.write(staging / "manifest.json", "manifest.json")

    opened = pm.open_project(path)
    assert opened.metadata.name == "Legacy"
    assert opened.dataset is None
