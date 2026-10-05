"""Recipe description reads top-level and metadata keys."""

from pathlib import Path

import yaml


def test_recipe_description_prefers_top_level(tmp_path: Path):
    path = tmp_path / "demo.yaml"
    path.write_text(
        yaml.dump({"description": "Top level", "metadata": {"description": "Nested"}}),
        encoding="utf-8",
    )
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    desc = data.get("description") or (data.get("metadata") or {}).get("description")
    assert desc == "Top level"


def test_recipe_description_falls_back_to_metadata(tmp_path: Path):
    path = tmp_path / "demo.yaml"
    path.write_text(yaml.dump({"metadata": {"description": "Nested only"}}), encoding="utf-8")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    desc = data.get("description") or (data.get("metadata") or {}).get("description")
    assert desc == "Nested only"
