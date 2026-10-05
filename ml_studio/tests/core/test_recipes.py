import pytest
import yaml
from ml_studio.core.recipes import apply_recipe, get_available_recipes


@pytest.fixture
def recipes_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("ml_studio.core.recipes.RECIPES_DIR", tmp_path)
    return tmp_path


def test_recipes_available(recipes_dir):
    recipe_data = {
        "description": "Test recipe for unit tests",
        "steps": [
            {"transform": "Standard", "params": {"columns": "@numeric"}},
            {"transform": "OneHot", "params": {"columns": "@categorical"}},
        ],
    }
    with open(recipes_dir / "test_recipe.yaml", "w", encoding="utf-8") as f:
        yaml.dump(recipe_data, f)

    recipes = get_available_recipes()
    assert "test_recipe" in recipes


def test_apply_recipe(recipes_dir):
    recipe_data = {
        "steps": [
            {"transform": "Standard", "params": {"columns": "@numeric"}},
            {"transform": "OneHot", "params": {"columns": "@categorical"}},
        ],
    }
    with open(recipes_dir / "test_recipe.yaml", "w", encoding="utf-8") as f:
        yaml.dump(recipe_data, f)

    schema_columns = {
        "num1": {"role": "feature", "kind": "numeric"},
        "num2": {"role": "feature", "kind": "numeric"},
        "cat1": {"role": "feature", "kind": "categorical"},
        "targ": {"role": "target", "kind": "numeric"},
    }

    pipeline = apply_recipe("test_recipe", schema_columns)

    assert len(pipeline.steps) == 2
    assert set(pipeline.steps[0].columns) == {"num1", "num2"}
    assert pipeline.steps[1].columns == ["cat1"]


def test_missing_recipe(recipes_dir):
    with pytest.raises(FileNotFoundError):
        apply_recipe("non_existent_recipe", {})
