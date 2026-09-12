import pytest
import os
import yaml
from ml_studio.core.recipes import apply_recipe, get_available_recipes, RECIPES_DIR
from ml_studio.core.pipeline import Pipeline
from ml_studio.core.schema import ColumnRole

def test_recipes_available():
    # Setup a dummy recipe
    os.makedirs(RECIPES_DIR, exist_ok=True)
    recipe_data = {
        "steps": [
            {"transform": "Standard", "params": {"columns": "@numeric"}},
            {"transform": "OneHot", "params": {"columns": "@categorical"}}
        ]
    }
    with open(RECIPES_DIR / "test_recipe.yaml", "w") as f:
        yaml.dump(recipe_data, f)
        
    recipes = get_available_recipes()
    assert "test_recipe" in recipes

def test_apply_recipe():
    schema_columns = {
        "num1": {"role": "feature", "kind": "numeric"},
        "num2": {"role": "feature", "kind": "numeric"},
        "cat1": {"role": "feature", "kind": "categorical"},
        "targ": {"role": "target", "kind": "numeric"}
    }
    
    pipeline = apply_recipe("test_recipe", schema_columns)
    
    assert len(pipeline.steps) == 2
    assert pipeline.steps[0].columns == ["num1", "num2"]  # or order-independent
    assert set(pipeline.steps[0].columns) == {"num1", "num2"}
    assert pipeline.steps[1].columns == ["cat1"]
    
def test_missing_recipe():
    with pytest.raises(FileNotFoundError):
        apply_recipe("non_existent_recipe", {})
