"""Pipeline Recipe engine."""

import yaml
from pathlib import Path
from ml_studio.core.pipeline import Pipeline
from ml_studio.transforms.registry import get as get_transform


# Define the default recipes directory
RECIPES_DIR = Path(__file__).parent.parent.parent / "recipes"


def get_available_recipes() -> list[str]:
    """Return list of available recipe names (without .yaml)."""
    if not RECIPES_DIR.exists():
        return []
    return [p.stem for p in RECIPES_DIR.glob("*.yaml")]


def apply_recipe(recipe_name: str, schema_columns: dict) -> Pipeline:
    """
    Build a Pipeline from a recipe YAML file, expanding column references
    based on the provided schema column roles and kinds.
    
    schema_columns should be a dict: {col_name: {"role": role, "kind": kind}}
    """
    recipe_path = RECIPES_DIR / f"{recipe_name}.yaml"
    if not recipe_path.exists():
        raise FileNotFoundError(f"Recipe '{recipe_name}' not found at {recipe_path}")
        
    with open(recipe_path, "r") as f:
        recipe_data = yaml.safe_load(f)
        
    steps = recipe_data.get("steps", [])
    pipeline = Pipeline()
    
    for step_config in steps:
        transform_name = step_config.get("transform")
        params = step_config.get("params", {}).copy()
        
        # Expand column references like "@numeric"
        if "columns" in params:
            val = params["columns"]
            if isinstance(val, str) and val.startswith("@"):
                params["columns"] = _expand_column_reference(val, schema_columns)
            elif isinstance(val, list):
                expanded = []
                for v in val:
                    if isinstance(v, str) and v.startswith("@"):
                        expanded.extend(_expand_column_reference(v, schema_columns))
                    else:
                        expanded.append(v)
                params["columns"] = list(set(expanded))
                
        # Handle cases where expansion results in empty columns and the transform needs them
        if "columns" in params and not params["columns"]:
            # If a transform expects columns but we have none matching the reference,
            # we should probably skip it or just pass empty.
            # Passing empty columns might cause the transform to do nothing, which is correct.
            pass
            
        cls = get_transform(transform_name)
        pipeline.add(cls(**params))
        
    return pipeline


def _expand_column_reference(ref: str, schema_columns: dict) -> list[str]:
    """Expand @numeric, @categorical, @feature etc."""
    ref = ref.lower().strip()
    result = []
    
    for col, info in schema_columns.items():
        role = info.get("role", "feature")
        kind = info.get("kind", "unknown")
        
        # Only expand over feature columns (ignore targets, ids) unless explicit
        if ref == "@feature" and role == "feature":
            result.append(col)
        elif ref == "@target" and role == "target":
            result.append(col)
        elif ref == "@numeric" and role == "feature" and kind == "numeric":
            result.append(col)
        elif ref == "@categorical" and role == "feature" and kind == "categorical":
            result.append(col)
        elif ref == "@datetime" and role == "feature" and kind == "datetime":
            result.append(col)
            
    return result
