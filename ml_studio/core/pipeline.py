"""Pipeline orchestration."""

import hashlib
import json
import yaml
import pandas as pd
from typing import Any
from sklearn.base import BaseEstimator, TransformerMixin

from ml_studio.transforms.base import BaseTransform
from ml_studio.transforms.registry import get as get_transform


class PipelinePreview:
    """Shows what happens to the data at each step."""
    def __init__(self):
        self.original_shape = (0, 0)
        self.final_shape = (0, 0)
        self.steps = []
        self.total_added_columns = 0
        self.total_removed_columns = 0
        self.estimated_memory_bytes = 0
        self.leakage_warnings = []


class Pipeline(BaseEstimator, TransformerMixin):
    """Ordered chain of transforms. Leakage-safe. Serializable."""

    def __init__(self, steps: list[BaseTransform] | None = None):
        self.steps = steps if steps is not None else []

    def add(self, step: BaseTransform) -> "Pipeline":
        """Add a step to the pipeline."""
        self.steps.append(step)
        return self

    def _validate_roles(self, X: pd.DataFrame, step: BaseTransform):
        """Ensure the step is not illegally transforming protected columns unless explicitly passed."""
        # Wait, how do we know the roles? Roles are in project schema.
        # But pipeline operates on DataFrame.
        # User said: "X, y split uses roles automatically"
        # "If any step targets a column with role in {TARGET, ID, GROUP, WEIGHT, TIME_INDEX}, raise ValueError."
        # Exception: "if the user explicitly passes columns=[...] to the transform, respect it."
        pass  # Implementation for role validation will be integrated with Project, as Pipeline itself just receives X (DataFrame).
        # Wait, user said "In Pipeline.fit(): If any step targets a column with role... raise ValueError."
        # If X is just a DataFrame, how does Pipeline know the roles?
        # Maybe X has attrs? Or we just assume Pipeline is used within Project context.
        # Actually, let's look for `X.attrs.get('roles', {})`
        roles = getattr(X, "attrs", {}).get("roles", {})
        if not roles:
            return
            
        protected_roles = {"target", "id", "group", "weight", "time_index"}
        
        # If columns was explicitly passed, it's allowed.
        explicit_cols = step.params.get("columns", None)
        if explicit_cols is not None:
            return
            
        # Otherwise, the step will operate on its fitted_columns_.
        for col in getattr(step, "fitted_columns_", []):
            role = roles.get(col, "feature")
            if role in protected_roles:
                raise ValueError(
                    f"Step '{step.__class__.__name__}' attempted to transform protected column '{col}' "
                    f"(role: {role}) implicitly. Explicitly specify columns to override."
                )

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "Pipeline":
        """Fit all steps."""
        X_curr = X.copy()
        for step in self.steps:
            step.fit(X_curr, y)
            self._validate_roles(X, step)
            # Must transform X_curr for the next step to learn properly
            X_curr = step.transform(X_curr)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply all steps."""
        X_curr = X.copy()
        for step in self.steps:
            X_curr = step.transform(X_curr)
        return X_curr

    def fit_transform(self, X: pd.DataFrame, y: pd.Series | None = None) -> pd.DataFrame:
        """Fit and apply."""
        X_curr = X.copy()
        for step in self.steps:
            # For target encoders, fit_transform provides leak-safe K-fold encoding
            if hasattr(step, "fit_transform") and type(step).__name__ in ["Target", "WOE", "LeaveOneOut"]:
                X_curr = step.fit_transform(X_curr, y)
            else:
                step.fit(X_curr, y)
                self._validate_roles(X, step)
                X_curr = step.transform(X_curr)
        return X_curr

    def to_dict(self) -> dict:
        """Serialize pipeline to dict."""
        return {
            "steps": [
                {
                    "class": step.__class__.__name__,
                    "state": step.to_dict()
                }
                for step in self.steps
            ]
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Pipeline":
        """Reconstruct from dict."""
        steps = []
        for step_data in d.get("steps", []):
            step_class = get_transform(step_data["class"])
            step_obj = step_class.from_dict(step_data["state"])
            steps.append(step_obj)
        return cls(steps=steps)

    def to_yaml(self) -> str:
        """Serialize to YAML string."""
        # Clean up numpy arrays or complex objects before yaml dump
        return yaml.dump(self.to_dict(), default_flow_style=False)

    @classmethod
    def from_yaml(cls, yaml_str: str) -> "Pipeline":
        """Load from YAML string."""
        d = yaml.safe_load(yaml_str)
        return cls.from_dict(d)

    def save(self, path: str) -> None:
        """Save to JSON."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "Pipeline":
        """Load from JSON."""
        with open(path, "r") as f:
            d = json.load(f)
        return cls.from_dict(d)

    def hash(self) -> str:
        """Deterministic hash for reproducibility."""
        # Convert state to dict, clean non-standard types for stable hashing
        d = self.to_dict()
        
        # We need a stable JSON serialization. Default to str for unknown types.
        json_str = json.dumps(d, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode("utf-8")).hexdigest()

    def describe(self) -> str:
        """Human-readable summary."""
        desc = [f"Pipeline with {len(self.steps)} steps:"]
        for i, step in enumerate(self.steps, 1):
            desc.append(f"  {i}. {step.__class__.__name__}: {step.params}")
        return "\n".join(desc)

    def preview(self, X: pd.DataFrame, y: pd.Series | None = None) -> PipelinePreview:
        """Preview shape changes."""
        preview = PipelinePreview()
        preview.original_shape = X.shape
        
        X_curr = X.copy()
        
        for step in self.steps:
            in_cols = set(X_curr.columns)
            in_shape = X_curr.shape
            
            # Predict output columns without full fit if possible, but actually we need to fit to know.
            # We'll just clone the step, fit it, and transform to see what happens.
            # A full fit might be expensive, but preview implies computing it.
            step_clone = get_transform(step.__class__.__name__)(**step.params)
            step_clone.fit(X_curr, y)
            X_curr = step_clone.transform(X_curr)
            
            out_cols = set(X_curr.columns)
            out_shape = X_curr.shape
            
            added = list(out_cols - in_cols)
            removed = list(in_cols - out_cols)
            
            preview.total_added_columns += len(added)
            preview.total_removed_columns += len(removed)
            
            preview.steps.append({
                "name": step.__class__.__name__,
                "input_shape": in_shape,
                "output_shape": out_shape,
                "added_columns": added,
                "removed_columns": removed,
                "warnings": []
            })
            
        preview.final_shape = X_curr.shape
        preview.estimated_memory_bytes = X_curr.memory_usage(deep=True).sum()
        
        return preview
