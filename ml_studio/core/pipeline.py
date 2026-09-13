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

    def __str__(self):
        lines = [f"Pipeline Preview: {self.original_shape} -> {self.final_shape}"]
        for i, step in enumerate(self.steps):
            lines.append(f"Step {i+1}: {step['name']}")
            lines.append(f"  Shape: {step['input_shape']} -> {step['output_shape']}")
            if step['added_columns']:
                lines.append(f"  + {len(step['added_columns'])} columns")
            if step['removed_columns']:
                lines.append(f"  - {len(step['removed_columns'])} columns")
        return "\n".join(lines)


class Pipeline(BaseEstimator, TransformerMixin):
    """Ordered chain of transforms. Leakage-safe. Serializable."""

    def __init__(self, steps: list[BaseTransform] | None = None):
        self.steps = steps if steps is not None else []

    def add(self, step: BaseTransform) -> "Pipeline":
        """Add a step to the pipeline."""
        self.steps.append(step)
        return self

    def _validate_roles(self, X: pd.DataFrame) -> None:
        roles = getattr(X, "attrs", {}).get("roles", {})
        for step in self.steps:
            for col in getattr(step, "columns", []) or []:
                role = roles.get(col)
                if role in ("target", "id", "group", "weight", "time_index") and role != "feature":
                    raise ValueError(
                        f"Transform '{step.__class__.__name__}' "
                        f"attempts to modify column '{col}' which has "
                        f"role '{role}'. Only 'feature' columns may be "
                        f"transformed. Pass columns=[...] explicitly "
                        f"to override."
                    )

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "Pipeline":
        """Fit all steps."""
        self._validate_roles(X)
        X_curr = X.copy()
        for i, step in enumerate(self.steps):
            step.fit(X_curr, y)
            if i < len(self.steps) - 1:
                X_curr = step.transform(X_curr)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply all steps."""
        for i, step in enumerate(self.steps):
            X = step.transform(X)
        return X

    @property
    def predict(self):
        """Apply transforms to the data, and predict with the final estimator."""
        if not self.steps or not hasattr(self.steps[-1], "predict"):
            raise AttributeError(f"{self.__class__.__name__} has no predict method")
        return self._predict

    def _predict(self, X: pd.DataFrame, **predict_params):
        Xt = X
        for step in self.steps[:-1]:
            Xt = step.transform(Xt)
        return self.steps[-1].predict(Xt, **predict_params)

    @property
    def predict_proba(self):
        """Apply transforms, and predict_proba of the final estimator."""
        if not self.steps or not hasattr(self.steps[-1], "predict_proba"):
            raise AttributeError(f"{self.__class__.__name__} has no predict_proba method")
        return self._predict_proba

    def _predict_proba(self, X: pd.DataFrame, **predict_proba_params):
        Xt = X
        for step in self.steps[:-1]:
            Xt = step.transform(Xt)
        return self.steps[-1].predict_proba(Xt, **predict_proba_params)

    @property
    def score(self):
        """Apply transforms, and score with the final estimator."""
        if not self.steps or not hasattr(self.steps[-1], "score"):
            raise AttributeError(f"{self.__class__.__name__} has no score method")
        return self._score

    def _score(self, X: pd.DataFrame, y=None, sample_weight=None):
        Xt = X
        for step in self.steps[:-1]:
            Xt = step.transform(Xt)
        score_params = {}
        if sample_weight is not None:
            score_params["sample_weight"] = sample_weight
        return self.steps[-1].score(Xt, y, **score_params)

    def fit_transform(self, X: pd.DataFrame, y: pd.Series | None = None) -> pd.DataFrame:
        """Fit and apply."""
        self._validate_roles(X)
        X_curr = X.copy()
        for i, step in enumerate(self.steps):
            # For target encoders, fit_transform provides leak-safe K-fold encoding
            if hasattr(step, "fit_transform") and type(step).__name__ in ["Target", "WOE", "LeaveOneOut"]:
                X_curr = step.fit_transform(X_curr, y)
            else:
                step.fit(X_curr, y)
                if i < len(self.steps) - 1 or hasattr(step, "transform"):
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
