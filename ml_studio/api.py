"""Programmatic API for ML Studio."""

from __future__ import annotations
from typing import Any
# pyrefly: ignore [invalid-syntax]
import pandas as pd
import joblib
import json
import sys
import platform
from pathlib import Path
from ml_studio.core.dataset import Dataset
from ml_studio.core.ingestion import load_dataset_from_path, load_dataset_from_url

class Project:
    """Core programmatic entry point for an ML Studio workflow."""

    def __init__(self, name: str):
        self.name = name
        self.dataset: Dataset | None = None
        self.task: str | None = None
        self.schema_columns: dict[str, dict] = {}  # {col: {"role": "feature", "kind": "unknown"}}
        self.pipeline: Any | None = None

    @property
    def target(self) -> str | None:
        """The currently configured target column."""
        return self.dataset.target_column if self.dataset else None

    @classmethod
    def create(cls, name: str) -> Project:
        """Create a new project workspace."""
        return cls(name)

    def load_data(self, source: str, sample: bool = True, **kwargs: Any) -> Dataset:
        """Load data from a local file, URL, or cloud storage URI. 
        If sample=True, automatically truncates files > 1M rows for interactive session speed."""
        self._source_uri = source
        if source.startswith(("s3://", "gcs://", "http://", "https://", "sqlite://", "postgresql://", "mysql://", "snowflake://")):
            self.dataset = load_dataset_from_url(source, sample=sample, **kwargs)
        else:
            self.dataset = load_dataset_from_path(Path(source), sample=sample)
            
        # Initialize schema_columns based on inferred schema
        from ml_studio.core.schema import infer_schema
        schema = infer_schema(self.dataset.dataframe)
        for col in schema.columns:
            self.schema_columns[col.name] = {"role": col.role.value, "kind": col.kind.value}
            
        self.save()
        return self.dataset

    def iter_chunks(self, chunksize: int = 100_000):
        """Iterate over the full dataset in chunks without loading entirely into memory."""
        if not hasattr(self, '_source_uri'):
            raise ValueError("Must load_data first before iterating chunks")
        import pandas as pd
        if self._source_uri.startswith(("s3://", "gcs://", "http://", "https://", "sqlite://", "postgresql://", "mysql://", "snowflake://")):
            import fsspec
            with fsspec.open(self._source_uri, "rb") as f:
                yield from pd.read_csv(f, chunksize=chunksize)
        else:
            yield from pd.read_csv(self._source_uri, chunksize=chunksize)

    def sample_info(self) -> str:
        """Return information about the dataset sampling."""
        if not self.dataset:
            return "No dataset loaded."
        if getattr(self.dataset, 'is_sampled', False):
            return f"Dataset was sampled: {self.dataset.row_count} rows loaded."
        return f"Full dataset loaded: {self.dataset.row_count} rows."

    def get_xy(self) -> tuple[pd.DataFrame, pd.Series]:
        """Return X and y for the dataset based on column roles."""
        if not self.dataset:
            raise ValueError("Dataset must be set")
            
        target_cols = [c for c, info in self.schema_columns.items() if info["role"] == "target"]
        if not target_cols:
            raise ValueError("No target column set.")
        target_col = target_cols[0]
        
        feature_cols = [c for c, info in self.schema_columns.items() if info["role"] == "feature" and c in self.dataset.dataframe.columns]
        
        df = self.dataset.dataframe
        y = df[target_col]
        X = df[feature_cols]
        # Attach roles to X for Pipeline role validation
        X.attrs["roles"] = {c: self.schema_columns.get(c, {}).get("role", "feature") for c in X.columns}
        X.attrs["roles"][target_col] = "target"
        
        return X, y

    def save(self) -> None:
        """Save the project state and environment manifest."""
        import joblib
        import sys
        import platform
        import json
        
        path = Path(self.name)
        path.mkdir(exist_ok=True)
        
        env = {
            "python": sys.version,
            "platform": platform.platform(),
            "libraries": {}
        }
        try:
            import pandas, numpy, sklearn
            env["libraries"] = {
                "pandas": pandas.__version__,
                "numpy": numpy.__version__,
                "scikit-learn": sklearn.__version__
            }
        except ImportError:
            pass
            
        manifest = {
            "name": self.name,
            "task": self.task,
            "target": self.target,
            "schema_columns": self.schema_columns,
            "environment": env
        }
        
        with open(path / "project.json", "w") as f:
            json.dump(manifest, f, indent=2)
            
        joblib.dump(self.dataset, path / "dataset.joblib")
        if self.pipeline:
            self.pipeline.save(str(path / "pipeline.json"))
        
    @classmethod
    def open(cls, name: str) -> Project:
        """Load a project state from disk."""
        import joblib
        import json
        
        path = Path(name)
        if not (path / "project.json").exists():
            raise FileNotFoundError(f"Project '{name}' not found.")
            
        with open(path / "project.json", "r") as f:
            manifest = json.load(f)
            
        proj = cls(manifest["name"])
        proj.task = manifest["task"]
        proj.schema_columns = manifest.get("schema_columns", {})
        proj.dataset = joblib.load(path / "dataset.joblib")
        
        if (path / "pipeline.json").exists():
            from ml_studio.core.pipeline import Pipeline
            proj.pipeline = Pipeline.load(str(path / "pipeline.json"))
            
        return proj

    def set_role(self, column: str, role: str) -> None:
        """Set the role for a specific column."""
        if not self.dataset:
            raise ValueError("Load a dataset first")
        if column not in self.dataset.dataframe.columns:
            raise KeyError(f"Column '{column}' not found in dataset")
            
        from ml_studio.core.schema import ColumnRole
        valid_roles = [r.value for r in ColumnRole]
        if role not in valid_roles:
            raise ValueError(f"Invalid role '{role}'. Must be one of {valid_roles}")
            
        if column not in self.schema_columns:
            self.schema_columns[column] = {"role": "feature", "kind": "unknown"}
            
        self.schema_columns[column]["role"] = role
        if role == "target":
            self.dataset.target_column = column
            self.task = self.task or "classification"
            
        self.save()

    def set_target(self, target: str, task: str = "classification") -> None:
        """Set the target column and task type."""
        self.task = task
        self.set_role(target, "target")

    def apply_pipeline(self, pipeline: Any) -> None:
        """Fit and apply a pipeline to the dataset, saving the result."""
        X, y = self.get_xy()
        transformed_X = pipeline.fit_transform(X, y)
        
        # Merge back with targets/ids that were ignored
        non_features = [c for c, info in self.schema_columns.items() if info["role"] != "feature" and info["role"] != "drop" and c in self.dataset.dataframe.columns]
        
        final_df = pd.concat([transformed_X, self.dataset.dataframe[non_features]], axis=1)
        self.dataset.dataframe = final_df
        self.pipeline = pipeline
        self.save()

    def prepare(self, pipeline: str | None = None) -> None:
        """Auto pipeline from profiling — not implemented yet."""
        if not self.dataset:
            raise ValueError("Load a dataset first")
        raise NotImplementedError(
            "api.Project.prepare() is not implemented. "
            "Build a pipeline in the GUI Prepare page or use recipes via CLI pipeline create --recipe."
        )

    def profile(self) -> dict[str, Any]:
        """Run engineering-grade data profiling and return structured JSON."""
        if not self.dataset:
            raise ValueError("Load a dataset first")
        from ml_studio.core.profiling import profile_dataset
        profile = profile_dataset(self.dataset)
        return profile.to_json()

    def sweep(self, models: list[str], tune: str = "optuna", n_trials: int = 50, metric: str = "roc_auc", **kwargs) -> None:
        """Batch Optuna sweeper — not implemented yet."""
        if not self.dataset or not self.dataset.target_column:
            raise ValueError("Dataset and target column must be set before sweeping")
        raise NotImplementedError(
            "api.Project.sweep() is not implemented. "
            "Use the GUI Train wizard with Optuna/Grid, or call OptunaTuner from Python."
        )
