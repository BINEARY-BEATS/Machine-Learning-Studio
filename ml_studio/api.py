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
        """Return X and y for the dataset."""
        if not self.dataset or not self.dataset.target_column:
            raise ValueError("Dataset and target column must be set")
        df = self.dataset.dataframe
        y = df[self.dataset.target_column]
        X = df.drop(columns=[self.dataset.target_column])
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
            "environment": env
        }
        
        with open(path / "project.json", "w") as f:
            json.dump(manifest, f, indent=2)
            
        joblib.dump(self.dataset, path / "dataset.joblib")
        
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
        proj.dataset = joblib.load(path / "dataset.joblib")
        return proj

    def set_target(self, target: str, task: str = "classification") -> None:
        """Set the target column and task type."""
        if not self.dataset:
            raise ValueError("Load a dataset first")
        if target not in self.dataset.dataframe.columns:
            raise ValueError(f"Target column '{target}' not found in dataset")
        self.dataset.target_column = target
        self.task = task

    def prepare(self, pipeline: str | None = None) -> None:
        """Run data preparation and feature engineering."""
        if not self.dataset:
            raise ValueError("Load a dataset first")
        # TODO: Implement auto pipeline generation based on profiling
        pass


    def profile(self) -> dict[str, Any]:
        """Run engineering-grade data profiling and return structured JSON."""
        if not self.dataset:
            raise ValueError("Load a dataset first")
        from ml_studio.core.profiling import profile_dataset
        profile = profile_dataset(self.dataset)
        return profile.to_json()

    def sweep(self, models: list[str], tune: str = "optuna", n_trials: int = 50, metric: str = "roc_auc", **kwargs) -> None:
        """Queue and run hyperparameter tuning experiments."""
        if not self.dataset or not self.dataset.target_column:
            raise ValueError("Dataset and target column must be set before sweeping")
        # TODO: Implement batch optuna sweeper
        pass
