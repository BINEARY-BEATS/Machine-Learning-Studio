"""Application configuration and startup dependency validation."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from importlib import import_module
from typing import Sequence


@dataclass(frozen=True)
class DependencySpec:
    module: str
    pip_name: str
    required: bool = True
    description: str = ""


CORE_DEPENDENCIES: tuple[DependencySpec, ...] = (
    DependencySpec("PyQt6", "PyQt6", True, "Desktop GUI framework"),
    DependencySpec("pandas", "pandas", True, "Tabular data manipulation"),
    DependencySpec("numpy", "numpy", True, "Numerical computing"),
    DependencySpec("sklearn", "scikit-learn", True, "Machine learning"),
    DependencySpec("joblib", "joblib", True, "Model serialization"),
    DependencySpec("scipy", "scipy", True, "Scientific computing"),
)

OPTIONAL_ML_DEPENDENCIES: tuple[DependencySpec, ...] = (
    DependencySpec("xgboost", "xgboost", False, "XGBoost models"),
    DependencySpec("lightgbm", "lightgbm", False, "LightGBM models"),
    DependencySpec("optuna", "optuna", False, "Hyperparameter optimization"),
    DependencySpec("shap", "shap", False, "SHAP explainability"),
    DependencySpec("pyqtgraph", "pyqtgraph", False, "Interactive charts"),
    DependencySpec("fsspec", "fsspec", False, "Remote file system access"),
    DependencySpec("openpyxl", "openpyxl", False, "Excel file support"),
    DependencySpec("pyarrow", "pyarrow", False, "Parquet/Arrow/Feather support"),
    DependencySpec("chardet", "chardet", False, "CSV encoding detection"),
    DependencySpec("imbalanced_learn", "imbalanced-learn", False, "SMOTE/ADASYN"),
    DependencySpec("skl2onnx", "skl2onnx", False, "ONNX export"),
    DependencySpec("onnx", "onnx", False, "ONNX runtime"),
)


@dataclass
class AppConfig:
    app_name: str = "Machine Learning Studio"
    app_version: str = "2.0.0"
    organization: str = "MLStudio"
    log_level: str = "INFO"
    autosave_interval_ms: int = 120_000
    pandas_copy_on_write: bool = True
    max_worker_threads: int = 2
    debounce_ms: int = 50
    project_extension: str = ".mlstudio"
    project_format_version: int = 2
    available_optional: set[str] = field(default_factory=set)


def check_dependencies(
    specs: Sequence[DependencySpec],
    *,
    required_only: bool = True,
) -> tuple[list[str], list[str]]:
    """Return (missing_required, available_optional_modules)."""
    missing: list[str] = []
    available: list[str] = []
    for spec in specs:
        try:
            import_module(spec.module)
            available.append(spec.module)
        except ImportError:
            if spec.required:
                missing.append(f"{spec.pip_name} — {spec.description}")
    if not required_only:
        for spec in OPTIONAL_ML_DEPENDENCIES:
            try:
                import_module(spec.module)
                available.append(spec.module)
            except ImportError:
                pass
    return missing, available


def validate_startup_dependencies() -> AppConfig:
    """Validate required dependencies; exit with clear message if missing."""
    missing, _ = check_dependencies(CORE_DEPENDENCIES)
    if missing:
        lines = [
            "Machine Learning Studio cannot start — missing required dependencies:",
            "",
        ]
        for item in missing:
            pip_name = item.split(" — ")[0]
            lines.append(f"  • {item}")
            lines.append(f"    pip install {pip_name}")
        lines.append("")
        lines.append("Install core dependencies:")
        lines.append("  pip install -r requirements-core.txt")
        message = "\n".join(lines)
        print(message, file=sys.stderr)
        sys.exit(1)

    config = AppConfig()
    _, optional = check_dependencies(OPTIONAL_ML_DEPENDENCIES, required_only=False)
    config.available_optional = set(optional)
    return config


def enable_pandas_copy_on_write() -> None:
    """Enable pandas 2.x copy-on-write mode when supported."""
    import pandas as pd

    try:
        pd.options.mode.copy_on_write = True  # type: ignore[attr-defined]
    except Exception:
        pass
