"""Dataset profiling with cached statistics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ml_studio.core.dataset import Dataset
from ml_studio.core.schema import infer_schema


@dataclass
class ColumnProfile:
    name: str
    dtype: str
    count: int
    missing: int
    missing_pct: float
    unique: int
    cardinality: float
    min: Any = None
    max: Any = None
    mean: float | None = None
    median: float | None = None
    std: float | None = None
    q25: float | None = None
    q75: float | None = None
    skew: float | None = None
    kurtosis: float | None = None
    entropy: float | None = None
    is_monotonic: bool | None = None


@dataclass
class DatasetProfile:
    row_count: int
    column_count: int
    memory_bytes: int
    duplicate_rows: int
    constant_columns: list[str] = field(default_factory=list)
    columns: list[ColumnProfile] = field(default_factory=list)
    correlation: pd.DataFrame | None = None
    leakage_warnings: list[dict[str, Any]] = field(default_factory=list)
    quality_score: float = 100.0
    issues: list[dict[str, Any]] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        """Output structured JSON for programmatic use."""
        import json
        from dataclasses import asdict
        
        d = asdict(self)
        
        # Remove null values recursively (Nit 3)
        def drop_nulls(obj):
            if isinstance(obj, dict):
                return {k: drop_nulls(v) for k, v in obj.items() if v is not None}
            elif isinstance(obj, list):
                return [drop_nulls(v) for v in obj]
            return obj
            
        d = drop_nulls(d)
        
        if self.correlation is not None:
            d["correlation"] = self.correlation.to_dict()
        return d


class ProfileCache:
    def __init__(self) -> None:
        self._cache: dict[str, Any] = {}

    def get(self, dataset: Dataset, column: str, statistic: str) -> Any | None:
        key = dataset.cache_key(column, statistic)
        return self._cache.get(key)

    def set(self, dataset: Dataset, column: str, statistic: str, value: Any) -> None:
        key = dataset.cache_key(column, statistic)
        self._cache[key] = value

    def invalidate(self, dataset_id: str) -> None:
        self._cache = {k: v for k, v in self._cache.items() if not k.startswith(dataset_id)}


_profile_cache = ProfileCache()


def profile_dataset(dataset: Dataset, *, use_cache: bool = True) -> DatasetProfile:
    df = dataset.dataframe
    n_rows = len(df)
    dupes = int(df.duplicated().sum()) if n_rows else 0
    constant = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]

    col_profiles: list[ColumnProfile] = []
    for col in df.columns:
        series = df[col]
        missing = int(series.isnull().sum())
        cp = ColumnProfile(
            name=str(col),
            dtype=str(series.dtype),
            count=n_rows,
            missing=missing,
            missing_pct=missing / n_rows * 100 if n_rows else 0,
            unique=int(series.nunique(dropna=True)),
            cardinality=series.nunique(dropna=True) / n_rows if n_rows else 0,
        )
        if pd.api.types.is_numeric_dtype(series):
            clean = series.dropna()
            if len(clean):
                cp.min = float(clean.min())
                cp.max = float(clean.max())
                cp.mean = float(clean.mean())
                cp.median = float(clean.median())
                cp.std = float(clean.std())
                cp.q25 = float(clean.quantile(0.25))
                cp.q75 = float(clean.quantile(0.75))
                cp.skew = float(clean.skew()) if len(clean) > 2 else None
                cp.kurtosis = float(clean.kurtosis()) if len(clean) > 3 else None
                cp.is_monotonic = bool(clean.is_monotonic_increasing or clean.is_monotonic_decreasing)
        
        # Calculate entropy (using nats or bits; we use bits with log2)
        # Binning strategy:
        # - If categorical or discrete numeric (< 50 unique values), use exact value counts.
        # - If continuous numeric, use numpy's 'fd' (Freedman-Diaconis) estimator which is robust to outliers,
        #   or fallback to 'sturges' if FD fails.
        clean_series = series.dropna()
        if len(clean_series) == 0:
            cp.entropy = None
        elif cp.unique <= 1:
            cp.entropy = 0.0
        else:
            if pd.api.types.is_numeric_dtype(series) and cp.unique > 50:
                counts, _ = np.histogram(clean_series, bins="fd")
                if len(counts) > 0 and counts.sum() > 0:
                    probs = counts / counts.sum()
                    probs = probs[probs > 0]
                    cp.entropy = float(-np.sum(probs * np.log2(probs)))
                else:
                    cp.entropy = 0.0
            else:
                counts = clean_series.value_counts(normalize=True).values
                counts = counts[counts > 0]
                if len(counts) > 0:
                    cp.entropy = float(-np.sum(counts * np.log2(counts)))
                else:
                    cp.entropy = 0.0
                
        col_profiles.append(cp)

    corr = None
    numeric = df.select_dtypes(include=np.number)
    if numeric.shape[1] >= 2:
        corr = numeric.corr()

    profile = DatasetProfile(
        row_count=n_rows,
        column_count=len(df.columns),
        memory_bytes=dataset.memory_bytes,
        duplicate_rows=dupes,
        constant_columns=constant,
        columns=col_profiles,
        correlation=corr,
    )
    
    # Advanced Leakage Probes (if target is set)
    if dataset.target_column and dataset.target_column in df.columns:
        target_series = df[dataset.target_column]
        for cp in profile.columns:
            if cp.name == dataset.target_column:
                continue
            # Single value dominance
            if cp.unique == 1:
                profile.leakage_warnings.append({"column": cp.name, "type": "single_value", "warning": "Column has only 1 unique value"})
            # Correlation with target
            elif corr is not None and cp.name in corr.columns and dataset.target_column in corr.columns:
                c_val = corr.loc[cp.name, dataset.target_column]
                if abs(c_val) > 0.95:
                    profile.leakage_warnings.append({"column": cp.name, "type": "high_correlation", "warning": f"Highly correlated with target ({c_val:.2f})"})
                    
    # Set on dataset first so detect_quality_issues can read it
    dataset.statistics["profile"] = profile
    
    # Quality score logic based on issues
    quality_score = 100.0
    issues = detect_quality_issues(dataset)
    
    frac_missing = sum(1 for cp in profile.columns if cp.missing_pct > 0) / max(1, len(profile.columns))
    quality_score -= 25.0 * frac_missing
    
    if profile.row_count > 0:
        frac_dupes = profile.duplicate_rows / profile.row_count
        quality_score -= 20.0 * frac_dupes
        
    if any(i["type"] == "leakage" for i in issues):
        quality_score -= 30.0
        
    frac_high_card = sum(1 for i in issues if i["type"] == "high_cardinality") / max(1, len(profile.columns))
    quality_score -= 10.0 * frac_high_card
    
    frac_constant = sum(1 for i in issues if i["type"] in ["constant", "near-constant"]) / max(1, len(profile.columns))
    quality_score -= 5.0 * frac_constant
    
    frac_skew = sum(1 for i in issues if i["type"] == "high_skew") / max(1, len(profile.columns))
    quality_score -= 5.0 * frac_skew
    
    frac_outliers = sum(1 for i in issues if i["type"] == "heavy_outliers") / max(1, len(profile.columns))
    quality_score -= 5.0 * frac_outliers
    
    profile.quality_score = max(0.0, quality_score)
    
    # Attach issues to profile
    profile.issues = issues
    
    return profile


def optimize_dtypes(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Optimize memory; never silently lossy-convert."""
    before = int(df.memory_usage(deep=True).sum())
    optimized = df.copy()
    changes: list[str] = []

    for col in optimized.columns:
        series = optimized[col]
        if series.dtype == object or pd.api.types.is_string_dtype(series):
            nunique = series.nunique(dropna=True)
            if nunique / max(len(series), 1) < 0.05:
                optimized[col] = series.astype("category")
                changes.append(f"{col}: str → category")
        elif pd.api.types.is_integer_dtype(series):
            optimized[col] = pd.to_numeric(series, downcast="integer")
            changes.append(f"{col}: int downcast")
        elif pd.api.types.is_float_dtype(series):
            optimized[col] = pd.to_numeric(series, downcast="float")
            changes.append(f"{col}: float downcast")

    after = int(optimized.memory_usage(deep=True).sum())
    report = {
        "memory_before": before,
        "memory_after": after,
        "reduction_pct": (1 - after / before) * 100 if before else 0,
        "changes": changes,
    }
    return optimized, report


def detect_quality_issues(dataset: Dataset) -> list[dict[str, Any]]:
    """Analyze dataset for ML quality issues."""
    issues: list[dict[str, Any]] = []
    
    profile = dataset.statistics.get("profile")
    if not profile:
        return issues
    
    if profile.duplicate_rows:
        issues.append({
            "type": "duplicates", "severity": "warning", "count": profile.duplicate_rows,
            "description": f"Found {profile.duplicate_rows} duplicate rows.",
            "suggestion": "Drop duplicates unless the repeated samples carry extra weight."
        })
        
    for lw in profile.leakage_warnings:
        issues.append({
            "type": "leakage", "severity": "critical", "column": lw["column"],
            "description": lw["warning"],
            "suggestion": "Drop this column as it contains label leakage."
        })
        
    for col in profile.columns:
        if col.missing_pct > 50:
            issues.append({
                "type": "missing", "severity": "error", "column": col.name, "pct": col.missing_pct,
                "description": f"Extremely high missing values ({col.missing_pct:.1f}%).",
                "suggestion": "Drop this column as imputation will be unreliable."
            })
        elif col.missing_pct > 0:
            issues.append({
                "type": "missing", "severity": "warning", "column": col.name, "pct": col.missing_pct,
                "description": f"Missing values detected ({col.missing_pct:.1f}%).",
                "suggestion": "Use an imputer (mean/median for numeric, mode for categorical)."
            })
            
        if col.unique == 1:
            issues.append({
                "type": "constant", "severity": "info", "column": col.name,
                "description": "Column has only 1 unique value.",
                "suggestion": "Drop the column; it provides no predictive power."
            })
        elif col.unique > 1 and col.cardinality < 0.05 and col.entropy is not None and col.entropy < 0.1:
            issues.append({
                "type": "near-constant", "severity": "warning", "column": col.name,
                "description": "Near-constant values (very low entropy).",
                "suggestion": "Consider dropping if variance is effectively zero."
            })
            
        if col.dtype in ["object", "str"] and col.unique > 100 and col.cardinality > 0.05:
            issues.append({
                "type": "high_cardinality", "severity": "warning", "column": col.name,
                "description": f"Categorical with high cardinality ({col.unique} unique values).",
                "suggestion": "Use Target Encoding or Hashing instead of One-Hot Encoding."
            })
            
        if col.skew is not None and abs(col.skew) > 3.0:
            issues.append({
                "type": "high_skew", "severity": "warning", "column": col.name,
                "description": f"Highly skewed distribution (skew={col.skew:.2f}).",
                "suggestion": "Apply a Power Transform (e.g., Yeo-Johnson) to normalize."
            })
            
        if col.kurtosis is not None and col.kurtosis > 10.0:
            issues.append({
                "type": "heavy_outliers", "severity": "warning", "column": col.name,
                "description": f"Extremely heavy tails (kurtosis={col.kurtosis:.2f}). Indicates outliers.",
                "suggestion": "Winsorize the data or use robust scaling."
            })
            
    return issues
