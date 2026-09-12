"""Generate a deterministic, known-issues dirty dataset for testing."""

import numpy as np
import pandas as pd
from pathlib import Path

def generate_dirty_dataset(output_path: str = "dirty_dataset.csv", n_rows: int = 100_000, seed: int = 42):
    np.random.seed(seed)
    
    # Base target
    target = np.random.choice([0, 1], size=n_rows)
    
    df = pd.DataFrame({'Target': target})
    
    # Column A: 30% missing values
    col_a = np.random.randn(n_rows)
    missing_indices = np.random.choice(n_rows, size=int(0.3 * n_rows), replace=False)
    col_a[missing_indices] = np.nan
    df['Col_A'] = col_a
    
    # Column B: constant value (all 42)
    df['Col_B'] = 42
    
    # Column C: 99% one value, 1% another (near-constant)
    df['Col_C'] = np.random.choice(['common', 'rare'], p=[0.99, 0.01], size=n_rows)
    
    # Column D: perfectly correlated with target (leakage)
    df['Col_D'] = target * 10 + np.random.randn(n_rows) * 0.01  # extremely high correlation
    
    # Column E: high cardinality categorical (10,000 unique values)
    df['Col_E'] = [f"cat_{i}" for i in np.random.randint(0, 10_000, size=n_rows)]
    
    # Column F: heavy outliers (10 values at 1e6, rest normal)
    col_f = np.random.randn(n_rows)
    outlier_indices = np.random.choice(n_rows, size=10, replace=False)
    col_f[outlier_indices] = 1_000_000
    df['Col_F'] = col_f
    
    # Column G: high skew (skew > 5) -> lognormal distribution
    # np.random.lognormal(mean=0, sigma=2) has very high positive skew
    df['Col_G'] = np.random.lognormal(mean=0, sigma=2, size=n_rows)
    
    # Column H: datetime with gaps
    base_dates = pd.date_range("2020-01-01", periods=n_rows + 5000, freq="min")
    # randomly drop some to create gaps
    dates = np.random.choice(base_dates, size=n_rows, replace=False)
    dates.sort()
    df['Col_H'] = dates
    
    # Column I: numeric column with mixed string values
    col_i = np.random.randn(n_rows).astype(object)
    string_indices = np.random.choice(n_rows, size=100, replace=False)
    col_i[string_indices] = "NOT_A_NUMBER"
    df['Col_I'] = col_i
    
    # Inject 15 duplicate rows
    # Take 15 random rows and append them, then drop 15 other random rows to keep n_rows constant
    dup_sources = df.sample(n=15, random_state=seed)
    df = pd.concat([df, dup_sources])
    df = df.drop(df.sample(n=15, random_state=seed+1).index)
    
    # Shuffle slightly so duplicates aren't all at the end
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Generated dirty dataset at {output_path} with shape {df.shape}")

if __name__ == "__main__":
    generate_dirty_dataset()
