"""Performance benchmark scripts."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd


def generate_benchmark_csv(path: Path, rows: int = 1_000_000, cols: int = 50) -> Path:
    rng = np.random.RandomState(42)
    data = {f"col_{i}": rng.randn(rows) for i in range(cols)}
    df = pd.DataFrame(data)
    df.to_csv(path, index=False)
    return path


def benchmark_csv_load(path: Path) -> float:
    from ml_studio.core.ingestion import load_dataset_from_path

    start = time.time()
    load_dataset_from_path(path)
    return time.time() - start


def benchmark_sort(model, col: int = 0) -> float:
    from PyQt6.QtCore import Qt

    start = time.time()
    model.sort(col, Qt.SortOrder.AscendingOrder)
    return time.time() - start


def benchmark_filter(model, text: str = "0.5") -> float:
    start = time.time()
    model.apply_filter(text)
    return time.time() - start


def run_benchmarks(data_dir: Path | None = None) -> dict[str, float]:
    data_dir = data_dir or Path(__file__).parents[2] / "data"
    csv_path = data_dir / "benchmark_1m.csv"
    if not csv_path.exists():
        print("Generating 1M row benchmark CSV (this may take a minute)...")
        generate_benchmark_csv(csv_path)

    results = {}
    print("Benchmarking CSV load...")
    results["csv_load_seconds"] = benchmark_csv_load(csv_path)

    from ml_studio.gui.widgets.data_table import DataFrameTableModel

    df = pd.read_csv(csv_path, nrows=100_000)
    model = DataFrameTableModel(df)
    results["sort_seconds"] = benchmark_sort(model)
    results["filter_seconds"] = benchmark_filter(model)

    print("Results:", results)
    return results


if __name__ == "__main__":
    run_benchmarks()
