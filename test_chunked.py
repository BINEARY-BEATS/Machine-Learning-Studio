"""Test chunked reading and memory usage."""

import os
import psutil
import time
import pandas as pd
from ml_studio.api import Project

def get_mem_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024**2

def main():
    print(f"Base RSS: {get_mem_mb():.2f} MB")
    
    # 1. Write a 500MB CSV
    csv_path = "large_test.csv"
    if not os.path.exists(csv_path):
        print("Generating large CSV (approx 500MB)...")
        # 5M rows x 10 cols ~ 400MB
        pd.DataFrame({
            f"col_{i}": range(1_500_000) for i in range(10)
        }).to_csv(csv_path, index=False)
        
    file_size_mb = os.path.getsize(csv_path) / 1024**2
    print(f"File size: {file_size_mb:.2f} MB")

    p = Project.create("chunk_test")

    # Test load with sampling
    print("\n--- Load with Sampling ---")
    start = time.time()
    p.load_data(csv_path, sample=True)
    dur = time.time() - start
    print(f"Loaded {p.dataset.row_count} rows in {dur:.2f}s")
    print(f"Peak RSS: {get_mem_mb():.2f} MB")

    # Test chunked iteration
    print("\n--- Chunked Iteration ---")
    start = time.time()
    total_rows = 0
    for i, chunk in enumerate(p.iter_chunks(chunksize=1_000_000)):
        total_rows += len(chunk)
        if i == 0:
            print(f"Chunk 0 RSS: {get_mem_mb():.2f} MB")
    dur = time.time() - start
    print(f"Iterated {total_rows} rows in {dur:.2f}s")
    print(f"Final RSS: {get_mem_mb():.2f} MB")

    # Test load full
    print("\n--- Disable Sampling ---")
    start = time.time()
    p.load_data(csv_path, sample=False)
    dur = time.time() - start
    print(f"Loaded {p.dataset.row_count} rows in {dur:.2f}s")
    print(f"Peak RSS: {get_mem_mb():.2f} MB")

if __name__ == "__main__":
    main()
