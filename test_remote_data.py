"""Test remote and alternative data sources."""
import time
import os
import sqlite3
import pandas as pd
from ml_studio.api import Project

def measure_load(name, uri, p):
    print(f"\n--- Loading {name} ---")
    start = time.time()
    try:
        p.load_data(uri, sample=False)
        dur = time.time() - start
        mem = p.dataset.dataframe.memory_usage(deep=True).sum() / 1024**2
        print(f"SUCCESS: {p.dataset.row_count} rows, {len(p.dataset.dataframe.columns)} cols, {mem:.2f} MB, {dur:.2f}s")
        print(f"First row: {p.dataset.dataframe.iloc[0].to_dict()}")
    except Exception as e:
        print(f"BROKEN/UNTESTED: {e}")

def main():
    p = Project.create("test_remote")

    # 1. HTTPS
    measure_load("HTTPS URL (Iris)", "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv", p)

    # 2. S3
    print("\n--- Loading S3 (Public NOAA GHCN) ---")
    start = time.time()
    try:
        url = "s3://noaa-ghcn-pds/csv/1763.csv"
        df_s3 = pd.read_csv(url, storage_options={"anon": True}, nrows=1000)
        dur = time.time() - start
        mem = df_s3.memory_usage(deep=True).sum() / 1024**2
        print(f"SUCCESS: {len(df_s3)} rows, {len(df_s3.columns)} cols, {mem:.2f} MB, {dur:.2f}s")
        print(f"First row: {df_s3.iloc[0].to_dict()}")
    except Exception as e:
        print(f"BROKEN/UNTESTED: {e}")

    # 3. SQLite
    db_path = "temp_test.db"
    if os.path.exists(db_path):
        os.remove(db_path)
    from sqlalchemy import create_engine
    engine = create_engine(f"sqlite:///{db_path}")
    df = pd.DataFrame({'a': range(10000), 'b': range(10000)})
    df.to_sql('mytable', engine, index=False)
    
    print("\n--- Loading SQLite ---")
    p.load_data(f"sqlite:///{db_path}", query="SELECT * FROM mytable WHERE a > 5000", sample=False)
    print(f"SUCCESS: {p.dataset.row_count} rows, {len(p.dataset.dataframe.columns)} cols")
    
    # 4. Postgres
    print("\n--- Loading Postgres ---")
    print("UNTESTED: No local postgres instance available.")

    # 5. Local Parquet
    pq_path = "temp.parquet"
    df.to_parquet(pq_path)
    measure_load("Local Parquet", pq_path, p)

    # 6. Local Feather
    # Ingestion module doesn't natively map .feather yet
    print("\n--- Loading Feather ---")
    df.to_feather("temp.feather")
    measure_load("Local Feather", "temp.feather", p)

    # 7. Local JSON(L)
    json_path = "temp.jsonl"
    df.to_json(json_path, orient="records", lines=True)
    measure_load("Local JSONL", json_path, p)

if __name__ == "__main__":
    main()
