import pandas as pd
from ml_studio.core.pipeline import Pipeline
from ml_studio.transforms.missing import Impute
from ml_studio.transforms.encoding import Target
from ml_studio.transforms.scaling import Robust
from ml_studio.transforms.selection import MISelect

# Create dummy data
data = {
    "age": [25, 30, None, 45, 50, 22, 35, 40, None, 60, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
    "income": [50000, 60000, 75000, None, 120000, 45000, 80000, 90000, 95000, 150000, 50000, 60000, 75000, 80000, 120000, 45000, 80000, 90000, 95000, 150000],
    "category": ["A", "B", "A", "C", "B", "A", "C", "B", "C", "A", "A", "B", "A", "C", "B", "A", "C", "B", "C", "A"],
    "target": [0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1]
}

df_train = pd.DataFrame(data)
df_test = pd.DataFrame(data)

X_train = df_train[["age", "income", "category"]]
y_train = df_train["target"]

X_test = df_test[["age", "income", "category"]]

# Build
pipe = (Pipeline()
    .add(Impute(strategy="median", columns=["age", "income"]))
    .add(Target(columns=["category"], cv=5))
    .add(Robust(columns=["age", "income"]))
)

# Fit on train
print("Fitting pipeline...")
X_train_t = pipe.fit_transform(X_train, y_train)
print(f"X_train_t shape: {X_train_t.shape}")
print(f"X_train_t columns: {list(X_train_t.columns)}")

# Save
print("Saving pipeline to test_pipeline.json...")
pipe.save("test_pipeline.json")

# Load
from ml_studio.core.pipeline import Pipeline as P
print("Loading pipeline from test_pipeline.json...")
pipe2 = P.load("test_pipeline.json")

# Transform new data
print("Transforming test data...")
X_test_t = pipe2.transform(X_test)

# Verify
assert list(X_test_t.columns) == list(X_train_t.columns), "Column order mismatch after save/load"
print("Column order: OK")
print("Shape:", X_test_t.shape)

# Verify hash determinism
h1 = pipe.hash()
h2 = pipe2.hash()
assert h1 == h2, f"Hash mismatch: {h1} != {h2}"
print("Hash stable:", h1)
