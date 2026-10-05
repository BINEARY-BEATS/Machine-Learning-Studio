# Machine Learning Studio

![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)
![Framework](https://img.shields.io/badge/Framework-PyQt6-green.svg)

Privacy-first desktop **tabular ML lab** for data scientists: import → prepare → train → evaluate → predict — all local, with leakage-safe training and frozen inference artifacts.

**Author:** Saeed Ur Rehman

---

## What works today

- **Correctness (Phase 0)** — Train/test split **before** prep/encoder fit; `InferencePipeline` stores Prepare pipeline + feature/target LabelEncoders; predictions inverse-decode class labels; categorical round-trip tested
- **Projects** — `.mlstudio` archives store metadata, dataset, schema roles, and prepare pipeline; experiment ledger (`.experiments.json`); autosave when a path exists
- **Import** — CSV, Excel, Parquet, Feather, Arrow, ORC, SQLite, JSON (optional deps for some formats)
- **Profiling + plots** — async stats/quality; Data Plots tab (histograms, missingness, correlation)
- **Prepare** — visual pipeline (impute, encode, scale, outliers, feature eng, selection) + recipes + preview
- **Train** — 5-step wizard (Task → Data → Model → Tune → Run); Classification / Regression / Clustering / Anomaly; Optuna/Grid; **AutoML leaderboard** → pick winner → train
- **Evaluate** — experiment history, metric details, confusion matrix / residual plots
- **Models / Predict** — registry, joblib/ZIP export with pipeline hash, single + batch predict, permutation importance (SHAP optional), **PSI/KS drift**
- **CLI** — `train` / `evaluate` / `predict` / `automl` on folder projects (same Trainer path)
- **GUI** — lab theme (teal-on-graphite), readiness Home, run-state chip, Ctrl+K palette

## Not finished / limited

- Dual persistence remains: GUI `.mlstudio` vs CLI folder `project.json` (CLI train/predict work on folder projects)
- Custom Python transform still hidden
- True time-series forecasters (TS = regression models + time CV)
- Full ONNX UI path (joblib export is primary; ONNX helper exists)

See [REFACTOR_PLAN.md](REFACTOR_PLAN.md) for history.

---

## Architecture

```
ml_studio/
├── app/          # config, theme, logger, DI container
├── core/         # ML logic (no Qt imports)
├── transforms/   # preprocessing registry
├── gui/          # presentation only
├── assets/
└── tests/
```

**Import rule:** `gui` / `app` → `core`; never `core` → `gui`.

**Inference contract:** every registered model is an `InferencePipeline` (prep + encoders + estimator). Predict never re-fits.

---

## Getting Started

### Prerequisites

Python 3.10+ and pip.

### Installation

```sh
git clone https://github.com/BINEARY-BEATS/Machine-Learning-Studio.git
cd Machine-Learning-Studio
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements-ml-extras.txt
pip install -r requirements-dev.txt   # optional, for tests
```

### Run

```sh
python main.py
```

### CLI (folder projects with `project.json`)

```sh
python -m ml_studio.cli project create myproj
python -m ml_studio.cli data load data.csv --project myproj
python -m ml_studio.cli project set-target myproj y
python -m ml_studio.cli train myproj --model logistic_regression
python -m ml_studio.cli evaluate myproj
python -m ml_studio.cli predict myproj batch.csv --out scored.csv
python -m ml_studio.cli automl myproj
```

Note: the GUI uses `.mlstudio` zip projects; the CLI uses directory-based `api.Project`. Both use the same `Trainer` / `InferencePipeline` correctness path.

---

## Testing

```sh
pytest ml_studio/tests --cov=ml_studio/core
```

Correctness suite: `ml_studio/tests/core/test_correctness_phase0.py`, `test_e2e_categorical.py`.

---

## Workflow

1. Create/open a project  
2. Import dataset → wait for async profile → set target role on Prepare  
3. Build preprocessing pipeline (optional)  
4. Train or AutoML → Evaluate (plots + durable runs)  
5. Predict (single / batch / drift) → export from Models  

For upgrade history, see [REFACTOR_PLAN.md](REFACTOR_PLAN.md).
