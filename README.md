# Machine Learning Studio

![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)
![Framework](https://img.shields.io/badge/Framework-PyQt6-green.svg)
![Tests](https://img.shields.io/badge/tests-287%20collected-brightgreen)

Privacy-first desktop app for **tabular** machine learning: import → prepare → train → evaluate → predict — all local.

**Author:** Saeed Ur Rehman

---

## What works today

- **Projects** — `.mlstudio` archives store metadata, dataset, schema roles, and prepare pipeline; autosave when a path exists
- **Import** — CSV, Excel, Parquet, Feather, Arrow, ORC, SQLite, JSON (optional deps for some formats)
- **Profiling** — async stats + quality issues; memory dtype optimization
- **Prepare** — visual pipeline (impute, encode, scale, outliers, feature eng, selection) + recipes + preview
- **Train** — Classification, Regression, Clustering, Anomaly Detection; Optuna/Grid tuning when configured; Time Series uses regression models + time-aware CV
- **Evaluate** — experiment history and metric details
- **Models / Predict** — registry, single + chunked batch predict, permutation importance (SHAP optional)
- **GUI** — 8-page shell, Ctrl+K command palette, light/dark themes

## Not finished / limited

- Full AutoML leaderboard UI (core `AutoMLRunner` exists; palette opens Train + Optuna)
- Drift monitoring tab (labeled Coming soon)
- Custom Python transform (hidden until implemented)
- CLI `api.Project.prepare` / `sweep` raise clearly as unimplemented
- Balancing transforms and some optional ML extras need extra packages

See [REFACTOR_PLAN.md](REFACTOR_PLAN.md) for the upgrade checklist.

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

### CLI (workspace folders with `project.json`)

```sh
python -m ml_studio.cli project create myproj
python -m ml_studio.cli data load data.csv --project myproj
python -m ml_studio.cli data head myproj -n 5
python -m ml_studio.cli data info myproj
```

Note: the GUI uses `.mlstudio` zip projects; the CLI uses a separate directory-based `api.Project`.

---

## Testing

```sh
pytest ml_studio/tests --cov=ml_studio/core
```

Core coverage target: **80%+**.

---

## Benchmarks

```sh
python ml_studio/tests/benchmarks/benchmark_io.py
```

---

## Workflow

1. Create/open a project  
2. Import dataset → wait for async profile → set target role on Prepare  
3. Build preprocessing pipeline (optional)  
4. Train (validate wizard steps; optional Optuna/Grid) → Evaluate  
5. Predict (single / batch) → export from Models  

For upgrade history and remaining polish, see [REFACTOR_PLAN.md](REFACTOR_PLAN.md).
