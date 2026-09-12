# Machine Learning Studio

![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)
![Framework](https://img.shields.io/badge/Framework-PyQt6-green.svg)
![Tests](https://img.shields.io/badge/tests-51%20passed-brightgreen)

A production-grade, privacy-first desktop ML platform for end-to-end tabular machine learning workflows.

**Author:** Saeed Ur Rehman

---

## Features

- **Project system** — `.mlstudio` versioned archives with autosave and recent projects
- **Multi-format data import** — CSV, Excel, Parquet, Feather, Arrow, ORC, SQLite, JSON
- **Dataset profiling** — cached statistics, quality issues, memory optimization
- **Preprocessing pipeline** — missing values, encoding, scaling, outliers, feature engineering, selection, balancing
- **5 task types** — Classification, Regression, Clustering, Anomaly Detection, Time Series
- **Real models** — sklearn, XGBoost, LightGBM with metadata and hyperparameter support
- **Cross-validation** — KFold, StratifiedKFold, GroupKFold, TimeSeriesSplit
- **Optuna tuning** — configurable trials, timeout, pruning
- **AutoML** — controlled leaderboard with runtime limits
- **Evaluation** — task-appropriate metrics and reports
- **Explainability** — permutation importance, SHAP (optional), partial dependence
- **Model registry** — versioned full-pipeline serialization
- **Inference** — single and batch prediction with chunked processing
- **Modern GUI** — 8-page workflow, command palette (Ctrl+K), design tokens, dark/light themes

---

## Architecture

```
ml_studio/
├── app/          # config, theme, logger, DI container
├── core/         # ML logic (no Qt imports)
├── gui/          # presentation only
├── assets/
└── tests/
```

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

---

## Testing

```sh
pytest ml_studio/tests --cov=ml_studio/core
```

Core coverage target: **80%+** (currently ~81%).

---

## Benchmarks

```sh
python ml_studio/tests/benchmarks/benchmark_io.py
```

---

## Workflow

1. Create/open project
2. Import dataset → profile → select target
3. Build preprocessing pipeline
4. Train/compare models → tune → evaluate
5. Explain → save model → predict → export

See [REFACTOR_PLAN.md](REFACTOR_PLAN.md) for migration details.
