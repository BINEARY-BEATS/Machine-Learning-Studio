# Machine Learning Studio — Refactor Plan

> Living document for the production-grade rewrite. Updated at the end of each migration phase.

**Last updated:** Phase 16 — Final Cleanup (complete)

---

## Phase Status

| Phase | Scope | Status |
|-------|-------|--------|
| 1 | Architecture foundation | Complete |
| 2 | Project system (.mlstudio) | Complete |
| 3 | Dataset abstraction + ingestion | Complete |
| 4 | Schema + profiling | Complete |
| 5 | Virtualized data viewer | Complete |
| 6 | Pipeline / transforms | Complete |
| 7 | Training engine (5 task types) | Complete |
| 8 | Model registry | Complete |
| 9 | Evaluation | Complete |
| 10 | Explainability | Complete |
| 11 | Inference | Complete |
| 12 | AutoML + Optuna | Complete |
| 13 | GUI redesign | Complete |
| 14 | Performance optimization | Complete |
| 15 | Testing (81% core coverage) | Complete |
| 16 | Final cleanup | Complete |

Legacy `gui/` and `utils/` modules removed. Single `ml_studio/` architecture remains.

The legacy application is a flat PyQt5 FYP desktop app with no package structure:

```
main.py
├── gui/main_window.py      # Shell: sidebar + QStackedWidget, shared DataFrame
├── gui/dashboard.py        # Static welcome screen
├── gui/data_viewer.py      # PandasTableModel + summary tab
├── gui/data_cleaner.py     # Cleaning UI (duplicates utils logic)
├── gui/model_trainer.py    # 1756-line god widget (ML + UI + persistence)
├── gui/visualizer.py       # Matplotlib plots
└── utils/
    ├── data_loader.py      # QThread CSV/Excel/JSON loader
    ├── data_cleaner.py     # Preprocessing helpers (unused by GUI cleaner)
    ├── model_utils.py      # Regression model factories
    └── visual_utils.py     # Seaborn/Matplotlib plot factory
```

**Problems:** No `core/` isolation, no project system, GUI implements ML logic, import fallbacks with DummyModel placeholders.

---

## 2. Current Data Flow

1. User imports CSV/Excel/JSON via `DataLoaderThread` (chunked CSV only).
2. `MainWindow` stores `current_data_frame` and emits `data_updated_signal`.
3. Viewer/Cleaner/Trainer/Visualizer each receive the full DataFrame.
4. Cleaning mutates copies in `gui/data_cleaner.py` — does **not** call `utils/data_cleaner.py`.
5. Training: `_prepare_training_data()` on UI thread → `TrainingThread` builds `Pipeline(StandardScaler, model)`.
6. Models saved via joblib from GUI — pipeline lacks full preprocessing consistency.

---

## 3. Current ML Capabilities

| Category | Status |
|----------|--------|
| Regression models (Linear, Ridge, Lasso, RF, XGB, LGBM, SVR, KNN) | Factories exist in `utils/model_utils.py` |
| GUI reachability | Broken bulk import → many models become DummyModel placeholders |
| Classification | Not implemented |
| Clustering | Not implemented |
| Anomaly detection | Not implemented |
| Time series | Not implemented |
| Optuna / SHAP / AutoML | Not present |
| Model registry / projects | Not present |

---

## 4. Current UI Structure

- Sidebar: Dashboard, Import, View, Clean, Train, Visualize
- No Prepare/Evaluate/Predict/Models/Settings pages
- No command palette, toasts, or guided workflow
- `assets/styles.qss` unused; ~180 hardcoded hex colors in Python
- `PandasTableModel` exists but is not truly virtualized for millions of rows

---

## 5. Performance Problems

| Issue | Count / Location |
|-------|------------------|
| DataFrame `.copy()` calls | 44 across 7 files |
| UI-thread blocking | prep, predict, scaling, PDF, plots |
| No profiling cache | Summary recalculated each view |
| CSV load | Chunks concatenated into single DataFrame |
| No benchmarks | Performance targets unverified |

---

## 6. Reliability Problems

- Import fallbacks with DummyModel (`gui/model_trainer.py`)
- Bare `except: pass` in data_viewer, visualizer, visual_utils
- `handle_missing_and_duplicates` imported but does not exist
- Training preprocessing ≠ inference preprocessing
- `requirements.txt` includes Django, Flask, tbomb (unrelated)

---

## 7. Missing ML Capabilities

See product vision in project specification. Key gaps: 5 task types, full preprocessing pipeline, CV strategies, Optuna, AutoML, explainability, model registry, batch inference, export formats, remote data sources.

---

## 8. Files to Reuse

| Legacy File | Migration Target |
|-------------|------------------|
| `utils/model_utils.py` | `ml_studio/core/training/registry.py` |
| `utils/data_cleaner.py` | `ml_studio/core/transforms/*` |
| `utils/data_loader.py` | `ml_studio/core/dataset.py` ingestion |
| `gui/data_viewer.py` | `ml_studio/gui/widgets/data_table.py` |
| `utils/visual_utils.py` | Reference for `ml_studio/gui/charts/` |
| `assets/icons/*` | `ml_studio/assets/icons/` |
| `data/*.csv` | Sample datasets for tests |

---

## 9. Files to Replace

All legacy `gui/` and `utils/` modules, root `main.py`, and `assets/icons/wather.py` (unrelated script). Removed in Phase 16 after migration complete.

---

## 10. Proposed New Architecture

```
ml_studio/
├── main.py
├── app/           # config, paths, theme, logger, container (DI)
├── core/          # NO Qt imports — usable from CLI/tests/GUI
│   ├── transforms/
│   ├── training/
│   ├── evaluation/
│   ├── inference/
│   └── persistence/
├── gui/           # presentation only
│   ├── widgets/
│   ├── pages/
│   ├── dialogs/
│   ├── charts/
│   └── workers/
├── assets/
└── tests/
```

**Import flow:** `app → core`, `app → gui`, `gui → core` (via DI). Never `core → gui`.

---

## 11. Migration Phases

| Phase | Scope | Status |
|-------|-------|--------|
| 1 | Architecture foundation | In progress |
| 2 | Project system (.mlstudio) | Pending |
| 3 | Dataset abstraction + ingestion | Pending |
| 4 | Schema + profiling | Pending |
| 5 | Virtualized data viewer | Pending |
| 6 | Pipeline / transforms | Pending |
| 7 | Training engine (5 task types) | Pending |
| 8 | Model registry | Pending |
| 9 | Evaluation | Pending |
| 10 | Explainability | Pending |
| 11 | Inference | Pending |
| 12 | AutoML + Optuna | Pending |
| 13 | GUI redesign | Pending |
| 14 | Performance optimization | Pending |
| 15 | Testing (80%+ core coverage) | Pending |
| 16 | Final cleanup | Pending |

---

## 12. Phase Gate Checklist

After each phase:

1. Run tests
2. Run the application
3. Fix errors
4. Verify imports (no circular deps, no core→gui)
5. Verify functionality
6. Update this document

---

## 13. Acceptance Test (Final)

The 23-step end-to-end workflow from the product specification must pass with real data, no fake metrics, no UI freezes, and a single `ml_studio/` architecture.
