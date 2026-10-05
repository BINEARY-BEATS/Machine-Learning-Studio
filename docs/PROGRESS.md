# Progress log

Branch workflow from `docs/CURSOR_PLAN.md`. One task per branch. Do not mark done while acceptance tests fail.

---

## B1 — Training leakage + row misalignment

**Branch:** `fix/b1-leakage`  
**Status:** Done (2026-10-06)  
**Depends on:** —  
**Next:** B2 (encoders / labels / feature schema)

### Problem (context)

- Prep fitted on data that CV/test rows could influence when CV ran on already-processed train.
- Row-dropping transforms (`DropRows`, etc.) could leave `X`/`y` misaligned.
- `build_training_worker` shared Prepare-page step instances, so training mutated the GUI pipeline.

### Done

1. `Pipeline.clone_unfitted()` — rebuild steps via registry + params; never share instances.
2. `core/training/cv_runner.py` — `split_data` (chronological if time series), `fit_preprocessing`, `apply_preprocessing`, `cross_val_score_leakfree`.
3. `trainer.py` — order: split RAW → fit prep on train only → transform test → tune/CV on RAW with per-fold prep clones → fit final → evaluate. `TrainingResult.preprocessing` is the fitted clone.
4. `tuning.py` — Optuna/grid call leak-free CV on RAW train; single `suggest_categorical`; cancel via `study.stop()`.
5. `app_controller.build_training_worker` — passes `Pipeline(...).clone_unfitted()`.
6. `.cursorrules` added (Appendix A).
7. Tests: `ml_studio/tests/training/test_no_leakage.py` (+ updates to correctness / app_controller tests).

### Acceptance checked

- Spy transform: no holdout-test indices in fits; CV fold fits exclude that fold’s val indices.
- Impute(mean) ignores extreme outlier only in test split.
- DropRows trains with `len(X_train_t) == len(y_train_aligned)`.
- Original Prepare pipeline stays unfitted after training.
- Related core/training/GUI tests green. Full GUI suite can hit flaky Windows Qt/GC access violations unrelated to B1.

### Files touched

- `ml_studio/core/pipeline.py`
- `ml_studio/core/training/cv_runner.py` (new)
- `ml_studio/core/training/trainer.py`
- `ml_studio/core/training/tuning.py`
- `ml_studio/gui/app_controller.py`
- `ml_studio/tests/training/test_no_leakage.py` (new)
- `ml_studio/tests/core/test_correctness_phase0.py`
- `ml_studio/tests/gui/test_app_controller.py`
- `.cursorrules`
- `docs/CURSOR_PLAN.md`
- `docs/PROGRESS.md` (this file)

---

## B2 — Encoders, labels and feature schema

**Branch:** `fix/b2-encoders`  
**Status:** Done (2026-10-06)  
**Depends on:** B1  
**Next:** B3 (Predict page typed inputs)

### Problem (context)

- Feature LabelEncoding created fake ordinals and was discarded / not reusable at predict.
- Classification returned `0/1` instead of original string labels.
- No feature schema for coerce/validate; NaNs crashed models without an Impute step.

### Done

1. `prepare_for_training` — no feature encoding; `meta["target_classes"]` for string targets.
2. `transforms/auto_encode.py` — `AutoEncode` (one-hot / freq / datetime / bool + impute); hidden from GUI picker, kept in registry for clone.
3. Trainer appends fitted `AutoEncode` when non-numeric/NaN remain; builds `feature_schema` from RAW `X_train`; stores `target_classes`.
4. `InferencePipeline` — `target_classes`, `feature_schema`, `coerce_row`, `decode`.
5. `Predictor` — coerce → predict → decoded label + `{label: prob}`; batch keeps original cols + proba columns.
6. `TrainingWorker` uses `prepare_for_training`.
7. Tests: `ml_studio/tests/inference/test_roundtrip.py`.

### Acceptance checked

- TrainingWorker → registry → `predict_single({"city":"Lahore","age":"31"})` returns `"yes"`/`"no"` + prob dict.
- Unseen category + blank numeric do not raise.
- Non-numeric text in numeric field → `ValueError` naming the field.
- 85 related core/training/inference/GUI tests green.

---

## B3 — Predict page: typed, validated inputs

**Branch:** `fix/b3-predict-inputs`  
**Status:** Done (2026-10-06)  
**Depends on:** B2  
**Next:** B4 (DBSCAN / unsupervised) or B5/B6/B8 as independent

### Problem (context)

- Predict form was all `QLineEdit` strings; no schema casting/validation/category lists.
- Batch wrote beside the input with no column preflight; predict button used a connect flag hack.

### Done

1. Schema-driven inputs via `gui/widgets/schema_form.py` (numeric/categorical/boolean/datetime; fallback LineEdit).
2. `_run_single` uses `coerce_row`; field `error=true` + inline banner (no QMessageBox for validation); prob bars.
3. Batch preflight (`BatchPredictWorker.preflight`) + save dialog default `*_predictions.csv`.
4. Predict button connected once in `_build_single_tab`.
5. QSS: `[error="true"]`, `#ValidationError`, `#ClassProbBar` using `palette.danger` / primary.
6. Tests: `tests/gui/test_predict_schema.py`.

### Acceptance checked

- Categorical combobox from schema; invalid numeric flagged; decoded label + bars; preflight lists missing columns.
- Existing predict/pages/workers tests updated and green.

---

## B4 — DBSCAN / unsupervised fixes

**Branch:** `fix/b4-dbscan`  
**Status:** Done (2026-10-06)  
**Depends on:** B1  
**Next:** B5 (data table) or B6/B8 as independent; F1 after B1–B2

### Problem (context)

- Trainer called `model.predict` after fit for clustering/anomaly; DBSCAN has no `predict` → crash despite being in the registry.
- Silhouette included noise label `-1`.
- `InferencePipeline.predict` assumed every estimator exposes `predict`.
- No cluster profile UI; anomaly metrics lacked score summaries.

### Done

1. `core/training/unsupervised.py` — `fit_predict_labels` / `predict_new` (DBSCAN nearest-core ≤ eps else `-1`; LOF novelty-safe).
2. `trainer.py` + `InferencePipeline.predict` use the helpers for unsupervised tasks.
3. `metrics.py` — noise-excluded silhouette/DBI, `noise_ratio`, `cluster_profile`, anomaly `score_*` from `decision_function` / `score_samples`.
4. EvaluatePage — Metrics / Clusters tabs via `evaluate_clusters.py`.
5. Tests: `tests/training/test_unsupervised.py`.

### Acceptance checked

- DBSCAN trains on blobs; `noise_ratio` present; far point → `-1`.
- KMeans/GMM silhouette + cluster counts OK.
- IsolationForest/LOF `anomaly_count` > 0; IF score summary present.
- 31 related training/evaluation/inference tests green.

### Files touched

- `ml_studio/core/training/unsupervised.py` (new)
- `ml_studio/core/training/trainer.py`
- `ml_studio/core/evaluation/metrics.py`
- `ml_studio/core/persistence/serializer.py`
- `ml_studio/gui/pages/evaluate_page.py`
- `ml_studio/gui/pages/evaluate_clusters.py` (new)
- `ml_studio/tests/training/test_unsupervised.py` (new)
- `docs/PROGRESS.md`

---

## B5 — Data table sort/filter correctness + speed

**Branch:** `fix/b5-data-table`  
**Status:** Done (2026-10-06)  
**Depends on:** —  
**Next:** B6 (ingestion) or B7/B8 as independent

### Problem (context)

- Filter stored index *labels* but `data()` used them as iloc *positions* → wrong rows after sort / non-RangeIndex.
- Filter did full-frame `astype(str)` + Python `axis=1` apply (very slow).
- `sort()` mutated/copied `_df` and cleared the filter.
- DataPage column combo was not wired; no visible row count.

### Done

1. `DataFrameTableModel` — `_order` / `_visible` position arrays; `_df` never mutated.
2. Stable sort of current visible rows (NaN last); filter re-applied over `_order`.
3. Vectorized literal substring filter; skip numeric columns when needle is not numeric-looking.
4. DataPage — column filter + search wired; "Showing N of M rows" label.
5. Vertical headers show original 1-based row positions.
6. Slow benchmark: 1M×10 filter < 3s; `pytest.mark.slow` registered.

### Acceptance checked

- Shuffled non-RangeIndex + sort + filter returns correct values.
- Column-restricted filter; backslash literal; sort keeps filter.
- Existing data_table tests green; slow 1M filter benchmark < 3s.

### Files touched

- `ml_studio/gui/widgets/data_table.py`
- `ml_studio/gui/pages/data_page.py`
- `ml_studio/tests/gui/test_data_table_model.py`
- `ml_studio/tests/gui/test_pages.py`
- `ml_studio/tests/benchmarks/benchmark_io.py`
- `pyproject.toml`
- `docs/PROGRESS.md`

---

## B6 — Ingestion: TSV / semicolon / sheets / preview

**Branch:** `fix/b6-ingestion`  
**Status:** Done (2026-10-06)  
**Depends on:** —  
**Next:** B7 (pipeline serialization)

### Problem (context)

- `.tsv` used CSV loader with `sep=","` → one column; semicolon CSVs same.
- Excel only first sheet; preview crashed on jsonl/tsv/sqlite; remote URL suffix ignored query strings and double-opened.

### Done

1. `LocalFileSource.preview` for every supported format; delimiter sniff (`.tsv` forces tab); `list_sheets` / `list_tables`.
2. `load()` accepts `sep`, `encoding`, `sheet`, `table`.
3. `ImportPreviewDialog` uses core preview; sheet/table/delimiter controls; `chosen_options()` → worker.
4. `RemoteFileSource` single open; extension from `urlparse(...).path`.

### Acceptance checked

- TSV + semicolon CSV column counts; json/jsonl preview; multi-sheet xlsx; multi-table sqlite; previews for common extensions; remote `?token=` extension.

### Files touched

- `ml_studio/core/ingestion.py`
- `ml_studio/gui/dialogs/import_preview.py`
- `ml_studio/gui/workers/dataset_worker.py`
- `ml_studio/gui/app_controller.py`
- `ml_studio/gui/main_window.py`
- `ml_studio/tests/core/test_ingestion_b6.py` (new)
- `docs/PROGRESS.md`

---

## B7 — Pipeline / project serialization safety

**Branch:** `fix/b7-serialization`  
**Status:** Done (2026-10-06)  
**Depends on:** —  
**Next:** B8 (verified bugs bundle)

### Problem (context)

- Fitted numpy state in `Pipeline.to_dict` broke JSON project saves.
- `from_dict` marked transforms fitted without restoring sklearn state.
- Loose try/except transform tests hid failures.

### Done

1. `core/serialization.py` — `to_jsonable`.
2. `Pipeline.to_dict(include_state=...)`; session_io writes `include_state=False`.
3. `from_dict` without state → unfitted steps; Power/Quantile/Polynomial/Binning/Impute knn restore real state or stay unfitted; IsolationForestFilter always unfitted from JSON.
4. Parametrized round-trip tests replace semantic blocker.

### Acceptance checked

- Round-trip for registered transforms; project pipeline JSON unfitted + jsonable.

### Files touched

- `ml_studio/core/serialization.py` (new)
- `ml_studio/core/pipeline.py`
- `ml_studio/core/persistence/session_io.py`
- `ml_studio/transforms/base.py`
- `ml_studio/transforms/scaling.py`
- `ml_studio/transforms/feature_eng.py`
- `ml_studio/transforms/outliers.py`
- `ml_studio/transforms/missing.py`
- `ml_studio/tests/transforms/test_transform_roundtrip.py` (new)
- `ml_studio/tests/transforms/test_semantic_blocker3.py`
- `docs/PROGRESS.md`

---

## B8 — Small verified bugs (bundle)

**Branch:** `fix/b8-bugs`  
**Status:** Done (2026-10-06)  
**Depends on:** —  
**Next:** F1 (evaluation upgrade) or independent B leftovers

### Done (one commit each)

1. Profile zeros + `ColumnProfile.top_values` / Data page Top values column.
2. StatCard N/A hint reachable.
3. Hard-coded colors → `#DangerText` / `#SuccessText` / `#WarningText` / `#TaskProgressPanel` QSS.
4. Recipe description reads top-level `description`.
5. Recipe tests use `tmp_path`; removed `recipes/test_recipe.yaml`.
6. Evaluate primary `metric_key` per task + `run_count()` (main_window uses it).

### Acceptance checked

- New regression tests for each fix; theme QSS assertions extended.
