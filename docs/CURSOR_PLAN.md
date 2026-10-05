# ML Studio — Master Plan + Cursor Prompts

Goal: turn ML Studio from "generic AutoML GUI" into **trustworthy, local ML for spreadsheet users**, with a Flutter companion app that talks to models you trained.

Drop this file in your repo as `docs/CURSOR_PLAN.md`.

---

## 0. How to use this

1. Add the `.cursorrules` file (Appendix A) once.
2. One task = one branch: `git checkout -b fix/b1-leakage`.
3. Paste **one** prompt into Cursor (Agent mode). Review the diff yourself.
4. Run `pytest ml_studio/tests -q` and `python scripts/check_rules.py`. Green? Commit. Next.
5. If Cursor wanders into unrelated files, stop it and re-prompt: "Only touch the files listed."

Rules for yourself: never batch prompts, never skip the tests listed in ACCEPTANCE, and learn from each diff (this is how you learn ML properly).

### Order and dependencies

| # | Task | Depends on | Effort |
|---|------|-----------|--------|
| B1 | Training leakage + row alignment | — | 1–2 d |
| B2 | Encoders / labels / feature schema | B1 | 1–2 d |
| B3 | Predict page: typed inputs | B2 | 1 d |
| B4 | DBSCAN / unsupervised fixes | B1 | 0.5 d |
| B5 | Data table sort/filter | — | 0.5–1 d |
| B6 | Ingestion (TSV, preview, sheets) | — | 1 d |
| B7 | Project/pipeline serialization | B1 | 1 d |
| B8 | Small verified bugs bundle | — | 0.5–1 d |
| F1 | Evaluation upgrade + charts | B1, B2 | 2 d |
| F2 | Honest model verdict | F1 | 1 d |
| F3 | Persist experiments, per-project registry | B2, B7 | 1–2 d |
| F4 | Column roles become real | B1 | 1–2 d |
| F5 | PDF model report | F1, F2 | 1 d |
| F6 | AutoML leaderboard in GUI | B1 | 1–2 d |
| TS1–TS3 | Real time-series forecasting | B1, F4 | 4–6 d |
| D1–D2 | Drift monitoring | B2 | 2 d |
| A1–A2 | Export as API + QR | B2, (TS for forecast) | 2–3 d |
| M1–M7 | Flutter companion app | A1 | 6–10 d |
| E1 | Optional AI explainer | F1, F2 | 1–2 d |
| H1–H4 | Security, packaging, CI, release, cleanup | any time (H2 early!) | 3–4 d |

Realistic total: ~6–8 weeks part-time. **Do B1–B3 first.** Without them the metrics are optimistic and prediction on text columns is broken.

---

# PHASE 1 — Bug fixes

## B1 — Training leakage + row misalignment

**Where:** `core/training/trainer.py`, `core/pipeline.py`, `core/training/tuning.py`, `gui/app_controller.py`

**Problem:** `Trainer.train()` runs `preprocessing.fit_transform(X, y)` on the full dataset *before* `train_test_split` and CV. Imputer means, scalers, target encoders, selectors all see test rows, so metrics are optimistic. Row-dropping transforms (`DropRows`, `IsolationForestFilter`) return fewer rows than `y` (misalignment). `build_training_worker` shares step instances with the Prepare page, so training mutates the GUI pipeline.

```text
CONTEXT
File: ml_studio/core/training/trainer.py (+ core/pipeline.py, core/training/tuning.py, gui/app_controller.py).
Bug: Trainer.train() calls preprocessing.fit_transform(X, y) on the FULL dataset and only then runs train_test_split and cross_val_score. Imputer means, scalers, target encoders and feature selectors see test/CV-validation rows = data leakage, so metrics are optimistic. Transforms that drop rows (DropRows, IsolationForestFilter) return fewer rows than y and break alignment. AppController.build_training_worker builds Pipeline(steps=list(active)) which SHARES step instances with the Prepare page, so training mutates the GUI pipeline's fitted state.

TASK
1. core/pipeline.py: add Pipeline.clone_unfitted() -> Pipeline. Rebuild each step with get_transform(type(step).__name__)(**step.params) and copy the `enabled` attribute. Never share instances.
2. New core/training/cv_runner.py (no Qt imports):
   - split_data(X, y, config) -> X_train, X_test, y_train, y_test on RAW (un-preprocessed) data. Keep the existing stratify logic. If config.is_time_series: chronological split (shuffle=False, no stratify).
   - fit_preprocessing(pipeline, X_train, y_train) -> (fitted_pipeline, X_train_t, y_train_aligned): clone, fit_transform on TRAIN only, then y_train_aligned = y_train.loc[X_train_t.index].
   - apply_preprocessing(fitted, X, y) -> (X_t, y_aligned) using transform only.
   - cross_val_score_leakfree(model_factory, preprocessing, X_train, y_train, cv, scoring) -> np.ndarray: per fold clone preprocessing, fit on fold-train, transform fold-val, align y, fit a fresh clone of the model, score with sklearn.metrics.get_scorer(scoring).
3. trainer.py: use cv_runner. Order: split RAW -> fit preprocessing on train -> transform test -> (optional tuning) -> leak-free CV on RAW X_train -> fit final model on processed train -> evaluate on processed test. TrainingResult.preprocessing must be the FITTED pipeline from step 2. Clustering/anomaly (no split) may fit on all data.
4. tuning.py + Trainer._maybe_tune: the tuning objective must call cross_val_score_leakfree with RAW X_train. Replace the three identical branches in OptunaTuner.objective with one suggest_categorical. Implement cancel via study.stop() instead of raising OptunaError.
5. app_controller.build_training_worker: pass Pipeline.clone_unfitted() of the enabled steps.

CONSTRAINTS
Keep public signatures of Trainer.train and TrainingConfig backward compatible. No Qt in core. Functions <= 40 lines, files <= 400 lines.

ACCEPTANCE (write tests FIRST in ml_studio/tests/training/test_no_leakage.py)
- A recording BaseTransform spy stores the indices it was fitted on; assert no test-split index and no CV-validation index is ever in its fit set.
- Impute(mean) with an extreme outlier only in the test split: fitted mean == train-only mean.
- Pipeline containing DropRows trains without error; len(X_train_t) == len(y_train_aligned).
- Prepare page pipeline is still unfitted after training.
- All existing tests pass; python scripts/check_rules.py passes.
```

## B2 — Encoders, labels and feature schema are lost

**Where:** `core/training/data_prep.py`, `trainer.py`, `core/persistence/serializer.py`, `model_registry.py`, `core/inference/predictor.py`, `gui/workers/training_worker.py`

**Problem:** `prepare_for_training` LabelEncodes every non-numeric feature and a text target on the whole dataframe, then `TrainingWorker` discards the encoders (`prepared_df, target, features, _ = ...`). Models trained on categorical columns can't predict on raw input; classification outputs `0/1` instead of original labels. LabelEncoder on features gives fake ordering to linear/KNN/SVM. NaNs crash models when no Impute step exists.

```text
CONTEXT
prepare_for_training() (core/training/data_prep.py) LabelEncodes every non-numeric FEATURE on the whole dataframe and encodes a text TARGET, but the encoders (`meta`) are discarded by TrainingWorker. A model trained on categorical columns cannot predict on new raw input and classification predictions come back as 0/1 instead of original labels. LabelEncoder on features also creates a fake ordering for linear/KNN/SVM. Models like LogisticRegression crash on NaN if the user added no Impute step.

TASK
1. data_prep.prepare_for_training: stop encoding features. Only select columns, drop rows with missing target, enforce min rows, and (classification with non-numeric target) fit LabelEncoder on the target, returning classes in meta["target_classes"]. Keep the return shape; update callers and tests/core/test_fixes.py.
2. New transforms/auto_encode.py: class AutoEncode(BaseTransform), fitted on TRAIN only. numeric -> passthrough; bool -> int; datetime -> year/month/day/dow; categorical/object with <=15 uniques -> one-hot (unknown -> all zeros); >15 uniques -> frequency encoding. Then impute remaining NaN (median for numeric, 0 for encoded). Full to_dict/from_dict. Exclude it from the GUI picker the same way CustomPython is excluded in transforms/registry.py.
3. Trainer: after the user's preprocessing, if any non-numeric column or NaN remains, append AutoEncode automatically inside the fitted preprocessing so it is saved with the model.
4. core/persistence/serializer.py: InferencePipeline gains `target_classes: list | None` and `feature_schema: dict`, built at train time from RAW X_train: per column {kind, dtype, nullable, categories (<=50), min, max, ref} where `ref` = median (numeric) or mode (categorical). Add methods: coerce_row(dict) -> DataFrame (cast per schema, "" -> NaN, unknown category allowed, uncastable value raises ValueError carrying a {field: message} dict) and decode(preds) -> original labels. TrainingResult and ModelRegistry.register carry both fields. Predictor.predict_single returns the DECODED label and probabilities as {label: prob}.
5. Predictor.predict_batch: same decoding; keep original columns.

ACCEPTANCE (tests/inference/test_roundtrip.py)
- Train on a df with string categoricals + string target ("yes"/"no") via the TrainingWorker.do_work path -> register in ModelRegistry(tmp_path) -> load_pipeline -> predict_single({"city": "Lahore", "age": "31"}) returns "yes"/"no" (str) and a {label: prob} dict.
- Unseen category and blank numeric do not raise.
- Non-numeric text in a numeric field raises ValueError naming the field.
- Existing tests pass.
```

## B3 — Predict page: typed, validated inputs

**Where:** `gui/pages/predict_page.py`, `gui/workers/batch_predict_worker.py` (needs B2)

```text
CONTEXT
gui/pages/predict_page.py builds a QLineEdit per feature and sends raw strings to Predictor.predict_single: no casting, no validation, no category lists, unlabeled results. Batch predict writes next to the input file and doesn't validate columns. Requires B2 (InferencePipeline.feature_schema / coerce_row / decode).

TASK
1. bind_predictor: build inputs from predictor.pipeline.feature_schema. numeric -> QLineEdit + QDoubleValidator (placeholder shows range/ref; blank = missing); categorical -> editable QComboBox of categories plus "(missing)"; boolean -> tri-state QCheckBox; datetime -> QDateTimeEdit or ISO-validated line edit. Scrollable form, label shows column kind. Fall back to QLineEdit when schema is missing (old models).
2. _run_single: call pipeline.coerce_row. On ValueError mark offending fields (dynamic property error=true + tooltip; add the QSS rule in theme_qss.py using palette.danger) and show the message inline. No QMessageBox for validation. Result card: decoded label, class probabilities sorted with bars (QProgressBar styled via QSS).
3. Batch: BatchPredictWorker preflight reads the header only and reports missing/extra columns before running. Output chosen via QFileDialog (default <input>_predictions.csv). Output keeps original columns + prediction + one probability column per class named with ORIGINAL class names.
4. Remove the _predict_connected flag hack: connect the predict button once in _build_single_tab.

ACCEPTANCE (pytest-qt, offscreen)
categorical combobox populated from schema; invalid numeric field flagged; predict returns decoded label; batch preflight lists missing columns; update existing predict tests.
```

## B4 — DBSCAN crashes, noise-aware clustering metrics

**Where:** `core/training/trainer.py`, `core/evaluation/metrics.py`, `core/persistence/serializer.py`

```text
CONTEXT
Trainer does model.fit(X) then model.predict(X_train) for clustering/anomaly. DBSCAN has no predict() -> AttributeError, yet it's in MODEL_REGISTRY. _clustering_metrics computes silhouette including the noise label -1. Predictor/InferencePipeline.predict also assumes predict() exists.

TASK
1. New core/training/unsupervised.py: fit_predict_labels(model, X) -> labels (use labels_ if present, else fit_predict, else predict). predict_new(model, X_new): if hasattr predict -> use it; for DBSCAN assign each row the label of the nearest core sample (components_ with the matching labels_[core_sample_indices_]) if distance <= eps else -1.
2. Trainer and InferencePipeline.predict use these helpers.
3. metrics: silhouette excludes noise (-1); add noise_ratio and davies_bouldin; if <2 clusters omit silhouette (metric_na_reason already explains).
4. Store per-cluster profile (size + feature means) in metrics["cluster_profile"] and show it in a new "Clusters" tab on EvaluatePage (table).
5. Anomaly: add score distribution summary (decision_function/score_samples when available).

ACCEPTANCE
DBSCAN trains on blobs without error; noise_ratio computed; predict_new returns -1 for a far point; KMeans/GMM results unchanged; IsolationForest/LOF anomaly_count > 0.
```

## B5 — Data table: wrong rows after sort/filter, slow filter

**Where:** `gui/widgets/data_table.py`, `gui/pages/data_page.py`

**Problem:** `apply_filter` stores `self._df.index[mask]` (index *labels*) but `data()` uses them as `iloc` *positions* → wrong rows after sort or with a non-Range index. Filter does `astype(str)` + a Python row-apply over the whole frame (minutes on 1M rows). `sort()` replaces the df with a sorted copy and silently resets the filter. `DataPage._column_filter` isn't connected.

```text
CONTEXT
gui/widgets/data_table.py DataFrameTableModel:
- apply_filter stores index LABELS in _filtered_indices but data() uses them as iloc POSITIONS -> wrong rows after sort or when the index isn't a RangeIndex.
- Filter does astype(str) on the whole frame and a Python-level row apply (axis=1): minutes on 1M rows.
- sort() replaces self._df with a sorted copy and silently resets the filter.
- DataPage._column_filter combo exists but is not connected.

TASK
1. Keep `_order: np.ndarray` of row positions and `_visible: np.ndarray` (after filtering). data(), rowCount(), headerData use positions. Never mutate or copy self._df.
2. sort(column, order): stable argsort of that column (NaN last) applied to the CURRENT visible positions.
3. apply_filter(text, column=None): vectorized. For each target column: s.astype(str).str.contains(re.escape(text), case=False, regex=True, na=False), OR-reduced with numpy. Must stay literal (existing test_filter_with_backslash). Skip str-conversion of numeric columns when the text is not numeric-looking.
4. Connect DataPage._column_filter -> model.apply_filter(search_text, column). Add a "Showing N of M rows" label.
5. Vertical header shows original 1-based row numbers.

ACCEPTANCE
Tests: shuffled non-Range index + sort + filter returns correct values; column-restricted filter; all existing data_table tests pass; benchmark (mark slow, update tests/benchmarks/benchmark_io.py): 1M rows x 10 cols filter < 3 s.
```

## B6 — Ingestion: TSV/semicolon CSV, preview crashes, Excel sheets

**Where:** `core/ingestion.py`, `gui/dialogs/import_preview.py`, `gui/main_window.py`, `gui/workers/dataset_worker.py`

```text
CONTEXT
- core/ingestion.py: ".tsv" maps to the csv loader which uses default sep="," -> TSV loads as ONE column. Semicolon CSVs (common in Excel exports) have the same issue. Excel loads only the first sheet. RemoteFileSource opens the URL twice and uses Path(url).suffix (breaks on query strings).
- gui/dialogs/import_preview.py preview_dataframe: pd.read_json(path, nrows=...) raises ValueError without lines=True; .jsonl, .tsv, .orc, .sqlite/.db fall to read_csv and crash or show garbage.

TASK
1. core/ingestion.py: add LocalFileSource.preview(nrows=100, **opts) for EVERY supported format: csv/tsv via delimiter sniffing (csv.Sniffer on first 64 KB, fallback ","; .tsv forces "\t"); json (try records, then lines); jsonl; parquet (pyarrow first batch); feather; orc; sqlite (first or chosen table, LIMIT n); excel (sheet_name option). Add list_sheets(path) and list_tables(path). load() accepts the same opts (sep, encoding, sheet, table).
2. ImportPreviewDialog uses LocalFileSource.preview (no pandas logic in gui). Shows detected delimiter/encoding; lets the user choose sheet/table (combo) and override delimiter. Return the chosen options; MainWindow._import_dataset passes them to DatasetLoadWorker.
3. RemoteFileSource: single open; derive extension from urllib.parse.urlparse(url).path.

ACCEPTANCE
Tests: tsv and semicolon csv load with correct column counts; json (non-lines) and jsonl preview; multi-sheet xlsx sheet selection; multi-table sqlite table selection; preview for every extension in LocalFileSource.SUPPORTED does not raise; remote URL with ?token=x resolves extension (mock fsspec).
```

## B7 — Project/pipeline serialization is unsafe and dishonest

**Where:** `core/pipeline.py`, `core/persistence/session_io.py`, `transforms/*.py`

```text
CONTEXT
Pipeline.to_dict() serializes fitted state and session_io writes it into project JSON. Problems:
(1) numpy scalars from fitted state (np.int64 mode values, etc.) are not JSON-serializable -> saving a project after training can crash;
(2) every transform.from_dict sets _is_fitted=True even if sklearn state was not restored (Power, Quantile, Polynomial, Binning, IsolationForestFilter, Impute knn) -> AttributeError or silent no-op on transform;
(3) Power.from_dict builds a PowerTransformer with an unfit private _scaler -> wrong output.
tests/transforms/test_semantic_blocker3.py wraps everything in try/except: Exception: pass, so none of this is caught.

TASK
1. Pipeline.to_dict(include_state: bool = True). Project persistence (session_io.write_session) uses include_state=False -> store only class, params, enabled. Pipeline.from_dict builds UNFITTED steps when no state is present (do not set _is_fitted).
2. Fitted state lives only in model artifacts via joblib (already true in ModelRegistry). Document this in BaseTransform docstring.
3. Each transform's from_dict sets _is_fitted=True ONLY if all state needed by _transform was restored; otherwise leave it unfitted. Fix or remove half-restores for Power (store lambdas_ + internal scaler mean_/scale_), Quantile, Binning, Polynomial (rebuild poly and fit on a zero row of the right width), IsolationForestFilter, Impute knn.
4. Add core/serialization.py to_jsonable(obj) (numpy scalars/arrays, pandas Timestamp, sets) used by to_dict and session_io.
5. Replace the loose try/except tests with a parametrized test over transforms.registry.list_all(): fit on a small frame -> to_dict -> json.dumps -> from_dict -> transform equals the original (assert_frame_equal), OR from_dict leaves it unfitted and transform raises a clear "not fitted" ValueError.

ACCEPTANCE
Parametrized round-trip test passes for every registered transform; saving a project after training no longer crashes; existing persistence tests pass.
```

## B8 — Small verified bugs (bundle, one commit each)

```text
Fix these small, verified bugs. One commit per item, each with a regression test.

a) gui/pages/data_page.py apply_profile_result: `str(col.min or "")` and `f"{col.mean:.4g}" if col.mean else ""` hide legitimate zeros. Use `is not None`. Also fill the "Top values" column (currently always "—") with the top 3 values: add ColumnProfile.top_values in core/profiling.py.
b) gui/widgets/stat_card.py set_value: the N/A hint branch is unreachable (`raw is None` checked inside `if ... raw is not None`). Restructure so None/NaN shows metric_na_reason and the metric_na color.
c) Hard-coded colors bypassing the theme: models_page.py (#3FB950), step_config.py ("color: red;"), preview_modal.py (green/red/orange), prepare_page.py and pipeline_step.py inline styles, task_progress.py rgba scrim. Replace with objectName + QSS rules in theme_qss.py (e.g. #DangerText, #SuccessText, #WarningText) using palette tokens; extend tests/gui/test_theme.py.
d) prepare_page._on_recipe_selected reads data["metadata"]["description"] but recipe YAMLs have a top-level `description`. Read both.
e) tests/core/test_recipes.py writes recipes/test_recipe.yaml into the REAL recipes/ dir (it then shows up in the app's recipe list). Monkeypatch RECIPES_DIR to tmp_path and delete recipes/test_recipe.yaml from the repo.
f) EvaluatePage primary StatCard has metric_key="r2" hard-coded; set the key per task (f1/accuracy/silhouette) so colors are right for classification/clustering.
g) HomePage.refresh_stats reads the private pages["evaluate"]._runs; add a public EvaluatePage.run_count().

ACCEPTANCE: each fix has a test; python scripts/check_rules.py passes.
```

---

# PHASE 2 — Trust features (what makes it worth using)

## F1 — Evaluation upgrade + real charts

```text
CONTEXT
Trainer._compute_metrics never passes y_proba, so roc_auc is never computed. EvaluatePage shows only numbers; pyqtgraph is in requirements but unused (gui/charts/__init__.py is a stub with a hard-coded white background).

TASK
Core (no Qt):
1. Trainer: after predict, compute y_proba when the estimator has predict_proba and pass it to compute_metrics. metrics.py: roc_auc (binary: proba[:,1]; multiclass: multi_class="ovr", average="weighted", skip if a class is missing in y_test), balanced_accuracy, per_class (classification_report output_dict), log_loss. Regression: median_ae, mape (guard zeros), residual mean/std.
2. New core/evaluation/curves.py returning plain arrays/dicts: roc_curve_data, pr_curve_data, confusion_matrix_data(labels), residuals_data, pred_vs_actual (downsample to 5000 pts), feature_importance (permutation on the TEST set, n_repeats=5, cap 2000 rows).
3. TrainingResult gets `eval_artifacts: dict` holding these (JSON-serializable, size-capped). Persist via the registry as a json file next to the model artifact.
GUI:
4. gui/charts: PlotCard widgets on pyqtgraph: ConfusionMatrixPlot (ImageItem + labels), RocPlot, ResidualPlot, PredVsActualPlot, ImportanceBar. Colors ONLY from color_token(mode, ...) (no hex); rebuild on set_theme_mode. Graceful label fallback if pyqtgraph is missing. Remove the white background hard-code.
5. EvaluatePage: PillTabs under the header: Overview | Charts | Per-class. Charts tab chooses plots by task.

ACCEPTANCE
Metric tests (roc_auc present for binary and multiclass logistic); curves array shapes; EvaluatePage renders charts for a classification and a regression result offscreen; no chart gets > 5000 points.
```

## F2 — Honest model verdict (replace the blunt quality gate)

```text
CONTEXT
AppController.on_training_complete has a hard gate (r2 < 0 or accuracy < baseline => model not saved). Too blunt, and it misses the most common real failures.

TASK
core/evaluation/verdict.py: build_verdict(result, train_stats, config) -> Verdict{level: "good"|"fair"|"weak"|"unreliable", reasons: list[str], suggestions: list[str]}. Checks:
- Baseline: DummyRegressor/DummyClassifier(most_frequent) score on the same test split -> "beats baseline by X".
- Overfit gap: train score vs CV mean vs test score; flag gap > 0.15.
- Suspicious perfection: test score > 0.99 (accuracy/R²/AUC) => "possible target leakage"; list top features; flag a single feature with > 60% importance share.
- Class imbalance: minority < 10% => recommend F1/balanced accuracy.
- Small data: n_train < 20 * n_features or < 200 rows.
- CV instability: cv std > 0.1.
Trainer must compute the train-set score and store metrics["baseline_score"].
GUI: Verdict banner (TagChip + bullet reasons) on top of EvaluatePage and in ModelsPage detail. Replace the hard gate: always evaluate; if R² < 0 or accuracy < baseline show "Save anyway?" (default No); saved weak models get a "weak"/"unreliable" tag in ModelVersion.tags.

ACCEPTANCE
Synthetic tests: pure-noise target -> weak/unreliable with a baseline reason; target copied into a feature -> leakage reason; 40-row dataset -> small-data reason; clean linear data -> good. Update test_app_controller quality-gate tests.
```

## F3 — Persist experiments, per-project registry, dataset lineage

```text
CONTEXT
(1) AppController.on_training_complete calls registry.register(result, name=...) WITHOUT dataset_id/dataset_version, so the Models page "Dataset" column is always "—". (2) The registry lives at ~/.mlstudio/models, shared by every project. (3) EvaluatePage._runs is memory-only: experiments vanish on restart. (4) .mlstudio archives create empty models/ and experiments/ folders.

TASK
1. on_training_complete: pass dataset_id, dataset_version and dataset name; store the TrainingConfig (split, cv, tuning, features) in ModelVersion.hyperparameters plus pipeline hash.
2. Project-scoped registry: <project_dir>/models when the project has a path, else ~/.mlstudio/scratch/<project_id>/models. Save-As copies artifacts. write_session archives models/<id>/ + experiments.json inside the .mlstudio zip; read_session restores them.
3. Move the ExperimentRun dataclass from gui/pages/evaluate_page.py into core/experiments.py with to_dict/from_dict; EvaluatePage imports it. Experiments round-trip with the project; HomePage counts come from it.
4. ModelsPage actions: Rename, Notes/Tags editor, Delete (confirm), "Mark as production" tag.

ACCEPTANCE
Round-trip test: train -> save project -> new ProjectManager -> open -> experiments + models present; load_pipeline + predict_single work; Dataset column shows the dataset name.
```

## F4 — Column roles become real

```text
CONTEXT
SchemaEditor lets users set roles (feature/target/id/group/time_index/weight/drop) but they are cosmetic: PreparePage._on_role_changed only stores schema_overrides and sets dataset.target_column; TrainPage.set_dataset ignores them; Pipeline._validate_roles reads X.attrs["roles"] which the GUI never sets; recommend_cv_strategy(has_groups=...) never gets groups.

TASK
1. core/session_schema.py: SessionSchema (roles dict + kinds) owned by AppController as the single source of truth, with features(), target(), group(), weight(), time_index(), excluded().
2. PreparePage emits roles_changed(dict); AppController updates SessionSchema, TrainPage (target combo + feature checkboxes: id/group/weight/time_index/drop are unchecked AND disabled with a tooltip) and dataset.target_column.
3. On dataset load suggest roles: near-unique int/str column or name matching id|uuid|index -> TagChip "Looks like an ID — exclude?"; datetime column -> suggest time_index.
4. TrainingConfig gains group_column, weight_column, time_column. Trainer: GroupKFold + GroupShuffleSplit when group set; pass sample_weight to fit() when the estimator supports it (inspect signature); chronological split when time_column set; set X.attrs["roles"] before preprocessing so Pipeline._validate_roles protects target/id columns.

ACCEPTANCE
id column -> absent from TrainingConfig.feature_columns; group set -> no group appears in both train and test; time_index -> max(train time) <= min(test time); weight reaches fit (spy estimator).
```

## F5 — PDF model report

```text
TASK
core/reporting/model_report.py (no Qt): build_pdf(model_version, eval_artifacts, verdict, dataset_profile, out_path) with reportlab (lazy import; clear error if missing). Sections: title/date/dataset (rows, cols); task, target, split, CV; verdict + reasons; metrics table vs baseline; charts rendered with matplotlib (Agg backend) to in-memory PNGs from eval_artifacts; top 10 feature importances; data-quality issues from profiling; reproducibility block (pipeline hash, library versions, random_state); auto-generated "Limitations" paragraph from the verdict. A4, page numbers, consistent fonts.
GUI: "Export report (PDF)" on EvaluatePage and ModelsPage detail; runs in new gui/workers/report_worker.py; QFileDialog for the path; toast on success.

ACCEPTANCE
Tests generate a PDF for a small classification and a regression model; file > 10 KB; page count >= 2; works headless; missing eval_artifacts skips charts without error.
```

## F6 — AutoML leaderboard in the GUI

```text
CONTEXT
core/training/automl.py AutoMLRunner exists but isn't in the GUI and has problems: model_id lookup via a __import__ hack, takes the first N models regardless of cost, scoring="silhouette" (invalid for cross_val_score), CV on preprocessed data (leak), silently skips failures, `parameters=meta.default_params`.

TASK
1. Rewrite AutoMLRunner: input RAW X/y + preprocessing; reuse cv_runner.cross_val_score_leakfree (B1); iterate registry model ids cheapest-first (cost low -> high); honor max_runtime and cancel; record error strings for failed models; supervised tasks only (clustering/anomaly -> clear ValueError).
2. AutoMLWorker with progress.
3. TrainPage Models step: checkbox "Run quick leaderboard". Dialog table: rank, model, CV mean±std, test score, train time, inference time. Double-click trains that model with defaults; "Tune best with Optuna" button.
4. Command palette "Run AutoML…" opens this flow (replace _open_train_for_automl).

ACCEPTANCE
Tiny-dataset tests: leaderboard sorted descending; failures recorded, not raised; cancel stops within one model; deterministic with random_state.
```

---

# PHASE 3 — Forecasting (your niche feature)

Today "TIME_SERIES" = regression models + `TimeSeriesSplit`, and `train_test_split` shuffles → future leaks into training. Fix this and it becomes the most useful feature for real businesses (sales, stock, demand).

## TS1 — Time-series features + chronological training

```text
CONTEXT
TIME_SERIES today = regression models + TimeSeriesSplit, but train_test_split shuffles (future leaks into training) and there are no lag/rolling/calendar features.

TASK (core only)
1. New transforms/timeseries.py (all with get_schema/to_dict/from_dict + tests):
   - LagFeatures(columns, lags=[1,7,14])
   - RollingStats(columns, windows=[7,28], stats=[mean,std,min,max]) — MUST be shifted so row t never sees y_t
   - CalendarFeatures(dow, week, month, quarter, is_month_end, is_weekend, cyclical sin/cos)
   - HolidayFlags(country="PK") using the optional `holidays` package; skip gracefully if not installed.
   All sort by the time_index role column and are group-aware when a group role exists. Leading NaN rows from lags are handled by dropping (keep y aligned via B1 helpers).
2. Trainer: when task == TIME_SERIES or time_column is set: sort by time, chronological split (last test_size fraction), TimeSeriesSplit(gap=horizon) for CV via cv_runner.
3. Update recipes/timeseries_ready.yaml to use the new transforms (the current recipe references @datetime_month which only exists after DateParts runs).

ACCEPTANCE
Rolling mean at t excludes y_t; chronological split property holds; on a synthetic seasonal series the model beats a seasonal-naive baseline on MASE.
```

## TS2 — Forecast engine: backtest, recursive forecast, intervals

```text
TASK
core/forecast/engine.py (no Qt):
- backtest(df, config, horizon, n_windows=5): walk-forward expanding window; per-window MAE, RMSE, sMAPE, MASE vs naive and seasonal-naive (season length from config or autocorrelation peak).
- Forecaster wrapping the trained estimator + fitted feature pipeline. forecast(history_df, horizon): RECURSIVE multi-step: for each step build lag/rolling features from history + previous predictions, predict, append.
- Prediction intervals: empirical residual quantiles (10/90 default, configurable) from backtest residuals, widened with sqrt(step).
- Store in InferencePipeline: time_column, freq (pd.infer_freq else most common delta), target, default horizon.
- Validation errors: duplicate timestamps, irregular gaps (report count), history shorter than max lag + window.

ACCEPTANCE
On synthetic trend + weekly seasonality: forecast MASE < 1 vs naive; nominal 80% interval covers 70–95%; recursive forecast uses previous predictions (spy); irregular index raises a clear ValueError.
```

## TS3 — Forecast GUI + demo data

```text
TASK
- TrainPage: when TIME_SERIES is selected require a time_index role; add horizon and backtest-windows spinboxes; include them in the summary card.
- EvaluatePage: "Backtest" tab: window table + chart (actual vs predicted per window).
- PredictPage: new "Forecast" PillTab: pick horizon, optional upload of fresh history CSV or use the training tail; ForecastWorker; pyqtgraph plot (history, forecast, shaded interval) + table + "Export CSV".
- scripts/make_demo_data.py generates examples/sales_demo.csv: daily sales with weekly + yearly seasonality, promotions, Eid/holiday bumps. (.gitignore ignores *.csv: add !examples/*.csv, see H2.)

ACCEPTANCE
Offscreen GUI test: bind a forecaster, run forecast, plot has 3 series, CSV export has `horizon` rows.
```

---

# PHASE 4 — Drift monitoring

## D1 — Core drift engine

```text
TASK
core/monitoring/drift.py:
- At train time build reference_stats and store them in the model artifact: numeric -> decile bin edges + counts, mean/std; categorical -> frequency table; missing rate; training prediction distribution.
- compute_drift(reference, new_df) -> DriftReport {per_feature: [{name, psi, ks_pvalue (numeric), chi2_pvalue (categorical), missing_delta, status: ok|warn|alert}], overall, schema_issues (missing/new columns, dtype changes), prediction_shift}.
- Thresholds: PSI < 0.1 ok, 0.1–0.25 warn, > 0.25 alert (configurable). Handle unseen categories, empty buckets (epsilon), < 30 rows => "insufficient data".
- InferencePipeline.reference_stats; ModelRegistry persists it.

ACCEPTANCE
Identical data -> all ok; shifted mean -> alert; new category -> schema issue; < 30 rows -> insufficient.
```

## D2 — Drift tab (replace "Coming soon")

```text
TASK
PredictPage Drift tab: choose a file (CSV/Parquet) -> DriftWorker -> table (feature, PSI, KS/Chi2 p-value, status chip) sorted by severity, overall chip, bar chart of PSI per feature, click a row to overlay training vs new histograms. "Export drift report" (CSV + PDF via the reporting module). When overall == alert show a "Consider retraining" card with a button that opens Train with the same config. Update tests/gui/test_evaluate_predict_polish.py (it currently asserts "Coming soon").
```

---

# PHASE 5 — Export as API (the bridge to the app)

## A1 — Generate a standalone FastAPI service

```text
TASK
core/serving/export_api.py (no Qt): export_service(model_id, registry, out_dir, api_key=None) writes:
  service/app.py (FastAPI), service/model/{pipeline.joblib, metadata.json, schema.json}, requirements.txt (pinned to the current sklearn/pandas/numpy versions), Dockerfile (python:3.11-slim, non-root user), README.md (curl examples), .env.example (API_KEY).
Endpoints exactly as in Appendix B. Request validation built dynamically from schema.json; invalid input -> 422 {"errors":[{"field","message"}]}. Auth: X-API-Key header, constant-time compare; generate secrets.token_urlsafe(24) when none is given. CORS off by default. In-memory rate limit 60 req/min/key. Never log request bodies.
IMPORTANT: joblib unpickles ml_studio.* classes. The exported service must NOT require PyQt6 or the full app. Vendor a minimal ml_studio runtime into the service folder (copy only transforms/, core/pipeline.py, core/schema.py, core/persistence/serializer.py, core/training/task.py, core/evaluation/local_explain.py, app/logger.py, with stub __init__.py files and IDENTICAL import paths so unpickling works).
Local explanations: core/evaluation/local_explain.py top_factors(pipeline, row, k=5): replace each feature with its schema `ref` value and measure the change in predicted value / probability of the predicted class; return signed impacts. No SHAP dependency.
/forecast only for time-series models.
Templates live in core/serving/templates/*.tpl (string.Template), not giant f-strings.

ACCEPTANCE
Test exports a service into tmp_path and imports app.py with fastapi.testclient: /health, /schema, /predict (valid, invalid -> 422, missing key -> 401), batch limit, top_factors sign test. Add a test that runs the service in a venv WITHOUT PyQt6 (mark slow). Add fastapi/uvicorn/httpx/qrcode to a `serve` extra.
```

## A2 — GUI: Export, serve locally, QR for the phone

```text
TASK
ModelsPage detail: "Export as API…" (choose folder) and "Serve locally" (QProcess runs `python -m uvicorn app:app --host <127.0.0.1|0.0.0.0> --port <free port>` inside the exported folder with the current interpreter when fastapi is installed). Show status chip Running/Stopped, a log panel, Stop button. Show the URL (http://<LAN-IP>:<port>) and the API key with Copy buttons, plus a QR code (package `qrcode`, rendered to QPixmap) encoding {"url","key","name"} so the mobile app can scan it. Default bind is 127.0.0.1; checkbox "Allow my phone on this network (0.0.0.0)" with the warning "Anyone on your Wi-Fi with the key can call this model." Kill the process on app exit. Show a Windows firewall hint.

ACCEPTANCE
Unit test for the QR payload; QProcess start/stop lifecycle with a dummy script; GUI test for button enabled states.
```

---

# PHASE 6 — Flutter companion app (separate repo: `ml_studio_mobile`)

Use case it serves: field staff / shop owner opens the app, enters a few values, gets a prediction (price, risk, lead score) or a forecast from the model you trained on desktop. The form is generated from `/schema`, so it works for any model.

## M1 — Scaffold + architecture

```text
Create a Flutter app "ml_studio_mobile" (stable channel, Dart 3, sound null safety, Android + iOS).
Stack: flutter_riverpod (AsyncNotifier), go_router, dio, flutter_secure_storage, mobile_scanner, fl_chart, sqflite, freezed + json_serializable, intl, flutter_lints.
Structure (feature-first): lib/core/{api,theme,router,errors}, lib/features/{connection,predict,history,forecast,settings}.
Deliver:
- Material 3 theme (light/dark via ColorScheme.fromSeed).
- go_router routes: /connect, /predict, /history, /forecast, /settings.
- Typed ApiClient (dio): interceptor adds X-API-Key, 10 s timeouts, errors mapped to a sealed AppFailure {Network, Unauthorized, Validation(fieldErrors), Server}.
- freezed models for the API contract in the plan appendix B.
- README. No UI polish yet. Include one unit test for ApiClient error mapping with a mocked Dio adapter.
```

## M2 — Connection (QR or manual)

```text
Connection feature: add a server by (a) scanning the QR (JSON {url,key,name}) with mobile_scanner or (b) manual URL + key. Validate via GET /health then GET /schema; on success show model name, task, version. Store connections in flutter_secure_storage (never plain prefs); multiple saved servers with an active selection. Android: network_security_config permitting cleartext ONLY for private IP ranges (document why). Errors: unreachable (hint: same Wi-Fi / firewall), 401, timeout. Unit tests: QR payload parsing, repository with mocked Dio.
```

## M3 — Schema-driven form builder

```text
Build SchemaFormBuilder from GET /schema features:
- numeric -> TextFormField (number keyboard, min/max validation, hint shows example/range)
- categorical -> searchable DropdownMenu from `categories` plus "Other/unknown"
- boolean -> Switch
- datetime -> date picker (ISO string)
Required vs nullable (blank => omitted). Group fields in Cards. "Use last values" per server. FormController (Riverpod) with draft state, dirty flag, reset. Map server 422 {"errors":[{"field","message"}]} back onto the matching fields. Widget tests: each kind renders, validation messages, 422 mapping.
```

## M4 — Predict + result screen

```text
Predict flow: submit -> POST /predict {features, explain: true}. Result screen: large prediction (label or number), for classification a horizontal bar chart of class probabilities (fl_chart) + confidence chip (Low < 0.6 / Medium / High); "Top factors" list from top_factors (signed bars, readable names); server warnings (e.g. unseen category). Loading / error / empty states, haptic on success, buttons "Edit inputs" and "Save to history". If the request fails with a network error offer "Queue for later". Widget tests with a mocked ApiClient.
```

## M5 — History, offline queue, batch entry

```text
History: sqflite table predictions(id, server_id, model_name, input_json, output_json, created_at, status[sent|queued|failed], note). List with search/filter (model/date), swipe to delete, detail view with "Duplicate as new", CSV export via share_plus. Queue worker: when connectivity_plus reports online, retry queued items sequentially with exponential backoff; badge shows queued count. Batch entry mode: "Add another" keeps shared fields (e.g. branch) and clears the rest. Tests for repository + queue worker.
```

## M6 — Forecast screen (only when schema.task == TIME_SERIES)

```text
Forecast feature: horizon picker (1–90), optional recent-history CSV via file_picker or use the server default; POST /forecast; fl_chart with history line, forecast line, shaded lo/hi interval, tooltips; table toggle; export CSV/PNG. Show actionable messages for server validation errors (irregular dates, history too short).
```

## M7 — Polish + release

```text
Empty states, skeleton loaders, accessibility (semantics labels, 48 dp targets, large-font support), dark mode check, intl-ready strings, launcher icon + native splash, error reporting hook WITHOUT PII, integration test of the happy path against a local mock server (shelf), GitHub Actions (flutter analyze + flutter test), Android release config with signing via key.properties (not committed), versioning. README with a real-screenshots section and GIF capture instructions.
```

---

# PHASE 7 — Optional AI explainer (keep it small and private)

## E1 — Plain-language explanations

```text
TASK
core/ai/explainer.py: Provider protocol {name, explain(prompt)->str}. OllamaProvider (http://localhost:11434, model configurable) and AnthropicProvider (user-supplied API key, stored with `keyring`, never in project files). PromptBuilder uses ONLY aggregates: task, metrics vs baseline, verdict reasons, top feature names + importance, data-quality issue summaries, drift summary. NEVER raw rows. Output: short plain-language summary + 3 concrete next steps. Cache by hash of the context.
Settings: "AI explanations" OFF by default, provider selector, "Test connection", and a privacy note listing exactly what is sent.
UI: "Explain results" on EvaluatePage; "Explain this prediction" on Predict (sends feature names + signed impacts only unless the user ticks "include values"). Runs in a worker with timeout; handles offline gracefully.

ACCEPTANCE
Tests with a fake provider. A PromptBuilder test inserts a canary string into the DataFrame and asserts it never appears in the prompt.
```

---

# PHASE 8 — Hardening, packaging, release

## H1 — Security

```text
1. Dataset.source can contain DB URLs with passwords (load_dataset_from_url -> Dataset.source -> to_dict -> project meta.json). Add core/security.py mask_url(url) (user:***@host); use it for persisted Dataset.source and logs. Keep the real URI only in memory; ask for the password again on reopen.
2. Unsafe deserialization: session_io falls back to joblib for the dataset; ModelRegistry/InferencePipeline.load use joblib (pickle). Datasets: parquet only (fallback: gzip CSV, not joblib). Model artifacts: write a SHA-256 manifest in metadata.json at save and verify before joblib.load; refuse on mismatch. Opening a .mlstudio or imported model from outside the registry shows "Models are Python pickles — only open files you trust."
3. ProjectManager.open_project extracts to `.<stem>_cache` beside the file and never cleans it: use tempfile.TemporaryDirectory cleaned in finally; explicitly prevent zip-slip (resolve each member path under the target).
4. LocalFileSource sqlite: validate table names against sqlite_master before building SQL.
Tests for each.
```

## H2 — Packaging + CI (do this EARLY)

```text
1. pyproject.toml dependencies miss PyYAML, statsmodels, sqlalchemy and pyarrow, which core imports -> `pip install .` breaks. Add them. Define extras: ml (xgboost, lightgbm, optuna, shap), io (openpyxl, fsspec, s3fs, gcsfs, chardet), serve (fastapi, uvicorn, qrcode), report (reportlab, matplotlib), dev, all. Make pyproject the single source of truth; remove or generate the requirements-*.txt files and document `pip install -e .[all,dev]`.
2. .gitignore currently lists ml_studio/tests/gui/test_main_window_extra.py and test_models_page.py -> those tests are never committed or run in CI. Remove those lines. It also ignores *.json/*.csv/*.txt which blocks fixtures and demo data: add negations !examples/*.csv and !ml_studio/tests/fixtures/**.
3. GitHub Actions: matrix ubuntu + windows, Python 3.10–3.12, QT_QPA_PLATFORM=offscreen (xvfb on Linux), install .[all,dev], run scripts/check_rules.py then pytest --cov. pyproject has fail_under = 85: measure the real coverage first and set an honest threshold.
4. Add pre-commit (ruff, ruff-format, check_rules).
5. scripts/check_rules.py only scans SLICE1_FILES. Make it scan everything under ml_studio/ except tests for hex colors and size limits; fix or whitelist the violations.
```

## H3 — Release + README

```text
Windows build with PyInstaller (--onedir; include assets/ and recipes/; hidden imports for sklearn, scipy, pandas, pyarrow). Spec in packaging/, version from ml_studio.__version__, app icon. GitHub Actions release workflow on tags: zip + SHA-256. Add `--selftest` to main.py: build MainWindow offscreen, load examples/sales_demo.csv, train a quick model, exit 0 (used as the CI smoke test).
README rewrite: problem statement, 60-second GIF, a screenshots table (Data, Prepare, Train, Evaluate with charts, Predict/Forecast, Drift) using REAL screenshots, install, quickstart with the examples dataset, architecture diagram, an honest limitations section, link to ROADMAP.md. Rename REFACTOR_PLAN.md -> ROADMAP.md and make the statuses match reality.
```

## H4 — Cleanup (no behavior change)

```text
Delete assets/styles.qss (unused legacy). Remove the duplicated THEMES_DIR (app/paths.py vs app/theme.py). Strip UTF-8 BOMs from step_config.py, transform_picker.py, preview_modal.py, schema_editor.py and the test files that have them. profiling.py: ProfileCache and the use_cache parameter are unused — wire the cache with dataset.cache_key or delete it; profile_dataset AND ProfilingWorker both call detect_quality_issues — call it once. Replace the __import__("PyQt6.QtGui", ...) hacks in pages with normal imports. Add type hints and docstrings on public APIs. All tests stay green.
```

---

# Appendix A — `.cursorrules`

```text
Project: Machine Learning Studio (PyQt6 + scikit-learn), Python 3.10+.

Architecture
- gui/ and app/ may import core/. core/ must NEVER import PyQt6. transforms/ must not import gui/.
- Heavy work (> ~100 ms) runs in gui/workers/*, never on the UI thread.
- Fitted state is learned from TRAINING data only. No leakage. Models/pipelines persist via joblib in artifacts; project files store config only.

Rules (enforced by scripts/check_rules.py)
- Files <= 400 lines, functions <= 40 lines, classes <= 300 lines.
- No TODO/FIXME/placeholder code.
- No hex colors outside app/theme_tokens.py and assets/themes/*.qss. Use color_token() or QSS objectNames.

Workflow
- Every bug fix ships with a failing-first regression test in ml_studio/tests/.
- After changes run: pytest ml_studio/tests -q -x  and  python scripts/check_rules.py
- Keep diffs small, one concern per change, do not refactor unrelated code.
- Never claim a task is done while tests fail.
- Only touch files named in the task unless you explain why first.
```

# Appendix B — API contract (desktop export ⇄ Flutter app)

Auth: header `X-API-Key: <key>` on everything except `/health`.

```text
GET /health
→ {"status":"ok","model":{"id":"...","name":"Random Forest","task":"CLASSIFICATION","version":1}}

GET /schema
→ {
    "task": "CLASSIFICATION|REGRESSION|TIME_SERIES",
    "target": "churn",
    "target_classes": ["no","yes"],
    "features": [
      {"name":"age","kind":"numeric","required":true,"nullable":false,"min":18,"max":90,"example":35},
      {"name":"city","kind":"categorical","required":true,"nullable":true,"categories":["Lahore","Karachi"]},
      {"name":"is_member","kind":"boolean","required":false,"nullable":true},
      {"name":"signup","kind":"datetime","required":false,"nullable":true}
    ],
    "time": {"column":"date","freq":"D","default_horizon":14}   // TIME_SERIES only
  }

POST /predict
← {"features":{"age":31,"city":"Lahore"},"explain":true}
→ {"prediction":"yes",
   "probabilities":{"no":0.22,"yes":0.78},        // classification only
   "top_factors":[{"feature":"age","impact":0.12},{"feature":"city","impact":-0.05}],
   "warnings":["Unseen category for 'city': 'Multan'"]}

POST /predict/batch        // max 1000 rows
← {"rows":[{...},{...}]}
→ {"predictions":[{"prediction":"yes","probabilities":{...}}, ...]}

POST /forecast             // TIME_SERIES only
← {"horizon":14,"history":[{"date":"2026-09-01","sales":120}, ...]}   // history optional
→ {"points":[{"ts":"2026-10-07","yhat":132.4,"lo":118.0,"hi":149.2}, ...]}

Errors
401 {"detail":"invalid api key"}
422 {"errors":[{"field":"age","message":"must be a number"}]}
```

# Appendix C — What NOT to build (yet)

- Neural nets / deep learning tab, text/image models: different product, huge scope.
- Cloud sync, accounts, multi-user: contradicts "local & private".
- A mobile "training progress viewer": looks cool, nobody needs it.
- On-device ONNX in Flutter: only after A1 works, and only for models without custom transforms.

# Appendix D — Demo that sells the whole thing

1. Dataset: a public used-car or house-price dataset (check the licence), or `examples/sales_demo.csv` for forecasting.
2. In Studio: import → set target → train → show verdict + charts → export PDF report.
3. "Export as API" → scan QR with the Flutter app → fill the generated form → get prediction + top factors.
4. Record a 60-second screen capture of exactly this flow. Put it at the top of both READMEs.
