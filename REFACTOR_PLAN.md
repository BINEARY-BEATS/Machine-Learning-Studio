# Machine Learning Studio — Product Upgrade Plan

> Living plan to make the app **stable, responsive, and useful**.
> Replaces the migration checklist (Phases 1–16). That rewrite landed the package
> layout; it did **not** finish a shippable product.

**Status:** Active — Sprint 5 (U5–U7) complete  
**Goal:** One reliable happy path — Import → Prepare → Train → Evaluate → Predict — with no freezes, no dead buttons, and projects that actually reopen.

### Sprint 1 progress (U0+U1)

| Item | Status |
|------|--------|
| U0.1 Predict API (`predict_single` / `predict_batch`) | Done |
| U0.2 Prepare uses session dataset (not phantom Project APIs) | Done |
| U0.3 Recipes `.yaml` + correct `apply_recipe(schema_columns)` | Done |
| U0.4 Toast `error`→`danger`; step enable toggle wired | Done |
| U1.1 `on_dataset_loaded` → `prepare.set_dataset` | Done |
| U1.4 Pipeline enable checkbox affects training | Done |

### Sprint 2 progress (U2 — kill UI freezes)

| Item | Status |
|------|--------|
| U2.1 ProfilingWorker + auto-profile after import | Done |
| U2.2 `prepare_for_training` inside TrainingWorker | Done |
| U2.3 OptimizeWorker (async optimize memory) | Done |
| U2.4 BatchPredictWorker (chunked, cancellable) | Done |
| U2.5 PreviewWorker + async PreviewModal | Done |
| Cancel path shared via `cancel_active_worker` | Done |

### Sprint 3 progress (U3 — real project persistence)

| Item | Status |
|------|--------|
| U3.1 Save dataset + schema + pipeline into `.mlstudio` | Done |
| U3.2 Open rehydrates controller + Data/Prepare/Train | Done |
| U3.3 Dirty flag on import/prepare changes | Done |
| U3.4 Autosave when path exists + Settings enabled | Done |

### Sprint 4 progress (U4 — train honesty)

| Item | Status |
|------|--------|
| U4.1 Wizard step validation before Next | Done |
| U4.2 Tune → Optuna / Grid wired into Trainer | Done |
| U4.3 Removed dead Eval/Save wizard steps | Done |
| U4.4 Quality-gate failures show clear reason | Done |
| U4.5 Command palette “Tune with Optuna…” (honest) | Done |

### Sprint 5 progress (U5/U6 — evaluate/predict + UX)

| Item | Status |
|------|--------|
| U5.1 `show_metrics` implemented + Train CTA | Done |
| U5.2 Explain tab: permutation importance | Done |
| U5.3 Drift tab labeled Coming soon | Done |
| U6 Soft nav toast warnings | Done |
| U6 Working breadcrumbs | Done |
| U6 Theme on Prepare/Models/Predict + PillTabs icons | Done |
| U6 Quality Review → Prepare; empty-state CTAs | Done |

### Sprint 6 progress (U7 — hardening & docs)

| Item | Status |
|------|--------|
| U7.1 Happy-path GUI/core smoke test | Done |
| U7.2 Honest README | Done |
| U7.3 CLI `data head` / `data info` | Done |
| U7.4 `api.prepare` / `api.sweep` → NotImplementedError | Done |
| U7.5 CustomPython hidden from picker | Done |
| U7.6 Time series honest task label | Done |

---

## Honest baseline

| Claimed in old plan / README | Reality today |
|------------------------------|---------------|
| Project system complete | `.mlstudio` saves metadata only; open does not restore dataset/pipeline |
| Prepare pipeline complete | UI exists but is not wired to imported data; recipes broken |
| Training + Optuna / AutoML | Train wizard runs basic fit; Tune/AutoML/Eval/Save steps are stubs |
| Explainability in product | Core helpers exist; Predict Explain/Drift tabs are placeholders |
| Autosave | Settings checkbox + config value; no timer |
| Responsive GUI | Profiling, prep, optimize, batch predict run on UI thread |
| Migration “complete” | Architecture yes; product quality no |

**Rule going forward:** a phase is Complete only when (1) tests pass, (2) the GUI path works by hand, (3) no UI freeze on a ~100k-row CSV, (4) this doc is updated.

---

## Product principle

Ship **one vertical slice** that works end-to-end before adding features.

```
Import (preview) → Data (async profile) → Prepare (wired pipeline)
  → Train (real config) → Evaluate → Models → Predict (single + batch)
  → Save/Open project restores all of the above
```

Anything outside that slice is P2+ until the slice is green.

---

## Phase overview

| Phase | Name | Outcome | Priority |
|-------|------|---------|----------|
| **U0** | Unblock critical crashes | Predict works; Prepare doesn’t lie; recipes apply | P0 |
| **U1** | Wire the happy path | Dataset flows into Prepare + Train; pages stay in sync | P0 |
| **U2** | Kill UI freezes | Heavy work always on workers; cancel works | P0/P1 |
| **U3** | Real project persistence | Save/Open restores dataset + pipeline + last model refs | P0 |
| **U4** | Train that matches marketing | Tune (Optuna), sensible wizard, quality feedback | P1 |
| **U5** | Evaluate / Predict polish | Metrics detail, explain tab, batch async | P1 |
| **U6** | UX consistency | Toasts, theme, empty states, nav gating, command palette honesty | P1 |
| **U7** | Hardening & docs | Tests for GUI flows, README truth, CLI parity later | P2 |

Do **U0 → U3** before any new ML features (AutoML UI, custom Python, TS models, remote SQL polish).

---

## U0 — Unblock critical crashes (P0)

**Exit criteria:** User can import a CSV, add one transform, train a model, and get a prediction without exceptions.

| # | Work item | Files | Done when |
|---|-----------|-------|-----------|
| U0.1 | Fix Predict API — call `predict_single` / `predict_batch`, not missing `predict()` | `gui/pages/predict_page.py`, `core/inference/predictor.py` | Single + batch return results |
| U0.2 | Stop Prepare from calling phantom `Project` APIs (`dataset`, `save()`, `schema_overrides`) | `gui/pages/prepare_page.py`, `core/project.py` | Page uses `AppController` / session state |
| U0.3 | Fix recipes — load `*.yaml`; call `apply_recipe(name, schema_columns)` correctly | `gui/pages/prepare_page.py`, `core/recipes.py` | Choosing a recipe adds real steps |
| U0.4 | Toast variants — use `danger` (not `error`) everywhere | `gui/widgets/toast.py`, `main_window.py`, `prepare_page.py` | Errors look like errors |
| U0.5 | Smoke test script or pytest covering import→train→predict on tiny CSV | `ml_studio/tests/gui/` | CI catches regressions |

**Estimate:** 2–4 focused days.

---

## U1 — Wire the happy path (P0)

**Exit criteria:** After import, Prepare shows schema; Preview works; Train sees columns; pipeline is used at train time (already partially true).

| # | Work item | Done when |
|---|-----------|-----------|
| U1.1 | `AppController.on_dataset_loaded` hydrates Prepare (schema + empty or restored pipeline) | SchemaEditor populated after import |
| U1.2 | Define a single **session model** (dataset, schema roles, pipeline) owned by controller — pages are views | No page holds the only copy of truth |
| U1.3 | Role changes update controller schema and train suggestions | Target/feature roles affect Train |
| U1.4 | Pipeline enable checkbox actually disables steps | Toggle changes what trains |
| U1.5 | Home / status bar refresh on navigate and after each major action | Stats not stale |

**Estimate:** 3–5 days.

---

## U2 — Kill UI freezes (P0/P1)

**Exit criteria:** 100k-row CSV: import, profile, optimize, train prep, batch predict — UI stays interactive; cancel stops workers.

| # | Work item | Notes |
|---|-----------|-------|
| U2.1 | Use `ProfilingWorker` from Data page | Replace sync `profile_dataset` on UI thread |
| U2.2 | Move `prepare_for_training` into `TrainingWorker` | Progress text for “preparing data…” |
| U2.3 | Move `optimize_dtypes` to a worker | Toast on complete |
| U2.4 | Batch predict via chunked `predict_batch` + worker | Progress bar |
| U2.5 | Pipeline preview on worker (sample still OK) | Modal shows spinner, not freeze |
| U2.6 | Ensure only one heavy worker at a time + cancel path | Already partly there via `AppController` |

**Estimate:** 3–5 days.

---

## U3 — Real project persistence (P0)

**Exit criteria:** Save → quit → Open restores dataset, schema roles, prepare pipeline, and points Models/Predict at last registered models.

| # | Work item | Notes |
|---|-----------|-------|
| U3.1 | Write into `.mlstudio` zip: dataset (parquet/joblib), `pipeline.json`, `schema.json`, settings | Stop writing empty stub folders only |
| U3.2 | `open_project` rehydrates `AppController` + all pages | Same as fresh import, from archive |
| U3.3 | Dirty flag when dataset/pipeline/roles change; Save enabled | Top bar / Ctrl+S |
| U3.4 | Decide fate of dual `api.Project` vs `core.Project` | Short term: document; medium: one persistence layer |
| U3.5 | Autosave timer honoring Settings + `autosave_interval_ms` | Only after U3.1 works |

**Estimate:** 5–8 days (highest leverage for “feels like real software”).

---

## U4 — Train that matches marketing (P1)

**Exit criteria:** Wizard only advances with valid config; Optuna path runs when selected; user lands on Evaluate with clear metrics.

| # | Work item |
|---|-----------|
| U4.1 | Validate each wizard step before Next (task, target, features, split, model) |
| U4.2 | Wire Tune step → `TrainingConfig` + `OptunaTuner` (or hide Optuna until wired) |
| U4.3 | Collapse or remove dead Eval/Save wizard steps — Evaluate/Models pages already own that |
| U4.4 | Surface quality-gate failures clearly (today silent `False` from controller) |
| U4.5 | Command palette `run_automl`: either run `AutoMLRunner` or remove the fake command |

**Estimate:** 4–6 days.

---

## U5 — Evaluate / Predict polish (P1)

| # | Work item |
|---|-----------|
| U5.1 | Implement `EvaluatePage.show_metrics` / richer run detail |
| U5.2 | Predict Explain tab: permutation importance or single-row explain (core already exists) |
| U5.3 | Hide or label Drift as “Coming soon” — no fake empty tab |
| U5.4 | Models theme refresh + Predict rebinding without duplicate signal connects |

**Estimate:** 3–5 days.

---

## U6 — UX consistency (P1)

| # | Work item |
|---|-----------|
| U6.1 | Soft nav gating: warn on Train/Predict without dataset/model (optional disable) |
| U6.2 | Implement or remove `TopBar.set_breadcrumb` |
| U6.3 | Prepare + Models `set_theme_mode`; fix PillTabs icon rebuild bug |
| U6.4 | Pipeline step icons (replace `"?"` placeholders) |
| U6.5 | Quality Issues “Review” buttons either navigate to Prepare or are removed |
| U6.6 | Empty states with one clear primary CTA each |

**Estimate:** 3–4 days.

---

## U7 — Hardening & docs (P2)

| # | Work item | Status |
|---|-----------|--------|
| U7.1 | GUI integration tests for U0–U3 happy path | Done (`test_happy_path_smoke.py`) |
| U7.2 | README: honest feature list, correct test badge, remove overclaims | Done |
| U7.3 | CLI `data head` / `data info` | Done |
| U7.4 | Stub `api.prepare` / `api.sweep` raise `NotImplementedError` | Done |
| U7.5 | CustomPython transform — hidden from picker | Done |
| U7.6 | Time series: honest task picker label (regression models + time CV) | Done |

---

## Explicitly out of scope (until U0–U3 green)

- New transform categories / balancing UI polish
- Remote SQL production hardening
- ONNX export marketing push
- Full AutoML product surface
- Web/cloud version
- Rewriting theme system from scratch

---

## Suggested execution order (sprints)

| Sprint | Focus | Ship check |
|--------|-------|------------|
| **1** | U0 + U1 | Import → Prepare preview → Train → Predict works |
| **2** | U2 | 100k-row CSV does not freeze |
| **3** | U3 | Save/Open round-trip |
| **4** | U4 + U5 | Optuna or honest UI; explain tab |
| **5** | U6 + U7 | Polish + docs + tests |

---

## Definition of Done (product)

A stranger can:

1. Create a project and import a CSV  
2. See profiling without the app locking up  
3. Set a target role and build a short prep pipeline with preview  
4. Train a classification or regression model  
5. See metrics on Evaluate and run single + batch predict  
6. Save, quit, reopen — and continue  

When that passes, this plan’s core is Complete. Everything else is enhancement.

---

## How we use this doc

1. Pick the next unchecked U-item (never skip P0 for shiny P2).  
2. Implement + test + manual GUI check.  
3. Mark the row Done with date.  
4. Update the Sprint table status at the top when a phase exits.

**Next action:** start **U0.1** (Predict API) and **U0.2/U0.3** (Prepare wiring + recipes) in the same sprint.
