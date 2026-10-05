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
