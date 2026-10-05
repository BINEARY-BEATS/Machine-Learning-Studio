"""Interruptible sklearn fits — kill worker process on cancel (Windows-friendly)."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, TimeoutError as FuturesTimeout
from typing import Any, Callable


def _fit_score_job(
    model_id: str,
    params: dict[str, Any],
    X_tr,
    y_tr,
    X_te,
    y_te,
    scoring: str,
) -> float:
    """Top-level for Windows process pickling."""
    from sklearn.metrics import get_scorer

    from ml_studio.core.training.registry import get_model

    est = get_model(model_id, **params)
    est.fit(X_tr, y_tr)
    return float(get_scorer(scoring)(est, X_te, y_te))


def fit_score_cancellable(
    model_id: str,
    params: dict[str, Any],
    X_tr,
    y_tr,
    X_te,
    y_te,
    scoring: str,
    cancel_check: Callable[[], bool],
    poll_s: float = 0.12,
) -> float:
    """
    Fit+score in a child process so Cancel can terminate mid-fit.
    Falls back to in-process fit if process pool fails to start.
    """
    try:
        X_tr_v = X_tr.to_numpy(copy=False) if hasattr(X_tr, "to_numpy") else X_tr
        y_tr_v = y_tr.to_numpy(copy=False) if hasattr(y_tr, "to_numpy") else y_tr
        X_te_v = X_te.to_numpy(copy=False) if hasattr(X_te, "to_numpy") else X_te
        y_te_v = y_te.to_numpy(copy=False) if hasattr(y_te, "to_numpy") else y_te
    except Exception:
        X_tr_v, y_tr_v, X_te_v, y_te_v = X_tr, y_tr, X_te, y_te

    pool: ProcessPoolExecutor | None = None
    try:
        pool = ProcessPoolExecutor(max_workers=1)
        fut = pool.submit(
            _fit_score_job,
            model_id,
            params,
            X_tr_v,
            y_tr_v,
            X_te_v,
            y_te_v,
            scoring,
        )
        while True:
            if cancel_check():
                _kill_pool(pool)
                pool = None
                raise InterruptedError("cancelled")
            try:
                result = fut.result(timeout=poll_s)
                _shutdown_pool(pool, wait=False)
                pool = None
                return result
            except FuturesTimeout:
                continue
    except InterruptedError:
        raise
    except Exception:
        if cancel_check():
            raise InterruptedError("cancelled")
        from sklearn.metrics import get_scorer

        from ml_studio.core.training.registry import get_model

        est = get_model(model_id, **params)
        est.fit(X_tr, y_tr)
        if cancel_check():
            raise InterruptedError("cancelled")
        return float(get_scorer(scoring)(est, X_te, y_te))
    finally:
        if pool is not None:
            _shutdown_pool(pool, wait=False)


def _kill_pool(pool: ProcessPoolExecutor) -> None:
    procs = getattr(pool, "_processes", None) or {}
    for proc in list(procs.values()):
        try:
            proc.terminate()
        except Exception:
            pass
    _shutdown_pool(pool, wait=False)


def _shutdown_pool(pool: ProcessPoolExecutor, *, wait: bool) -> None:
    try:
        pool.shutdown(wait=wait, cancel_futures=True)
    except TypeError:
        pool.shutdown(wait=wait)
    except Exception:
        pass
