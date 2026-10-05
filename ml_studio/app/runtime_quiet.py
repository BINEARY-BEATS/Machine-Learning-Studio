"""Silence third-party ML library console noise for the desktop app."""

from __future__ import annotations

import logging
import os
import warnings


def silence_ml_console_noise() -> None:
    """Call once at process start before importing heavy ML stacks."""
    # Prefer fewer loky worker leaks on Windows when the GUI cancels mid-fit
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", "2")
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")
    # LightGBM / XGBoost chatty defaults
    os.environ.setdefault("LIGHTGBM_EXEC", "")

    warnings.filterwarnings("ignore", category=UserWarning, module=r"joblib\..*")
    warnings.filterwarnings("ignore", message=r".*resource_tracker.*")
    warnings.filterwarnings("ignore", message=r".*ill-conditioned matrix.*")
    warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"sklearn\..*")
    try:
        from sklearn.exceptions import ConvergenceWarning

        warnings.filterwarnings("ignore", category=ConvergenceWarning)
    except Exception:
        pass
    try:
        from numpy.linalg import LinAlgWarning

        warnings.filterwarnings("ignore", category=LinAlgWarning)
    except Exception:
        pass

    for name in (
        "lightgbm",
        "xgboost",
        "joblib",
        "loky",
        "sklearn",
    ):
        logging.getLogger(name).setLevel(logging.ERROR)
