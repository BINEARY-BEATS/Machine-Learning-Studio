"""Model registry with metadata for all supported estimators."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from sklearn.base import BaseEstimator
from sklearn.cluster import DBSCAN, KMeans
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    IsolationForest,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
)
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor, LocalOutlierFactor
from sklearn.svm import SVC, SVR

from ml_studio.core.training.task import TaskType


@dataclass
class ModelMetadata:
    name: str
    task_types: list[TaskType]
    factory: Callable[..., BaseEstimator]
    supported_features: list[str] = field(default_factory=lambda: ["numeric", "categorical"])
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    default_params: dict[str, Any] = field(default_factory=dict)
    pros: list[str] = field(default_factory=list)
    cons: list[str] = field(default_factory=list)
    doc_url: str = ""
    cost: str = "medium"


def _optional_xgb_classifier(**kw):
    import xgboost as xgb
    return xgb.XGBClassifier(**kw)


def _optional_xgb_regressor(**kw):
    import xgboost as xgb
    return xgb.XGBRegressor(**kw)


def _optional_lgbm_classifier(**kw):
    import lightgbm as lgb
    return lgb.LGBMClassifier(**kw)


def _optional_lgbm_regressor(**kw):
    import lightgbm as lgb
    return lgb.LGBMRegressor(**kw)


MODEL_REGISTRY: dict[str, ModelMetadata] = {
    "logistic_regression": ModelMetadata(
        name="Logistic Regression",
        task_types=[TaskType.CLASSIFICATION],
        factory=lambda **kw: LogisticRegression(max_iter=1000, **kw),
        default_params={"C": 1.0},
        hyperparameters={"C": [0.01, 0.1, 1.0, 10.0]},
        pros=["Interpretable", "Fast"],
        cons=["Linear decision boundary"],
        doc_url="https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression",
        cost="low",
    ),
    "random_forest_classifier": ModelMetadata(
        name="Random Forest",
        task_types=[TaskType.CLASSIFICATION],
        factory=lambda **kw: RandomForestClassifier(random_state=42, n_jobs=-1, **kw),
        default_params={"n_estimators": 100},
        hyperparameters={"n_estimators": [50, 100, 200], "max_depth": [None, 5, 10]},
        pros=["Robust", "Feature importance"],
        cons=["Less interpretable"],
        cost="medium",
    ),
    "extra_trees_classifier": ModelMetadata(
        name="Extra Trees",
        task_types=[TaskType.CLASSIFICATION],
        factory=lambda **kw: ExtraTreesClassifier(random_state=42, n_jobs=-1, **kw),
        default_params={"n_estimators": 100},
        cost="medium",
    ),
    "hist_gb_classifier": ModelMetadata(
        name="HistGradientBoosting",
        task_types=[TaskType.CLASSIFICATION],
        factory=lambda **kw: HistGradientBoostingClassifier(random_state=42, **kw),
        default_params={"max_iter": 100},
        cost="medium",
    ),
    "knn_classifier": ModelMetadata(
        name="K-Nearest Neighbors",
        task_types=[TaskType.CLASSIFICATION],
        factory=lambda **kw: KNeighborsClassifier(**kw),
        default_params={"n_neighbors": 5},
        cost="low",
    ),
    "svc": ModelMetadata(
        name="Support Vector Machine",
        task_types=[TaskType.CLASSIFICATION],
        factory=lambda **kw: SVC(probability=True, **kw),
        default_params={"C": 1.0, "kernel": "rbf"},
        cost="high",
    ),
    "linear_regression": ModelMetadata(
        name="Linear Regression",
        task_types=[TaskType.REGRESSION],
        factory=lambda **kw: LinearRegression(**kw),
        pros=["Interpretable", "Fast baseline"],
        cost="low",
    ),
    "ridge": ModelMetadata(
        name="Ridge",
        task_types=[TaskType.REGRESSION],
        factory=lambda **kw: Ridge(**kw),
        default_params={"alpha": 1.0},
        hyperparameters={"alpha": [0.1, 1.0, 10.0]},
        cost="low",
    ),
    "lasso": ModelMetadata(
        name="Lasso",
        task_types=[TaskType.REGRESSION],
        factory=lambda **kw: Lasso(max_iter=5000, **kw),
        default_params={"alpha": 1.0},
        cost="low",
    ),
    "elasticnet": ModelMetadata(
        name="ElasticNet",
        task_types=[TaskType.REGRESSION],
        factory=lambda **kw: ElasticNet(max_iter=5000, **kw),
        default_params={"alpha": 1.0, "l1_ratio": 0.5},
        cost="low",
    ),
    "random_forest_regressor": ModelMetadata(
        name="Random Forest Regressor",
        task_types=[TaskType.REGRESSION],
        factory=lambda **kw: RandomForestRegressor(random_state=42, n_jobs=-1, **kw),
        default_params={"n_estimators": 100},
        cost="medium",
    ),
    "extra_trees_regressor": ModelMetadata(
        name="Extra Trees Regressor",
        task_types=[TaskType.REGRESSION],
        factory=lambda **kw: ExtraTreesRegressor(random_state=42, n_jobs=-1, **kw),
        cost="medium",
    ),
    "hist_gb_regressor": ModelMetadata(
        name="HistGradientBoosting Regressor",
        task_types=[TaskType.REGRESSION],
        factory=lambda **kw: HistGradientBoostingRegressor(random_state=42, **kw),
        cost="medium",
    ),
    "knn_regressor": ModelMetadata(
        name="KNN Regressor",
        task_types=[TaskType.REGRESSION],
        factory=lambda **kw: KNeighborsRegressor(**kw),
        cost="low",
    ),
    "svr": ModelMetadata(
        name="SVR",
        task_types=[TaskType.REGRESSION],
        factory=lambda **kw: SVR(**kw),
        cost="high",
    ),
    "kmeans": ModelMetadata(
        name="KMeans",
        task_types=[TaskType.CLUSTERING],
        factory=lambda **kw: KMeans(random_state=42, n_init=10, **kw),
        default_params={"n_clusters": 3},
        cost="medium",
    ),
    "dbscan": ModelMetadata(
        name="DBSCAN",
        task_types=[TaskType.CLUSTERING],
        factory=lambda **kw: DBSCAN(**kw),
        default_params={"eps": 0.5, "min_samples": 5},
        cost="medium",
    ),
    "gmm": ModelMetadata(
        name="Gaussian Mixture",
        task_types=[TaskType.CLUSTERING],
        factory=lambda **kw: GaussianMixture(random_state=42, **kw),
        default_params={"n_components": 3},
        cost="medium",
    ),
    "isolation_forest": ModelMetadata(
        name="Isolation Forest",
        task_types=[TaskType.ANOMALY_DETECTION],
        factory=lambda **kw: IsolationForest(random_state=42, **kw),
        default_params={"contamination": 0.05},
        cost="medium",
    ),
    "lof": ModelMetadata(
        name="Local Outlier Factor",
        task_types=[TaskType.ANOMALY_DETECTION],
        factory=lambda **kw: LocalOutlierFactor(novelty=True, **kw),
        cost="medium",
    ),
}


def _register_optional() -> None:
    try:
        MODEL_REGISTRY["xgboost_classifier"] = ModelMetadata(
            name="XGBoost Classifier",
            task_types=[TaskType.CLASSIFICATION],
            factory=_optional_xgb_classifier,
            default_params={"n_estimators": 100},
            cost="medium",
        )
        MODEL_REGISTRY["xgboost_regressor"] = ModelMetadata(
            name="XGBoost Regressor",
            task_types=[TaskType.REGRESSION],
            factory=_optional_xgb_regressor,
            default_params={"n_estimators": 100},
            cost="medium",
        )
    except ImportError:
        pass
    try:
        MODEL_REGISTRY["lightgbm_classifier"] = ModelMetadata(
            name="LightGBM Classifier",
            task_types=[TaskType.CLASSIFICATION],
            factory=_optional_lgbm_classifier,
            cost="medium",
        )
        MODEL_REGISTRY["lightgbm_regressor"] = ModelMetadata(
            name="LightGBM Regressor",
            task_types=[TaskType.REGRESSION],
            factory=_optional_lgbm_regressor,
            cost="medium",
        )
    except ImportError:
        pass


_register_optional()


def get_models_for_task(task: TaskType) -> list[ModelMetadata]:
    return [m for m in MODEL_REGISTRY.values() if task in m.task_types]


def get_model(model_id: str, **params) -> BaseEstimator:
    if model_id not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model: {model_id}")
    meta = MODEL_REGISTRY[model_id]
    merged = {**meta.default_params, **params}
    return meta.factory(**merged)
