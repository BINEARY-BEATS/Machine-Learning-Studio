"""DBSCAN / unsupervised training and metrics acceptance tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import DBSCAN, KMeans
from sklearn.datasets import make_blobs
from sklearn.ensemble import IsolationForest

from ml_studio.core.evaluation.metrics import compute_metrics
from ml_studio.core.persistence.serializer import InferencePipeline
from ml_studio.core.training.task import TaskType
from ml_studio.core.training.trainer import Trainer, TrainingConfig
from ml_studio.core.training.unsupervised import fit_predict_labels, predict_new


@pytest.fixture
def blobs_df():
    X, _ = make_blobs(n_samples=120, centers=3, cluster_std=0.6, random_state=0)
    return pd.DataFrame(X, columns=["x", "y"])


def _cluster_cfg(model_id: str, **hyper) -> TrainingConfig:
    return TrainingConfig(
        task=TaskType.CLUSTERING,
        model_id=model_id,
        target_column="",
        feature_columns=["x", "y"],
        test_size=0.2,
        cv_splits=0,
        hyperparameters=hyper,
    )


def test_dbscan_trains_without_error(blobs_df):
    result = Trainer().train(blobs_df, _cluster_cfg("dbscan", eps=0.8, min_samples=5))
    assert "noise_ratio" in result.metrics
    assert result.metrics["n_clusters"] >= 1
    assert "cluster_profile" in result.metrics
    assert isinstance(result.metrics["cluster_profile"], list)


def test_predict_new_dbscan_far_point_is_noise(blobs_df):
    model = DBSCAN(eps=0.8, min_samples=5)
    fit_predict_labels(model, blobs_df)
    far = pd.DataFrame([[100.0, 100.0]], columns=["x", "y"])
    labels = predict_new(model, far)
    assert labels[0] == -1


def test_predict_new_dbscan_near_core(blobs_df):
    model = DBSCAN(eps=0.8, min_samples=5)
    labels_fit = fit_predict_labels(model, blobs_df)
    core_mask = labels_fit >= 0
    assert core_mask.any()
    near = blobs_df.iloc[[int(np.flatnonzero(core_mask)[0])]]
    labels = predict_new(model, near)
    assert labels[0] >= 0


def test_kmeans_results(blobs_df):
    result = Trainer().train(blobs_df, _cluster_cfg("kmeans", n_clusters=3))
    assert result.metrics["n_clusters"] == 3
    assert "silhouette" in result.metrics
    assert result.metrics.get("noise_ratio", 0.0) == 0.0


def test_gmm_results(blobs_df):
    result = Trainer().train(blobs_df, _cluster_cfg("gmm", n_components=3))
    assert result.metrics["n_clusters"] == 3
    assert "silhouette" in result.metrics


def test_silhouette_excludes_noise():
    X = pd.DataFrame(
        {"a": [0.0, 0.1, 5.0, 5.1, 50.0], "b": [0.0, 0.1, 5.0, 5.1, 50.0]}
    )
    labels = np.array([0, 0, 1, 1, -1])
    m = compute_metrics(TaskType.CLUSTERING, y_pred=labels, X=X)
    assert m["noise_ratio"] == pytest.approx(0.2)
    assert "silhouette" in m
    assert "davies_bouldin" in m
    m2 = compute_metrics(
        TaskType.CLUSTERING, y_pred=np.array([-1, -1, 0, 0]), X=X.iloc[:4]
    )
    assert m2["n_clusters"] == 1
    assert "silhouette" not in m2


def test_isolation_forest_anomaly_scores(blobs_df):
    cfg = TrainingConfig(
        task=TaskType.ANOMALY_DETECTION,
        model_id="isolation_forest",
        target_column="",
        feature_columns=["x", "y"],
        hyperparameters={"contamination": 0.1},
    )
    result = Trainer().train(blobs_df, cfg)
    assert result.metrics["anomaly_count"] > 0
    assert "score_mean" in result.metrics
    assert "score_min" in result.metrics


def test_lof_anomaly_count(blobs_df):
    cfg = TrainingConfig(
        task=TaskType.ANOMALY_DETECTION,
        model_id="lof",
        target_column="",
        feature_columns=["x", "y"],
        hyperparameters={"n_neighbors": 10, "contamination": 0.1},
    )
    result = Trainer().train(blobs_df, cfg)
    assert result.metrics["anomaly_count"] > 0


def test_inference_pipeline_predict_dbscan(blobs_df):
    model = DBSCAN(eps=0.8, min_samples=5)
    fit_predict_labels(model, blobs_df)
    pipe = InferencePipeline(
        estimator=model,
        preprocessing=None,
        feature_columns=["x", "y"],
        target_column="",
        task=TaskType.CLUSTERING,
        feature_schema={},
    )
    far = pd.DataFrame([[999.0, 999.0]], columns=["x", "y"])
    preds = pipe.predict(far)
    assert int(np.asarray(preds).ravel()[0]) == -1


def test_fit_predict_labels_helpers():
    X = np.array([[0, 0], [0.1, 0.1], [5, 5], [5.1, 5.1]])
    km = KMeans(n_clusters=2, n_init=10, random_state=0)
    labels = fit_predict_labels(km, X)
    assert set(labels.tolist()) == {0, 1}
    assert set(predict_new(km, X).tolist()) == {0, 1}

    iso = IsolationForest(contamination=0.25, random_state=0)
    lab = fit_predict_labels(iso, X)
    assert set(lab.tolist()) <= {-1, 1}
