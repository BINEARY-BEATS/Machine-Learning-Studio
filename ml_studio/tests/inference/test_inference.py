"""Inference tests."""

import numpy as np
import pandas as pd

from ml_studio.core.inference.predictor import Predictor
from ml_studio.core.persistence.serializer import InferencePipeline
from ml_studio.core.training.task import TaskType
from ml_studio.core.training.registry import get_model


def test_batch_predict_csv(tmp_path):
    rng = np.random.RandomState(42)
    train = pd.DataFrame({"a": rng.randn(50), "b": rng.randn(50), "target": rng.randn(50)})
    model = get_model("ridge")
    model.fit(train[["a", "b"]], train["target"])
    pipe = InferencePipeline(model, None, ["a", "b"], "target", TaskType.REGRESSION)
    predictor = Predictor(pipe)

    batch = tmp_path / "batch.csv"
    pd.DataFrame({"a": [0.1, 0.2], "b": [0.3, 0.4]}).to_csv(batch, index=False)
    out = tmp_path / "out.csv"
    result_path = predictor.predict_batch(batch, out)
    assert result_path.exists()
    result = pd.read_csv(result_path)
    assert "prediction" in result.columns
