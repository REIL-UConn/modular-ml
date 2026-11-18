"""YAML round trip of a scikit-learn ModelNode keeps the estimator and its hyperparameters."""

from pathlib import Path

import numpy as np
import pytest

from modularml import Experiment, FeatureSet, FitPhase, ModelGraph, ModelNode
from modularml.core.models import ScikitModelWrapper


@pytest.mark.unit
def test_scikit_node_yaml_roundtrip(tmp_path):
    from sklearn.ensemble import RandomForestClassifier

    exp = Experiment("yaml_rt", registration_policy="overwrite")
    rng = np.random.default_rng(0)
    fs = FeatureSet.from_dict(
        "fs",
        {
            "a": rng.normal(size=40),
            "b": rng.normal(size=40),
            "y": rng.integers(0, 2, 40),
        },
        feature_keys=["a", "b"],
        target_keys="y",
    )
    fs.split_random({"train": 0.75, "test": 0.25}, seed=1)
    node = ModelNode(
        label="rf",
        upstream_ref=fs.reference(features=["a", "b"], targets="y"),
        model=ScikitModelWrapper(
            RandomForestClassifier(n_estimators=7, max_depth=3, random_state=5),
            output_method="predict_proba",
            training_mode="batch_fit",
        ),
    )
    ModelGraph(label="g", nodes=[node]).build()
    exp.execution_plan.add_phase(
        FitPhase.from_split(label="fit", split="train", active_nodes=[node]),
    )

    path = exp.to_yaml(tmp_path / "exp.yaml", overwrite=True)
    text = Path(path).read_text()
    assert "RandomForestClassifier" in text
    assert "n_estimators: 7" in text

    reloaded = Experiment.from_yaml(path, overwrite=True)
    (rf,) = [n for n in reloaded.model_graph.nodes.values() if n.label == "rf"]
    est = rf._model.model
    assert isinstance(est, RandomForestClassifier)
    params = est.get_params()
    assert (params["n_estimators"], params["max_depth"], params["random_state"]) == (
        7,
        3,
        5,
    )
    assert rf._model._output_method == "predict_proba"
