"""Integration tests for tag access through EvalResults."""

import numpy as np
import pytest

from modularml.core.data.featureset import FeatureSet
from modularml.core.experiment.experiment import Experiment
from modularml.core.experiment.phases.eval_phase import EvalPhase
from modularml.core.experiment.phases.fit_phase import FitPhase
from modularml.core.models.scikit_wrapper import ScikitModelWrapper
from modularml.core.topology.model_graph import ModelGraph
from modularml.core.topology.model_node import ModelNode
from modularml.utils.data.data_format import DataFormat

sklearn_linear = pytest.importorskip("sklearn.linear_model")


@pytest.fixture
def eval_setup():
    """Fit a linear model on 'train' and evaluate on 'test' (tagged samples)."""
    n = 30
    rng = np.random.default_rng(0)

    exp = Experiment(label="tags_exp", registration_policy="overwrite")
    fs = FeatureSet.from_dict(
        label="tags_fs",
        data={
            "a": rng.normal(size=n),
            "b": rng.normal(size=n),
            "y": rng.normal(size=n),
            "group": np.asarray([f"g{i % 3}" for i in range(n)], dtype=str),
            "idx": np.arange(n),
        },
        feature_keys=["a", "b"],
        target_keys=["y"],
        tag_keys=["group", "idx"],
    )
    fs.split_random(ratios={"train": 0.7, "test": 0.3}, seed=0)

    node = ModelNode(
        label="lin",
        upstream_ref=fs.reference(features=["a", "b"], targets="y"),
        model=ScikitModelWrapper(
            sklearn_linear.LinearRegression(),
            training_mode="batch_fit",
        ),
    )
    ModelGraph(label="graph", nodes=[node]).build()

    exp.execution_plan.add_phase(
        FitPhase.from_split(label="fit", split="train", active_nodes=[node]),
    )
    exp.execution_plan.add_phase(EvalPhase.from_split(label="eval", split="test"))
    results = exp.run()
    return fs, node, results[-1]


@pytest.mark.integration
def test_stacked_tensors_tags_shape_and_order(eval_setup):
    """Test tags stack across batches as (n_samples, n_tags), sorted by key."""
    _, node, ev = eval_setup

    tags = ev.stacked_tensors(node, domain="tags")
    uuids = ev.stacked_tensors(node, domain="sample_uuids")

    assert isinstance(tags, np.ndarray)
    assert tags.shape == (len(uuids), 2)

    # Columns are sorted by key: ("group", "idx")
    assert set(tags[:, 0]) <= {"g0", "g1", "g2"}
    assert all(int(v) in range(30) for v in tags[:, 1])


@pytest.mark.integration
def test_stacked_tensors_tags_align_with_sample_uuids(eval_setup):
    """Test each stacked tag row matches the source tags of its sample UUID."""
    fs, node, ev = eval_setup

    tags = ev.stacked_tensors(node, domain="tags")
    uuids = np.asarray(ev.stacked_tensors(node, domain="sample_uuids")).reshape(-1)

    src_uuids = np.asarray(fs.get_sample_uuids()).reshape(-1)
    src_tags = fs.get_tags(fmt=DataFormat.NUMPY)
    pos = {u: i for i, u in enumerate(src_uuids)}

    expected = src_tags[[pos[u] for u in uuids]]
    np.testing.assert_array_equal(tags, expected)
