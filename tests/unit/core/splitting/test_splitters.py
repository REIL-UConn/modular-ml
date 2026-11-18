"""Tests for FeatureSet splitters and split registration."""

import time

import numpy as np
import pytest

from modularml.core.data.featureset import FeatureSet
from modularml.predicates import EQ, In


def _featureset(n: int = 200) -> FeatureSet:
    rng = np.random.default_rng(0)
    return FeatureSet.from_dict(
        label="fs",
        data={
            "x0": rng.normal(size=n),
            "x1": rng.normal(size=n),
            "y": rng.integers(0, 3, n),
            "row_id": np.arange(n, dtype=np.int64),
            "group": np.asarray([f"g{i % 7}" for i in range(n)]),
        },
        feature_keys=["x0", "x1"],
        target_keys="y",
        tag_keys=["row_id", "group"],
    )


def _row_ids(fs: FeatureSet, split: str) -> np.ndarray:
    return np.asarray(fs.get_split(split).get_tags(fmt="numpy", tags="row_id")).ravel()


# ================================================
# ConditionSplitter
# ================================================
@pytest.mark.unit
def test_condition_in_on_int_and_str_tags():
    fs = _featureset()
    ids = list(range(0, 200, 3))
    fs.split_by_condition(
        {
            "ids": {"tags.row_id.raw": In(ids)},
            "g2": {"tags.group.raw": EQ("g2")},
            "g13": {"tags.group.raw": In(["g1", "g3"])},
        },
    )
    np.testing.assert_array_equal(_row_ids(fs, "ids"), ids)
    groups = np.asarray([f"g{i % 7}" for i in range(200)])
    np.testing.assert_array_equal(_row_ids(fs, "g2"), np.flatnonzero(groups == "g2"))
    np.testing.assert_array_equal(
        _row_ids(fs, "g13"),
        np.flatnonzero(np.isin(groups, ["g1", "g3"])),
    )


@pytest.mark.unit
def test_condition_in_many_values_is_vectorized():
    n = 60_000
    fs = FeatureSet.from_dict(
        label="big",
        data={"x": np.zeros(n), "y": np.zeros(n), "row_id": np.arange(n)},
        feature_keys="x",
        target_keys="y",
        tag_keys="row_id",
    )
    ids = np.arange(0, n, 2)
    t0 = time.perf_counter()
    fs.split_by_condition({"half": {"tags.row_id.raw": In(ids.tolist())}})
    elapsed = time.perf_counter() - t0
    np.testing.assert_array_equal(_row_ids(fs, "half"), ids)
    # An object-dtype membership test takes several seconds at this size.
    assert elapsed < 2.0
