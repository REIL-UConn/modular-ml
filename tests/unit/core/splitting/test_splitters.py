"""Tests for FeatureSet splitters and split registration."""

import time

import numpy as np
import pytest

from modularml.core.data.featureset import FeatureSet
from modularml.core.splitting.base_splitter import BaseSplitter
from modularml.predicates import EQ, In
from modularml.splitters import IndexSplitter, splitter_registry


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


# ================================================
# IndexSplitter
# ================================================
@pytest.mark.unit
def test_index_splitter_keeps_order_and_repeats():
    fs = _featureset()
    order = [5, 3, 199, 3, 0]
    fs.split_by_indices({"bag": order, "rest": [1, 2]})
    np.testing.assert_array_equal(_row_ids(fs, "bag"), order)
    np.testing.assert_array_equal(_row_ids(fs, "rest"), [1, 2])


@pytest.mark.unit
def test_index_splitter_relative_to_view():
    fs = _featureset()
    fs.split_by_condition({"g0": {"tags.group.raw": EQ("g0")}})
    view = fs.get_split("g0")
    view.split_by_indices({"g0_first_two": [1, 0]})
    expected = _row_ids(fs, "g0")[[1, 0]]
    np.testing.assert_array_equal(_row_ids(fs, "g0_first_two"), expected)


@pytest.mark.unit
def test_index_splitter_by_sample_uuid():
    fs = _featureset()
    uuids = np.asarray(fs.get_sample_uuids()).ravel()
    pick = [17, 2, 150]
    fs.split_by_indices({"picked": uuids[pick].tolist()}, by="sample_uuid")
    np.testing.assert_array_equal(_row_ids(fs, "picked"), pick)
    with pytest.raises(KeyError):
        fs.split_by_indices({"bad": ["not-a-uuid"]}, by="sample_uuid")


@pytest.mark.unit
def test_index_splitter_validation():
    with pytest.raises(ValueError, match="by"):
        IndexSplitter({"a": [0]}, by="row")
    with pytest.raises(ValueError, match="integer"):
        IndexSplitter({"a": [0.5]})
    fs = _featureset(10)
    with pytest.raises(IndexError):
        fs.split_by_indices({"a": [10]})


@pytest.mark.unit
def test_index_splitter_config_roundtrip_and_record():
    sp = IndexSplitter({"a": np.array([4, 1]), "b": [2]})
    cfg = sp.get_config()
    assert cfg == {
        "splitter_name": "IndexSplitter",
        "by": "position",
        "subsets": {"a": [4, 1], "b": [2]},
    }
    rebuilt = BaseSplitter.from_config(cfg)
    assert isinstance(rebuilt, IndexSplitter)
    assert rebuilt.subsets == sp.subsets
    assert "IndexSplitter" in splitter_registry

    fs = _featureset()
    fs.split_by_indices({"a": [4, 1]})
    rec = fs._split_recs[-1]
    assert rec.produced_splits == ["a"]
    assert rec.get_config()["splitter_config"]["subsets"] == {"a": [4, 1]}


# ================================================
# remove_splits
# ================================================
@pytest.mark.unit
def test_remove_splits_drops_record():
    fs = _featureset()
    fs.split_by_condition({"g0": {"tags.group.raw": EQ("g0")}})
    fs.split_by_indices({"tr": [0, 1], "va": [2]})
    n_recs = len(fs._split_recs)
    fs.remove_splits(["tr", "va"])
    assert set(fs.available_splits) == {"g0"}
    assert len(fs._split_recs) == n_recs - 1
    # the names can be reused afterwards
    fs.split_by_indices({"tr": [3], "va": [4]})
    np.testing.assert_array_equal(_row_ids(fs, "tr"), [3])


@pytest.mark.unit
def test_remove_splits_refuses_partial_record_and_dependents():
    fs = _featureset()
    fs.split_by_indices({"tr": [0, 1, 2, 3], "va": [4]})
    with pytest.raises(ValueError, match="together"):
        fs.remove_splits("tr")
    fs.get_split("tr").split_by_indices({"tr_sub": [0]})
    with pytest.raises(ValueError, match="derived"):
        fs.remove_splits(["tr", "va"])
    with pytest.raises(KeyError):
        fs.remove_splits("nope")


@pytest.mark.unit
def test_remove_splits_refuses_scaler_fit_split():
    from sklearn.preprocessing import StandardScaler

    fs = _featureset()
    fs.split_by_indices({"tr": list(range(100))})
    fs.fit_transform(StandardScaler(), domain="features", fit_to_split="tr")
    with pytest.raises(ValueError, match="scaler"):
        fs.remove_splits("tr")
