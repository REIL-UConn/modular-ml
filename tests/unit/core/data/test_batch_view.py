"""Unit tests for modularml.core.data.batch_view module."""

import numpy as np
import pytest

from modularml.core.data.batch import Batch
from modularml.core.data.batch_view import BatchView
from modularml.core.data.featureset_view import FeatureSetView
from modularml.utils.data.data_format import DataFormat
from tests.conftest import generate_dummy_featureset


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------
@pytest.fixture
def simple_featureset():
    """Create a simple FeatureSet for testing."""
    return generate_dummy_featureset(
        n_samples=20,
        feature_shape_map={"X": (5,)},
        target_shape_map={"Y": (1,)},
        tag_type_map={"group": "str"},
        label="TestFS",
    )


@pytest.fixture
def single_role_bv(simple_featureset):
    """Create a BatchView with a single role."""
    return BatchView(
        source=simple_featureset,
        role_indices={"default": np.array([0, 1, 2, 3])},
    )


@pytest.fixture
def multi_role_bv(simple_featureset):
    """Create a BatchView with multiple roles."""
    return BatchView(
        source=simple_featureset,
        role_indices={
            "anchor": np.array([0, 1, 2, 3]),
            "pair": np.array([4, 5, 6, 7]),
        },
    )


@pytest.fixture
def weighted_bv(simple_featureset):
    """Create a BatchView with sample weights."""
    return BatchView(
        source=simple_featureset,
        role_indices={"default": np.array([0, 1, 2, 3])},
        role_indice_weights={
            "default": np.array([1.0, 0.5, 0.5, 1.0], dtype=np.float32),
        },
    )


@pytest.fixture
def padded_bv(simple_featureset):
    """Create a BatchView with padded (negative) indices."""
    return BatchView(
        source=simple_featureset,
        role_indices={"default": np.array([0, -1, 2, -1])},
    )


# ---------------------------------------------------------------------
# Initialization tests
# ---------------------------------------------------------------------
@pytest.mark.unit
def test_batch_view_init(single_role_bv):
    """Test basic BatchView initialization."""
    assert isinstance(single_role_bv, BatchView)
    assert single_role_bv.n_samples == 4
    assert single_role_bv.roles == ["default"]


@pytest.mark.unit
def test_batch_view_multi_role(multi_role_bv):
    """Test multi-role BatchView initialization."""
    assert set(multi_role_bv.roles) == {"anchor", "pair"}
    assert multi_role_bv.n_samples == 4


@pytest.mark.unit
def test_batch_view_non_featureset_source_raises():
    """Test BatchView raises for non-FeatureSet source."""
    with pytest.raises(TypeError, match="FeatureSet"):
        BatchView(
            source="not a featureset",
            role_indices={"default": np.array([0, 1])},
        )


@pytest.mark.unit
def test_batch_view_non_dict_indices_raises(simple_featureset):
    """Test BatchView raises for non-dict role_indices."""
    with pytest.raises(TypeError, match="dict"):
        BatchView(
            source=simple_featureset,
            role_indices="not a dict",
        )


@pytest.mark.unit
def test_batch_view_mismatched_weight_keys_raises(simple_featureset):
    """Test BatchView raises for mismatched weight keys."""
    with pytest.raises(ValueError, match="same role keys"):
        BatchView(
            source=simple_featureset,
            role_indices={"default": np.array([0, 1])},
            role_indice_weights={"wrong": np.array([1.0, 1.0], dtype=np.float32)},
        )


# ---------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------
@pytest.mark.unit
def test_batch_view_roles(single_role_bv):
    """Test roles property."""
    assert single_role_bv.roles == ["default"]


@pytest.mark.unit
def test_batch_view_n_samples(single_role_bv):
    """Test n_samples property."""
    assert single_role_bv.n_samples == 4


@pytest.mark.unit
def test_batch_view_role_masks(single_role_bv):
    """Test role_masks property."""
    masks = single_role_bv.role_masks
    assert "default" in masks
    assert np.all(masks["default"] == 1)


@pytest.mark.unit
def test_batch_view_role_masks_padded(padded_bv):
    """Test role_masks with padded indices."""
    masks = padded_bv.role_masks
    mask = masks["default"]
    assert mask[0] == 1
    assert mask[1] == 0
    assert mask[2] == 1
    assert mask[3] == 0


# ---------------------------------------------------------------------
# Data accessor tests
# ---------------------------------------------------------------------
@pytest.mark.unit
def test_get_role_mask(single_role_bv):
    """Test get_role_mask method."""
    mask = single_role_bv.get_role_mask("default")
    assert mask.shape == (4,)
    assert np.all(mask == 1)


@pytest.mark.unit
def test_get_role_mask_invalid_role_raises(single_role_bv):
    """Test get_role_mask raises for invalid role."""
    with pytest.raises(KeyError, match="not found"):
        single_role_bv.get_role_view("nonexistent")


@pytest.mark.unit
def test_get_role_view(single_role_bv):
    """Test get_role_view returns FeatureSetView."""
    view = single_role_bv.get_role_view("default")
    assert isinstance(view, FeatureSetView)
    assert view.n_samples == 4


# ---------------------------------------------------------------------
# Materialize tests
# ---------------------------------------------------------------------
@pytest.mark.unit
def test_materialize_batch_numpy(single_role_bv):
    """Test materialize_batch with NUMPY format."""
    batch = single_role_bv.materialize_batch(fmt=DataFormat.NUMPY)
    assert isinstance(batch, Batch)
    assert batch.batch_size == 4
    assert batch.features is not None


@pytest.mark.unit
def test_materialize_batch_non_tensorlike_raises(single_role_bv):
    """Test materialize_batch raises for non-tensorlike format."""
    with pytest.raises(TypeError, match="tensor-like"):
        single_role_bv.materialize_batch(fmt=DataFormat.DICT_NUMPY)


@pytest.mark.unit
def test_materialize_batch_with_weights(weighted_bv):
    """Test materialize_batch preserves weights."""
    batch = weighted_bv.materialize_batch(fmt=DataFormat.NUMPY)
    assert batch.batch_size == 4
    weights = batch.role_weights["default"]
    assert weights[0] == 1.0
    assert weights[1] == 0.5


# ---------------------------------------------------------------------
# Representation tests
# ---------------------------------------------------------------------
@pytest.mark.unit
def test_batch_view_repr(single_role_bv):
    """Test __repr__ method."""
    repr_str = repr(single_role_bv)
    assert "BatchView" in repr_str
    assert "n_samples=4" in repr_str
