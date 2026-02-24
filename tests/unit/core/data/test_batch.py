"""Unit tests for modularml.core.data.batch module."""

import numpy as np
import pytest

from modularml.core.data.batch import Batch
from modularml.core.data.sample_data import RoleData, SampleData, SampleShapes
from modularml.utils.data.data_format import DataFormat


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------
@pytest.fixture
def single_role_batch():
    """Create a Batch with a single role."""
    features = np.random.rand(4, 10).astype(np.float32)  # noqa: NPY002
    targets = np.random.rand(4, 1).astype(np.float32)  # noqa: NPY002
    sd = SampleData(features=features, targets=targets)
    rd = RoleData(data={"default": sd})
    return Batch(
        batch_size=4,
        role_data=rd,
        shapes=sd.shapes,
        role_weights={"default": np.ones(4, dtype=np.float32)},
        role_masks={"default": np.ones(4, dtype=np.int8)},
    )


@pytest.fixture
def multi_role_batch():
    """Create a Batch with multiple roles."""
    sd_anchor = SampleData(
        features=np.random.rand(4, 10).astype(np.float32),  # noqa: NPY002
        targets=np.random.rand(4, 1).astype(np.float32),  # noqa: NPY002
    )
    sd_pair = SampleData(
        features=np.random.rand(4, 10).astype(np.float32),  # noqa: NPY002
        targets=np.random.rand(4, 1).astype(np.float32),  # noqa: NPY002
    )
    rd = RoleData(data={"anchor": sd_anchor, "pair": sd_pair})
    return Batch(
        batch_size=4,
        role_data=rd,
        shapes=sd_anchor.shapes,
        role_weights={
            "anchor": np.ones(4, dtype=np.float32),
            "pair": np.ones(4, dtype=np.float32),
        },
        role_masks={
            "anchor": np.ones(4, dtype=np.int8),
            "pair": np.ones(4, dtype=np.int8),
        },
    )


# ---------------------------------------------------------------------
# Initialization tests
# ---------------------------------------------------------------------
@pytest.mark.unit
def test_batch_init(single_role_batch):
    """Test basic Batch initialization."""
    assert single_role_batch.batch_size == 4
    assert isinstance(single_role_batch.role_data, RoleData)
    assert isinstance(single_role_batch.shapes, SampleShapes)


@pytest.mark.unit
def test_batch_init_multi_role(multi_role_batch):
    """Test multi-role Batch initialization."""
    assert multi_role_batch.batch_size == 4
    assert set(multi_role_batch.available_roles) == {"anchor", "pair"}


@pytest.mark.unit
def test_batch_init_mismatched_weights_raises():
    """Test Batch raises for mismatched role_weights keys."""
    sd = SampleData(features=np.ones((4, 5)), targets=np.ones((4, 1)))
    rd = RoleData(data={"default": sd})
    with pytest.raises(ValueError, match="role_weights"):
        Batch(
            batch_size=4,
            role_data=rd,
            shapes=sd.shapes,
            role_weights={"wrong_role": np.ones(4, dtype=np.float32)},
            role_masks={"default": np.ones(4, dtype=np.int8)},
        )


@pytest.mark.unit
def test_batch_init_wrong_weight_shape_raises():
    """Test Batch raises for wrong weight shape."""
    sd = SampleData(features=np.ones((4, 5)), targets=np.ones((4, 1)))
    rd = RoleData(data={"default": sd})
    with pytest.raises(ValueError, match="role_weights"):
        Batch(
            batch_size=4,
            role_data=rd,
            shapes=sd.shapes,
            role_weights={"default": np.ones(3, dtype=np.float32)},
            role_masks={"default": np.ones(4, dtype=np.int8)},
        )


@pytest.mark.unit
def test_batch_init_wrong_mask_shape_raises():
    """Test Batch raises for wrong mask shape."""
    sd = SampleData(features=np.ones((4, 5)), targets=np.ones((4, 1)))
    rd = RoleData(data={"default": sd})
    with pytest.raises(ValueError, match="role_masks"):
        Batch(
            batch_size=4,
            role_data=rd,
            shapes=sd.shapes,
            role_weights={"default": np.ones(4, dtype=np.float32)},
            role_masks={"default": np.ones(3, dtype=np.int8)},
        )


# ---------------------------------------------------------------------
# Data access tests
# ---------------------------------------------------------------------
@pytest.mark.unit
def test_batch_available_roles(single_role_batch):
    """Test available_roles property."""
    assert single_role_batch.available_roles == ["default"]


@pytest.mark.unit
def test_batch_get_data_no_args(single_role_batch):
    """Test get_data returns RoleData when no args."""
    result = single_role_batch.get_data()
    assert isinstance(result, RoleData)


@pytest.mark.unit
def test_batch_get_data_with_role(single_role_batch):
    """Test get_data with role returns SampleData."""
    result = single_role_batch.get_data(role="default")
    assert isinstance(result, SampleData)


@pytest.mark.unit
def test_batch_get_data_with_role_and_domain(single_role_batch):
    """Test get_data with role and domain returns tensor."""
    result = single_role_batch.get_data(role="default", domain="features")
    assert isinstance(result, np.ndarray)


# ---------------------------------------------------------------------
# Single-role pass-through tests
# ---------------------------------------------------------------------
@pytest.mark.unit
def test_batch_features_single_role(single_role_batch):
    """Test features property for single role."""
    assert single_role_batch.features is not None
    assert single_role_batch.features.shape[0] == 4


@pytest.mark.unit
def test_batch_targets_single_role(single_role_batch):
    """Test targets property for single role."""
    assert single_role_batch.targets is not None


@pytest.mark.unit
def test_batch_features_multi_role_raises(multi_role_batch):
    """Test features property raises for multi-role batch."""
    with pytest.raises(RuntimeError, match="exactly one role"):
        _ = multi_role_batch.features


# ---------------------------------------------------------------------
# Pseudo-attribute access tests
# ---------------------------------------------------------------------
@pytest.mark.unit
def test_batch_getattr_role(multi_role_batch):
    """Test attribute-style role access."""
    anchor_data = multi_role_batch.anchor
    assert isinstance(anchor_data, SampleData)


@pytest.mark.unit
def test_batch_getattr_invalid_raises(single_role_batch):
    """Test invalid attribute access raises AttributeError."""
    with pytest.raises(AttributeError, match="no attribute"):
        _ = single_role_batch.nonexistent


# ---------------------------------------------------------------------
# Format conversion tests
# ---------------------------------------------------------------------
@pytest.mark.unit
def test_batch_to_format(single_role_batch):
    """Test to_format returns new Batch."""
    new_batch = single_role_batch.to_format(DataFormat.NUMPY)
    assert isinstance(new_batch, Batch)
    assert new_batch is not single_role_batch


# ---------------------------------------------------------------------
# Concatenation tests
# ---------------------------------------------------------------------
@pytest.mark.unit
def test_batch_concat():
    """Test Batch.concat merges batches."""
    sd1 = SampleData(features=np.ones((2, 5)), targets=np.ones((2, 1)))
    sd2 = SampleData(features=np.zeros((3, 5)), targets=np.zeros((3, 1)))
    rd1 = RoleData(data={"default": sd1})
    rd2 = RoleData(data={"default": sd2})
    b1 = Batch(
        batch_size=2,
        role_data=rd1,
        shapes=sd1.shapes,
        role_weights={"default": np.ones(2, dtype=np.float32)},
        role_masks={"default": np.ones(2, dtype=np.int8)},
    )
    b2 = Batch(
        batch_size=3,
        role_data=rd2,
        shapes=sd2.shapes,
        role_weights={"default": np.ones(3, dtype=np.float32)},
        role_masks={"default": np.ones(3, dtype=np.int8)},
    )
    result = Batch.concat(b1, b2)
    assert result.batch_size == 5
    assert result.features.shape == (5, 5)


@pytest.mark.unit
def test_batch_concat_with():
    """Test Batch.concat_with method."""
    sd1 = SampleData(features=np.ones((2, 5)), targets=np.ones((2, 1)))
    sd2 = SampleData(features=np.zeros((3, 5)), targets=np.zeros((3, 1)))
    rd1 = RoleData(data={"default": sd1})
    rd2 = RoleData(data={"default": sd2})
    b1 = Batch(
        batch_size=2,
        role_data=rd1,
        shapes=sd1.shapes,
        role_weights={"default": np.ones(2, dtype=np.float32)},
        role_masks={"default": np.ones(2, dtype=np.int8)},
    )
    b2 = Batch(
        batch_size=3,
        role_data=rd2,
        shapes=sd2.shapes,
        role_weights={"default": np.ones(3, dtype=np.float32)},
        role_masks={"default": np.ones(3, dtype=np.int8)},
    )
    result = b1.concat_with(b2)
    assert result.batch_size == 5


@pytest.mark.unit
def test_batch_concat_type_error():
    """Test concat raises TypeError for non-Batch."""
    sd = SampleData(features=np.ones((2, 5)), targets=np.ones((2, 1)))
    rd = RoleData(data={"default": sd})
    b = Batch(
        batch_size=2,
        role_data=rd,
        shapes=sd.shapes,
        role_weights={"default": np.ones(2, dtype=np.float32)},
        role_masks={"default": np.ones(2, dtype=np.int8)},
    )
    with pytest.raises(TypeError, match="Expected Batch"):
        Batch.concat(b, "not a batch")


# ---------------------------------------------------------------------
# Representation tests
# ---------------------------------------------------------------------
@pytest.mark.unit
def test_batch_repr(single_role_batch):
    """Test __repr__ method."""
    repr_str = repr(single_role_batch)
    assert "Batch" in repr_str
    assert "batch_size=4" in repr_str
