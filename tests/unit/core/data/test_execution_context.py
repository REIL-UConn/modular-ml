"""Unit tests for modularml.core.data.execution_context module."""

import numpy as np
import pytest

from modularml.core.data.batch import Batch
from modularml.core.data.execution_context import ExecutionContext
from modularml.core.data.sample_data import RoleData, SampleData
from modularml.core.training.loss_record import LossCollection, LossRecord


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------
@pytest.fixture
def ctx():
    """Create a basic ExecutionContext."""
    return ExecutionContext(
        phase_label="train",
        epoch_idx=0,
        batch_idx=0,
    )


@pytest.fixture
def sample_batch():
    """Create a sample Batch for testing."""
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


# ---------------------------------------------------------------------
# Initialization tests
# ---------------------------------------------------------------------
@pytest.mark.unit
def test_init_basic(ctx):
    """Test basic ExecutionContext initialization."""
    assert ctx.phase_label == "train"
    assert ctx.epoch_idx == 0
    assert ctx.batch_idx == 0
    assert ctx.inputs == {}
    assert ctx.outputs == {}
    assert ctx.losses is None


@pytest.mark.unit
def test_init_with_custom_values():
    """Test ExecutionContext with custom epoch and batch."""
    ctx = ExecutionContext(
        phase_label="validation",
        epoch_idx=5,
        batch_idx=3,
    )
    assert ctx.phase_label == "validation"
    assert ctx.epoch_idx == 5
    assert ctx.batch_idx == 3


# ---------------------------------------------------------------------
# set_output tests
# ---------------------------------------------------------------------
@pytest.mark.unit
def test_set_output(ctx, sample_batch):
    """Test set_output stores batch correctly."""
    ctx.set_output(node_id="node_1", batch=sample_batch)
    assert "node_1" in ctx.outputs
    assert ctx.outputs["node_1"] is sample_batch


@pytest.mark.unit
def test_set_output_duplicate_raises(ctx, sample_batch):
    """Test set_output raises for duplicate node_id."""
    ctx.set_output(node_id="node_1", batch=sample_batch)
    with pytest.raises(ValueError, match="already set"):
        ctx.set_output(node_id="node_1", batch=sample_batch)


# ---------------------------------------------------------------------
# add_losses tests
# ---------------------------------------------------------------------
@pytest.mark.unit
def test_add_losses_first_time(ctx):
    """Test add_losses when no losses exist yet."""
    lr = LossRecord(label="mse", trainable=0.5)
    lc = LossCollection(records=[lr])
    ctx.add_losses(lc)
    assert ctx.losses is not None
    assert len(ctx.losses) == 1


@pytest.mark.unit
def test_add_losses_merges(ctx):
    """Test add_losses merges with existing losses."""
    lr1 = LossRecord(label="mse", trainable=0.5)
    lr2 = LossRecord(label="mae", trainable=0.3)
    lc1 = LossCollection(records=[lr1])
    lc2 = LossCollection(records=[lr2])
    ctx.add_losses(lc1)
    ctx.add_losses(lc2)
    assert ctx.losses is not None
    assert len(ctx.losses) == 2
