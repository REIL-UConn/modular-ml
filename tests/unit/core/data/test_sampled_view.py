"""Unit tests for modularml.core.data.sampled_view module."""

import numpy as np
import pytest

from modularml.core.data.batch_view import BatchView
from modularml.core.data.sampled_view import SampledView
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
def batch_view_list(simple_featureset):
    """Create a list of BatchViews."""
    return [
        BatchView(
            source=simple_featureset,
            role_indices={"default": np.array([0, 1, 2, 3])},
        ),
        BatchView(
            source=simple_featureset,
            role_indices={"default": np.array([4, 5, 6, 7])},
        ),
        BatchView(
            source=simple_featureset,
            role_indices={"default": np.array([8, 9, 10, 11])},
        ),
    ]


@pytest.fixture
def single_stream_sampled_view(batch_view_list):
    """Create a SampledView with a single stream."""
    return SampledView(streams={"main": batch_view_list})


@pytest.fixture
def multi_stream_sampled_view(simple_featureset):
    """Create a SampledView with multiple streams."""
    main_batches = [
        BatchView(
            source=simple_featureset,
            role_indices={"default": np.array([0, 1, 2, 3])},
        ),
        BatchView(
            source=simple_featureset,
            role_indices={"default": np.array([4, 5, 6, 7])},
        ),
    ]
    aux_batches = [
        BatchView(
            source=simple_featureset,
            role_indices={"default": np.array([8, 9, 10, 11])},
        ),
        BatchView(
            source=simple_featureset,
            role_indices={"default": np.array([12, 13, 14, 15])},
        ),
    ]
    return SampledView(streams={"main": main_batches, "aux": aux_batches})


# ---------------------------------------------------------------------
# Initialization tests
# ---------------------------------------------------------------------
@pytest.mark.unit
def test_sampled_view_init_valid(single_stream_sampled_view):
    """Test valid SampledView initialization."""
    assert isinstance(single_stream_sampled_view, SampledView)
    assert single_stream_sampled_view.num_streams == 1
    assert single_stream_sampled_view.num_batches == 3


@pytest.mark.unit
def test_sampled_view_init_multi_stream(multi_stream_sampled_view):
    """Test multi-stream SampledView initialization."""
    assert multi_stream_sampled_view.num_streams == 2
    assert set(multi_stream_sampled_view.stream_names) == {"main", "aux"}


@pytest.mark.unit
def test_sampled_view_init_empty_raises():
    """Test that empty streams raises ValueError."""
    with pytest.raises(ValueError, match="at least one stream"):
        SampledView(streams={})


@pytest.mark.unit
def test_sampled_view_init_non_string_key_raises(batch_view_list):
    """Test that non-string stream key raises TypeError."""
    with pytest.raises(TypeError, match="str"):
        SampledView(streams={123: batch_view_list})


@pytest.mark.unit
def test_sampled_view_init_non_list_value_raises():
    """Test that non-list stream value raises TypeError."""
    with pytest.raises(TypeError, match="list"):
        SampledView(streams={"main": "not a list"})


@pytest.mark.unit
def test_sampled_view_init_non_batch_view_in_list_raises():
    """Test that non-BatchView items in list raise TypeError."""
    with pytest.raises(TypeError, match="non-BatchView"):
        SampledView(streams={"main": ["not a batch view"]})


@pytest.mark.unit
def test_sampled_view_init_mismatched_lengths_raises(simple_featureset):
    """Test that mismatched stream lengths raise ValueError."""
    main_batches = [
        BatchView(
            source=simple_featureset,
            role_indices={"default": np.array([0, 1])},
        ),
        BatchView(
            source=simple_featureset,
            role_indices={"default": np.array([2, 3])},
        ),
    ]
    aux_batches = [
        BatchView(
            source=simple_featureset,
            role_indices={"default": np.array([4, 5])},
        ),
    ]
    with pytest.raises(ValueError, match="same number of batches"):
        SampledView(streams={"main": main_batches, "aux": aux_batches})


# ---------------------------------------------------------------------
# Mapping interface tests
# ---------------------------------------------------------------------
@pytest.mark.unit
def test_sampled_view_getitem(single_stream_sampled_view):
    """Test __getitem__ returns list of BatchViews."""
    batches = single_stream_sampled_view["main"]
    assert isinstance(batches, list)
    assert len(batches) == 3
    assert all(isinstance(b, BatchView) for b in batches)


@pytest.mark.unit
def test_sampled_view_iter(single_stream_sampled_view):
    """Test __iter__ yields aligned batch groups."""
    batch_groups = list(single_stream_sampled_view)
    assert len(batch_groups) == 3
    for group in batch_groups:
        assert isinstance(group, dict)
        assert "main" in group
        assert isinstance(group["main"], BatchView)


@pytest.mark.unit
def test_sampled_view_iter_multi_stream(multi_stream_sampled_view):
    """Test __iter__ with multiple streams."""
    batch_groups = list(multi_stream_sampled_view)
    assert len(batch_groups) == 2
    for group in batch_groups:
        assert "main" in group
        assert "aux" in group


@pytest.mark.unit
def test_sampled_view_len(single_stream_sampled_view, multi_stream_sampled_view):
    """Test __len__ returns number of streams."""
    assert len(single_stream_sampled_view) == 1
    assert len(multi_stream_sampled_view) == 2


# ---------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------
@pytest.mark.unit
def test_sampled_view_stream_names(multi_stream_sampled_view):
    """Test stream_names property."""
    names = multi_stream_sampled_view.stream_names
    assert isinstance(names, list)
    assert set(names) == {"main", "aux"}


@pytest.mark.unit
def test_sampled_view_num_streams(
    single_stream_sampled_view,
    multi_stream_sampled_view,
):
    """Test num_streams property."""
    assert single_stream_sampled_view.num_streams == 1
    assert multi_stream_sampled_view.num_streams == 2


@pytest.mark.unit
def test_sampled_view_num_batches(
    single_stream_sampled_view,
    multi_stream_sampled_view,
):
    """Test num_batches property."""
    assert single_stream_sampled_view.num_batches == 3
    assert multi_stream_sampled_view.num_batches == 2


# ---------------------------------------------------------------------
# Accessor tests
# ---------------------------------------------------------------------
@pytest.mark.unit
def test_sampled_view_get_stream(single_stream_sampled_view):
    """Test get_stream method."""
    stream = single_stream_sampled_view.get_stream("main")
    assert isinstance(stream, list)
    assert len(stream) == 3


@pytest.mark.unit
def test_sampled_view_getattr_stream_access(multi_stream_sampled_view):
    """Test attribute-style stream access."""
    main = multi_stream_sampled_view.main
    assert isinstance(main, list)

    aux = multi_stream_sampled_view.aux
    assert isinstance(aux, list)


@pytest.mark.unit
def test_sampled_view_getattr_invalid_raises(single_stream_sampled_view):
    """Test invalid attribute raises AttributeError."""
    with pytest.raises(AttributeError, match="no stream"):
        _ = single_stream_sampled_view.nonexistent_stream


# ---------------------------------------------------------------------
# Utility tests
# ---------------------------------------------------------------------
@pytest.mark.unit
def test_sampled_view_to_dict(single_stream_sampled_view):
    """Test to_dict method."""
    d = single_stream_sampled_view.to_dict()
    assert isinstance(d, dict)
    assert "main" in d
    assert isinstance(d["main"], list)


# ---------------------------------------------------------------------
# Representation tests
# ---------------------------------------------------------------------
@pytest.mark.unit
def test_sampled_view_repr(single_stream_sampled_view):
    """Test __repr__ method."""
    repr_str = repr(single_stream_sampled_view)
    assert "SampledView" in repr_str
    assert "main" in repr_str
    assert "num_batches=3" in repr_str
