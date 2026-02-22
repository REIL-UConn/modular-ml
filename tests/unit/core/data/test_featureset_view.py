"""Unit tests for modularml.core.data.featureset_view module."""

import numpy as np
import pandas as pd
import pytest

from modularml.core.data.featureset_view import FeatureSetView
from tests.conftest import generate_dummy_featureset


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------
@pytest.fixture
def simple_featureset():
    """Create a simple FeatureSet for testing."""
    return generate_dummy_featureset(
        n_samples=10,
        feature_shape_map={"X": (5,)},
        target_shape_map={"Y": (1,)},
        tag_type_map={"group": "str"},
        label="TestFS",
    )


@pytest.fixture
def basic_view(simple_featureset):
    """Create a basic FeatureSetView."""
    return FeatureSetView.from_featureset(simple_featureset)


@pytest.fixture
def subset_view(simple_featureset):
    """Create a FeatureSetView with subset of rows."""
    return FeatureSetView(
        source=simple_featureset,
        indices=np.array([0, 2, 4, 6, 8]),
        columns=simple_featureset.get_all_keys(
            include_domain_prefix=True,
            include_rep_suffix=True,
        ),
        label="subset",
    )


@pytest.fixture
def another_subset_view(simple_featureset):
    """Create another FeatureSetView with different rows."""
    return FeatureSetView(
        source=simple_featureset,
        indices=np.array([1, 3, 5, 7, 9]),
        columns=simple_featureset.get_all_keys(
            include_domain_prefix=True,
            include_rep_suffix=True,
        ),
        label="another_subset",
    )


@pytest.fixture
def overlapping_view(simple_featureset):
    """Create a FeatureSetView that overlaps with subset_view."""
    return FeatureSetView(
        source=simple_featureset,
        indices=np.array([0, 1, 2, 3, 4]),
        columns=simple_featureset.get_all_keys(
            include_domain_prefix=True,
            include_rep_suffix=True,
        ),
        label="overlapping",
    )


@pytest.fixture
def padded_view(simple_featureset):
    """Create a FeatureSetView with negative (padded) indices."""
    return FeatureSetView(
        source=simple_featureset,
        indices=np.array([0, -1, 2, -1, 4]),
        columns=simple_featureset.get_all_keys(
            include_domain_prefix=True,
            include_rep_suffix=True,
        ),
        label="padded",
    )


# ---------------------------------------------------------------------
# Initialization tests
# ---------------------------------------------------------------------
@pytest.mark.unit
def test_featureset_view_from_featureset(simple_featureset):
    """Test creating view from FeatureSet."""
    view = FeatureSetView.from_featureset(simple_featureset)
    assert isinstance(view, FeatureSetView)
    assert view.source is simple_featureset
    assert len(view.indices) == 10


@pytest.mark.unit
def test_featureset_view_from_featureset_with_rows(simple_featureset):
    """Test creating view with specific rows."""
    rows = np.array([0, 2, 4])
    view = FeatureSetView.from_featureset(simple_featureset, rows=rows)
    assert len(view.indices) == 3
    assert np.array_equal(view.indices, rows)


@pytest.mark.unit
def test_featureset_view_from_featureset_with_columns(simple_featureset):
    """Test creating view with specific columns."""
    columns = ["features.X.raw", "targets.Y.raw"]
    view = FeatureSetView.from_featureset(simple_featureset, columns=columns)
    assert view.columns == columns


@pytest.mark.unit
def test_featureset_view_init_with_label(simple_featureset):
    """Test creating view with label."""
    view = FeatureSetView(
        source=simple_featureset,
        indices=np.arange(10),
        columns=simple_featureset.get_all_keys(
            include_domain_prefix=True,
            include_rep_suffix=True,
        ),
        label="my_view",
    )
    assert view.label == "my_view"


# ---------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------
@pytest.mark.unit
def test_featureset_view_n_samples(basic_view, subset_view):
    """Test n_samples property."""
    assert basic_view.n_samples == 10
    assert subset_view.n_samples == 5


@pytest.mark.unit
def test_featureset_view_valid_indices(subset_view, padded_view):
    """Test valid_indices property."""
    valid = subset_view.valid_indices
    assert len(valid) == 5
    assert np.all(valid >= 0)

    padded_valid = padded_view.valid_indices
    assert len(padded_valid) == 3  # Only 0, 2, 4 are valid


@pytest.mark.unit
def test_featureset_view_sample_mask(basic_view, padded_view):
    """Test sample_mask property."""
    mask = basic_view.sample_mask
    assert mask is None or np.all(mask == 1)

    padded_mask = padded_view.sample_mask
    assert padded_mask[0] == 1
    assert padded_mask[1] == 0
    assert padded_mask[2] == 1
    assert padded_mask[3] == 0
    assert padded_mask[4] == 1


# ---------------------------------------------------------------------
# Equality and comparison tests
# ---------------------------------------------------------------------
@pytest.mark.unit
def test_featureset_view_equality_same(simple_featureset):
    """Test equality for same views."""
    v1 = FeatureSetView.from_featureset(simple_featureset, rows=np.array([0, 1, 2]))
    v2 = FeatureSetView.from_featureset(simple_featureset, rows=np.array([0, 1, 2]))
    assert v1.source == v2.source
    assert np.array_equal(v1.indices, v2.indices)


@pytest.mark.unit
def test_featureset_view_equality_type_error(basic_view):
    """Test equality raises for non-FeatureSetView."""
    with pytest.raises(TypeError):
        _ = basic_view == "not a view"


# ---------------------------------------------------------------------
# Immutability tests
# ---------------------------------------------------------------------
@pytest.mark.unit
def test_featureset_view_frozen_source(basic_view):
    """Test that source attribute is frozen."""
    another_fs = generate_dummy_featureset(n_samples=5, label="AnotherFS")
    with pytest.raises(AttributeError, match="frozen"):
        basic_view.source = another_fs


@pytest.mark.unit
def test_featureset_view_frozen_indices(basic_view):
    """Test that indices attribute is frozen."""
    with pytest.raises(AttributeError, match="frozen"):
        basic_view.indices = np.array([0, 1])


@pytest.mark.unit
def test_featureset_view_frozen_columns(basic_view):
    """Test that columns attribute is frozen."""
    with pytest.raises(AttributeError, match="frozen"):
        basic_view.columns = ["new_col"]


@pytest.mark.unit
def test_featureset_view_label_can_change(basic_view):
    """Test that label can be changed."""
    basic_view.label = "new_label"
    assert basic_view.label == "new_label"


# ---------------------------------------------------------------------
# Disjointness tests
# ---------------------------------------------------------------------
@pytest.mark.unit
def test_featureset_view_is_disjoint_true(subset_view, another_subset_view):
    """Test is_disjoint_with for disjoint views."""
    assert subset_view.is_disjoint_with(another_subset_view)


@pytest.mark.unit
def test_featureset_view_is_disjoint_false(subset_view, overlapping_view):
    """Test is_disjoint_with for overlapping views."""
    assert not subset_view.is_disjoint_with(overlapping_view)


@pytest.mark.unit
def test_featureset_view_is_disjoint_type_error(basic_view):
    """Test is_disjoint_with raises for non-FeatureSetView."""
    with pytest.raises(TypeError):
        basic_view.is_disjoint_with("not a view")


# ---------------------------------------------------------------------
# Overlap tests
# ---------------------------------------------------------------------
@pytest.mark.unit
def test_featureset_view_get_overlap_empty(subset_view, another_subset_view):
    """Test get_overlap_with for disjoint views."""
    overlap = subset_view.get_overlap_with(another_subset_view)
    assert len(overlap) == 0


@pytest.mark.unit
def test_featureset_view_get_overlap_non_empty(subset_view, overlapping_view):
    """Test get_overlap_with for overlapping views."""
    overlap = subset_view.get_overlap_with(overlapping_view)
    assert len(overlap) == 3  # Indices 0, 2, 4 overlap


@pytest.mark.unit
def test_featureset_view_get_overlap_type_error(basic_view):
    """Test get_overlap_with raises for non-FeatureSetView."""
    with pytest.raises(TypeError):
        basic_view.get_overlap_with("not a view")


# ---------------------------------------------------------------------
# Configuration tests
# ---------------------------------------------------------------------
@pytest.mark.unit
def test_featureset_view_get_config(subset_view):
    """Test get_config method."""
    config = subset_view.get_config()
    assert "source" in config
    assert "indices" in config
    assert "columns" in config
    assert "label" in config
    assert config["label"] == "subset"


# ---------------------------------------------------------------------
# Data access tests (via SampleCollectionMixin)
# ---------------------------------------------------------------------
@pytest.mark.unit
def test_featureset_view_get_feature_keys(basic_view):
    """Test get_feature_keys method."""
    keys = basic_view.get_feature_keys(
        include_domain_prefix=False,
        include_rep_suffix=False,
    )
    assert "X" in keys


@pytest.mark.unit
def test_featureset_view_get_target_keys(basic_view):
    """Test get_target_keys method."""
    keys = basic_view.get_target_keys(
        include_domain_prefix=False,
        include_rep_suffix=False,
    )
    assert "Y" in keys


@pytest.mark.unit
def test_featureset_view_get_features(basic_view):
    """Test get_features method."""
    features = basic_view.get_features()
    assert isinstance(features, dict)
    assert "X" in features


@pytest.mark.unit
def test_featureset_view_get_targets(basic_view):
    """Test get_targets method."""
    targets = basic_view.get_targets()
    assert isinstance(targets, dict)
    assert "Y" in targets


@pytest.mark.unit
def test_featureset_view_to_pandas(basic_view):
    """Test to_pandas conversion."""
    df = basic_view.to_pandas()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == basic_view.n_samples


# ---------------------------------------------------------------------
# Representation tests
# ---------------------------------------------------------------------
@pytest.mark.unit
def test_featureset_view_repr(subset_view):
    """Test __repr__ method."""
    repr_str = repr(subset_view)
    assert "FeatureSetView" in repr_str
    assert "TestFS" in repr_str
    assert "n_samples=5" in repr_str
