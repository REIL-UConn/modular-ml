"""
Unit tests for modularml.core.data.sample_collection_mixin module.

The SampleCollectionMixin is tested through concrete implementations:
FeatureSet and FeatureSetView.
"""

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from modularml.core.data.featureset import FeatureSet
from modularml.core.data.featureset_view import FeatureSetView
from modularml.core.data.sample_collection import SampleCollection
from modularml.core.data.sample_data import SampleData
from modularml.utils.data.data_format import DataFormat
from tests.conftest import generate_dummy_featureset


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------
@pytest.fixture
def simple_featureset():
    """Create a simple FeatureSet for testing."""
    return generate_dummy_featureset(
        n_samples=10,
        feature_shape_map={"X1": (5,), "X2": (3,)},
        target_shape_map={"Y1": (1,), "Y2": (2,)},
        tag_type_map={"group": "str", "value": "float"},
        label="TestFS",
    )


@pytest.fixture
def featureset_view(simple_featureset):
    """Create a FeatureSetView for testing."""
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
    )


@pytest.fixture
def padded_view(simple_featureset):
    """Create a FeatureSetView with negative indices (padded)."""
    return FeatureSetView(
        source=simple_featureset,
        indices=np.array([0, -1, 2, -1, 4]),
        columns=simple_featureset.get_all_keys(
            include_domain_prefix=True,
            include_rep_suffix=True,
        ),
    )


# ---------------------------------------------------------------------
# Basic property tests
# ---------------------------------------------------------------------
@pytest.mark.unit
def test_n_samples_featureset(simple_featureset):
    """Test n_samples property on FeatureSet."""
    assert simple_featureset.n_samples == 10


@pytest.mark.unit
def test_n_samples_view(featureset_view, subset_view):
    """Test n_samples property on FeatureSetView."""
    assert featureset_view.n_samples == 10
    assert subset_view.n_samples == 5


@pytest.mark.unit
def test_len_featureset(simple_featureset):
    """Test __len__ on FeatureSet."""
    assert len(simple_featureset) == 10


@pytest.mark.unit
def test_len_view(featureset_view, subset_view):
    """Test __len__ on FeatureSetView."""
    assert len(featureset_view) == 10
    assert len(subset_view) == 5


@pytest.mark.unit
def test_sample_mask_no_padding(featureset_view):
    """Test sample_mask when no padding exists."""
    mask = featureset_view.sample_mask
    assert mask is None or np.all(mask == 1)


@pytest.mark.unit
def test_sample_mask_with_padding(padded_view):
    """Test sample_mask with padded indices."""
    mask = padded_view.sample_mask
    assert mask is not None
    assert mask[0] == 1
    assert mask[1] == 0
    assert mask[2] == 1
    assert mask[3] == 0
    assert mask[4] == 1


# ---------------------------------------------------------------------
# Key accessor tests
# ---------------------------------------------------------------------
@pytest.mark.unit
def test_get_feature_keys(simple_featureset):
    """Test get_feature_keys method."""
    keys = simple_featureset.get_feature_keys(
        include_domain_prefix=False,
        include_rep_suffix=False,
    )
    assert "X1" in keys
    assert "X2" in keys


@pytest.mark.unit
def test_get_feature_keys_with_prefix(simple_featureset):
    """Test get_feature_keys with domain prefix."""
    keys = simple_featureset.get_feature_keys(
        include_domain_prefix=True,
        include_rep_suffix=False,
    )
    assert all(k.startswith("features.") for k in keys)


@pytest.mark.unit
def test_get_feature_keys_with_suffix(simple_featureset):
    """Test get_feature_keys with rep suffix."""
    keys = simple_featureset.get_feature_keys(
        include_domain_prefix=False,
        include_rep_suffix=True,
    )
    assert all("." in k for k in keys)


@pytest.mark.unit
def test_get_target_keys(simple_featureset):
    """Test get_target_keys method."""
    keys = simple_featureset.get_target_keys(
        include_domain_prefix=False,
        include_rep_suffix=False,
    )
    assert "Y1" in keys
    assert "Y2" in keys


@pytest.mark.unit
def test_get_tag_keys(simple_featureset):
    """Test get_tag_keys method."""
    keys = simple_featureset.get_tag_keys(
        include_domain_prefix=False,
        include_rep_suffix=False,
    )
    assert "group" in keys
    assert "value" in keys


@pytest.mark.unit
def test_get_all_keys(simple_featureset):
    """Test get_all_keys method."""
    keys = simple_featureset.get_all_keys(
        include_domain_prefix=True,
        include_rep_suffix=True,
    )
    assert any("features" in k for k in keys)
    assert any("targets" in k for k in keys)
    assert any("tags" in k for k in keys)


# ---------------------------------------------------------------------
# Shape/dtype accessor tests
# ---------------------------------------------------------------------
@pytest.mark.unit
def test_get_feature_shapes(simple_featureset):
    """Test get_feature_shapes method."""
    shapes = simple_featureset.get_feature_shapes(
        include_domain_prefix=False,
        include_rep_suffix=False,
    )
    assert "X1" in shapes
    assert shapes["X1"] == (5,)
    assert "X2" in shapes
    assert shapes["X2"] == (3,)


@pytest.mark.unit
def test_get_target_shapes(simple_featureset):
    """Test get_target_shapes method."""
    shapes = simple_featureset.get_target_shapes(
        include_domain_prefix=False,
        include_rep_suffix=False,
    )
    assert "Y1" in shapes
    # Shape (1,) targets get squeezed to scalar () by pyarrow storage
    assert shapes["Y1"] in ((1,), ())


@pytest.mark.unit
def test_get_tag_shapes(simple_featureset):
    """Test get_tag_shapes method."""
    shapes = simple_featureset.get_tag_shapes(
        include_domain_prefix=False,
        include_rep_suffix=False,
    )
    assert "group" in shapes or "value" in shapes


@pytest.mark.unit
def test_get_feature_dtypes(simple_featureset):
    """Test get_feature_dtypes method."""
    dtypes = simple_featureset.get_feature_dtypes(
        include_domain_prefix=False,
        include_rep_suffix=False,
    )
    assert "X1" in dtypes
    assert isinstance(dtypes["X1"], str)


@pytest.mark.unit
def test_get_target_dtypes(simple_featureset):
    """Test get_target_dtypes method."""
    dtypes = simple_featureset.get_target_dtypes(
        include_domain_prefix=False,
        include_rep_suffix=False,
    )
    assert "Y1" in dtypes


@pytest.mark.unit
def test_get_tag_dtypes(simple_featureset):
    """Test get_tag_dtypes method."""
    dtypes = simple_featureset.get_tag_dtypes(
        include_domain_prefix=False,
        include_rep_suffix=False,
    )
    assert "group" in dtypes or "value" in dtypes


# ---------------------------------------------------------------------
# Data access tests
# ---------------------------------------------------------------------
@pytest.mark.unit
def test_get_features_dict_numpy(simple_featureset):
    """Test get_features returns dict of numpy arrays."""
    features = simple_featureset.get_features(fmt=DataFormat.DICT_NUMPY)
    assert isinstance(features, dict)
    assert "X1" in features
    assert isinstance(features["X1"], np.ndarray)


@pytest.mark.unit
def test_get_features_numpy():
    """Test get_features returns stacked numpy array (requires uniform shapes)."""
    fs = generate_dummy_featureset(
        n_samples=10,
        feature_shape_map={"X": (5,)},
        target_shape_map={"Y": (1,)},
        tag_type_map={},
        label="UniformFS",
    )
    features = fs.get_features(fmt=DataFormat.NUMPY)
    assert isinstance(features, np.ndarray)
    assert features.shape[0] == 10


@pytest.mark.unit
def test_get_targets_dict_numpy(simple_featureset):
    """Test get_targets returns dict of numpy arrays."""
    targets = simple_featureset.get_targets(fmt=DataFormat.DICT_NUMPY)
    assert isinstance(targets, dict)
    assert "Y1" in targets


@pytest.mark.unit
def test_get_tags_dict_numpy(simple_featureset):
    """Test get_tags returns dict of numpy arrays."""
    tags = simple_featureset.get_tags(fmt=DataFormat.DICT_NUMPY)
    assert isinstance(tags, dict)


@pytest.mark.unit
def test_get_data_with_feature_selector(simple_featureset):
    """Test get_data with feature selector."""
    data = simple_featureset.get_data(
        features="X1",
        fmt=DataFormat.DICT_NUMPY,
    )
    assert isinstance(data, dict)


# ---------------------------------------------------------------------
# Sample UUID access tests
# ---------------------------------------------------------------------
@pytest.mark.unit
def test_get_sample_uuids_numpy(simple_featureset):
    """Test get_sample_uuids returns numpy array."""
    uuids = simple_featureset.get_sample_uuids(fmt=DataFormat.NUMPY)
    assert isinstance(uuids, np.ndarray)
    assert len(uuids) == 10
    assert len(set(uuids)) == len(uuids)


# ---------------------------------------------------------------------
# Export tests
# ---------------------------------------------------------------------
@pytest.mark.unit
def test_to_arrow(simple_featureset):
    """Test to_arrow conversion."""
    table = simple_featureset.to_arrow()
    assert isinstance(table, pa.Table)
    assert table.num_rows == 10


@pytest.mark.unit
def test_to_pandas(simple_featureset):
    """Test to_pandas conversion."""
    df = simple_featureset.to_pandas()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 10


@pytest.mark.unit
def test_to_sample_collection(simple_featureset):
    """Test to_sample_collection conversion."""
    coll = simple_featureset.to_sample_collection()
    assert isinstance(coll, SampleCollection)
    assert coll.n_samples == 10


@pytest.mark.unit
def test_to_featureset(featureset_view):
    """Test to_featureset conversion."""
    fs = featureset_view.to_featureset(label="NewFS")
    assert isinstance(fs, FeatureSet)
    assert fs.label == "NewFS"
    assert fs.n_samples == 10


@pytest.mark.unit
def test_to_sample_data():
    """Test to_sample_data conversion (requires uniform shapes)."""
    fs = generate_dummy_featureset(
        n_samples=10,
        feature_shape_map={"X": (5,)},
        target_shape_map={"Y": (1,)},
        tag_type_map={},
        label="UniformFS2",
    )
    sd = fs.to_sample_data(fmt=DataFormat.NUMPY)
    assert isinstance(sd, SampleData)
    assert sd.features is not None
    assert sd.targets is not None


@pytest.mark.unit
def test_to_sample_data_non_tensorlike_raises(simple_featureset):
    """Test to_sample_data raises for non-tensorlike format."""
    with pytest.raises(ValueError, match="tensor-like"):
        simple_featureset.to_sample_data(fmt=DataFormat.DICT_NUMPY)
