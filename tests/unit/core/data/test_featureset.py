"""Unit tests for modularml.core.data.featureset module."""

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from modularml.core.data.featureset import FeatureSet
from modularml.core.data.sample_collection import SampleCollection
from modularml.core.data.schema_constants import DOMAIN_SAMPLE_UUIDS
from modularml.utils.data.data_format import DataFormat
from modularml.utils.data.pyarrow_data import build_sample_schema_table
from tests.conftest import generate_dummy_featureset


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------
@pytest.fixture
def simple_featureset():
    """Create a simple FeatureSet for testing."""
    return generate_dummy_featureset(
        n_samples=20,
        feature_shape_map={"X1": (5,), "X2": (3,)},
        target_shape_map={"Y1": (1,)},
        tag_type_map={"group": "str"},
        label="TestFS",
    )


@pytest.fixture
def minimal_featureset():
    """Create a minimal FeatureSet without tags."""
    return generate_dummy_featureset(
        n_samples=10,
        feature_shape_map={"X": (4,)},
        target_shape_map={"Y": (1,)},
        tag_type_map={},
        label="MinFS",
    )


# ---------------------------------------------------------------------
# Constructor tests
# ---------------------------------------------------------------------
@pytest.mark.unit
def test_featureset_init(simple_featureset):
    """Test basic FeatureSet initialization."""
    assert isinstance(simple_featureset, FeatureSet)
    assert simple_featureset.label == "TestFS"
    assert simple_featureset.n_samples == 20


@pytest.mark.unit
def test_featureset_init_non_collection_raises():
    """Test FeatureSet raises for non-SampleCollection."""
    with pytest.raises(TypeError, match="SampleCollection"):
        FeatureSet(label="bad", collection="not a collection")


@pytest.mark.unit
def test_from_dict():
    """Test FeatureSet.from_dict constructor."""
    fs = FeatureSet.from_dict(
        label="DictFS",
        data={
            "voltage": np.random.rand(5, 4),  # noqa: NPY002
            "soh": np.random.rand(5, 1),  # noqa: NPY002
            "cell_id": np.array(["A", "B", "C", "D", "E"]).reshape(5, 1),
        },
        feature_keys=["voltage"],
        target_keys=["soh"],
        tag_keys=["cell_id"],
    )
    assert isinstance(fs, FeatureSet)
    assert fs.label == "DictFS"
    assert fs.n_samples == 5


@pytest.mark.unit
def test_from_pyarrow_table():
    """Test FeatureSet.from_pyarrow_table constructor."""
    n = 5
    features = {"f": np.random.rand(n, 3)}  # noqa: NPY002
    targets = {"y": np.random.rand(n, 1)}  # noqa: NPY002
    table = build_sample_schema_table(features=features, targets=targets, tags=None)
    fs = FeatureSet.from_pyarrow_table(label="ArrowFS", table=table)
    assert isinstance(fs, FeatureSet)
    assert fs.n_samples == n


# ---------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------
@pytest.mark.unit
def test_featureset_n_samples(simple_featureset):
    """Test n_samples property."""
    assert simple_featureset.n_samples == 20


@pytest.mark.unit
def test_featureset_collection(simple_featureset):
    """Test collection attribute."""
    assert isinstance(simple_featureset.collection, SampleCollection)


@pytest.mark.unit
def test_featureset_label(simple_featureset):
    """Test label property."""
    assert simple_featureset.label == "TestFS"


# ---------------------------------------------------------------------
# Key accessor tests (via SampleCollectionMixin)
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
def test_get_target_keys(simple_featureset):
    """Test get_target_keys method."""
    keys = simple_featureset.get_target_keys(
        include_domain_prefix=False,
        include_rep_suffix=False,
    )
    assert "Y1" in keys


@pytest.mark.unit
def test_get_tag_keys(simple_featureset):
    """Test get_tag_keys method."""
    keys = simple_featureset.get_tag_keys(
        include_domain_prefix=False,
        include_rep_suffix=False,
    )
    assert "group" in keys


@pytest.mark.unit
def test_get_all_keys(simple_featureset):
    """Test get_all_keys method."""
    keys = simple_featureset.get_all_keys(
        include_domain_prefix=True,
        include_rep_suffix=True,
    )
    assert any("features" in k for k in keys)
    assert any("targets" in k for k in keys)
    assert DOMAIN_SAMPLE_UUIDS in keys


# ---------------------------------------------------------------------
# Data accessor tests
# ---------------------------------------------------------------------
@pytest.mark.unit
def test_get_features(simple_featureset):
    """Test get_features method."""
    features = simple_featureset.get_features(fmt=DataFormat.DICT_NUMPY)
    assert isinstance(features, dict)
    assert "X1" in features
    assert features["X1"].shape[0] == 20


@pytest.mark.unit
def test_get_targets(simple_featureset):
    """Test get_targets method."""
    targets = simple_featureset.get_targets(fmt=DataFormat.DICT_NUMPY)
    assert isinstance(targets, dict)
    assert "Y1" in targets


@pytest.mark.unit
def test_get_tags(simple_featureset):
    """Test get_tags method."""
    tags = simple_featureset.get_tags(fmt=DataFormat.DICT_NUMPY)
    assert isinstance(tags, dict)


@pytest.mark.unit
def test_get_features_numpy(minimal_featureset):
    """Test get_features with NUMPY format (requires uniform shapes)."""
    features = minimal_featureset.get_features(fmt=DataFormat.NUMPY)
    assert isinstance(features, np.ndarray)
    assert features.shape[0] == 10


# ---------------------------------------------------------------------
# Shape/dtype tests
# ---------------------------------------------------------------------
@pytest.mark.unit
def test_get_feature_shapes(simple_featureset):
    """Test get_feature_shapes method."""
    shapes = simple_featureset.get_feature_shapes(
        include_domain_prefix=False,
        include_rep_suffix=False,
    )
    assert shapes["X1"] == (5,)
    assert shapes["X2"] == (3,)


@pytest.mark.unit
def test_get_feature_dtypes(simple_featureset):
    """Test get_feature_dtypes method."""
    dtypes = simple_featureset.get_feature_dtypes(
        include_domain_prefix=False,
        include_rep_suffix=False,
    )
    assert "X1" in dtypes
    assert "float" in dtypes["X1"]


# ---------------------------------------------------------------------
# Copy test
# ---------------------------------------------------------------------
@pytest.mark.unit
def test_copy(simple_featureset):
    """Test copy method."""
    copy = simple_featureset.copy()
    assert isinstance(copy, FeatureSet)
    assert copy is not simple_featureset
    assert copy.n_samples == simple_featureset.n_samples
    assert copy.label == simple_featureset.label + "_copy"


# ---------------------------------------------------------------------
# Export tests
# ---------------------------------------------------------------------
@pytest.mark.unit
def test_to_arrow(simple_featureset):
    """Test to_arrow method."""
    table = simple_featureset.to_arrow()
    assert isinstance(table, pa.Table)
    assert table.num_rows == 20


@pytest.mark.unit
def test_to_pandas(simple_featureset):
    """Test to_pandas method."""
    df = simple_featureset.to_pandas()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 20


# ---------------------------------------------------------------------
# Serialization tests
# ---------------------------------------------------------------------
@pytest.mark.unit
def test_save_and_load(tmp_path, simple_featureset):
    """Test save and load roundtrip."""
    path = simple_featureset.save(tmp_path / "test_fs")
    loaded = FeatureSet.load(path)
    assert isinstance(loaded, FeatureSet)
    assert loaded.n_samples == simple_featureset.n_samples
    assert loaded.label == simple_featureset.label


# ---------------------------------------------------------------------
# Equality test
# ---------------------------------------------------------------------
@pytest.mark.unit
def test_equality(simple_featureset):
    """Test FeatureSet equality is based on data content."""
    copy = simple_featureset.copy()
    # Same data, should compare as equal via SampleCollection
    assert simple_featureset.collection == copy.collection
