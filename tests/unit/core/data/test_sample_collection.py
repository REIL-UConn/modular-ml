"""Unit tests for modularml.core.data.sample_collection module."""

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from modularml.core.data.sample_collection import SampleCollection
from modularml.core.data.sample_schema import (
    METADATA_SCHEMA_VERSION_KEY,
    SCHEMA_VERSION,
)
from modularml.core.data.schema_constants import (
    DOMAIN_FEATURES,
    DOMAIN_SAMPLE_UUIDS,
)
from modularml.utils.data.data_format import DataFormat
from modularml.utils.data.pyarrow_data import build_sample_schema_table
from tests.conftest import generate_dummy_data


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------
@pytest.fixture
def simple_pyarrow_table():
    """Builds a small but valid Arrow table with features, targets, and tags."""
    n = 5
    features = {
        "voltage": generate_dummy_data((n, 4), "float"),
        "current": generate_dummy_data((n, 4), "float"),
    }
    targets = {"soh": generate_dummy_data((n, 1), "float")}
    tags = {
        "cell_id": generate_dummy_data(
            shape=(n, 1),
            dtype="str",
            choices=["A", "B", "C", "D", "E"],
        ),
    }
    return build_sample_schema_table(features=features, targets=targets, tags=tags)


@pytest.fixture
def sample_collection(simple_pyarrow_table) -> SampleCollection:
    """Instantiate a SampleCollection from a valid table."""
    return SampleCollection(simple_pyarrow_table)


# ---------------------------------------------------------------------
# Basic initialization tests
# ---------------------------------------------------------------------
@pytest.mark.unit
def test_init_and_metadata(sample_collection: SampleCollection):
    """Validate initialization, schema inference, and metadata embedding."""
    coll = sample_collection
    meta = coll.table.schema.metadata
    assert isinstance(coll, SampleCollection)
    assert coll.n_samples == 5

    # Metadata version
    assert METADATA_SCHEMA_VERSION_KEY.encode() in meta
    assert meta[METADATA_SCHEMA_VERSION_KEY.encode()].decode() == SCHEMA_VERSION

    # Check UUID column
    assert DOMAIN_SAMPLE_UUIDS in coll.table.column_names
    ids = coll.table[DOMAIN_SAMPLE_UUIDS].to_pylist()
    assert len(set(ids)) == len(ids) == coll.n_samples


@pytest.mark.unit
def test_optional_tags_are_supported():
    """Ensure SampleCollection works even with no tags."""
    n = 5
    features = {"f": generate_dummy_data((n, 2), dtype="float")}
    targets = {"y": generate_dummy_data((n, 1), dtype="float")}
    table_no_tags = build_sample_schema_table(
        features=features,
        targets=targets,
        tags=None,
    )
    coll = SampleCollection(table_no_tags)
    assert not coll.has_tags
    assert coll.get_tag_keys() == []


# ---------------------------------------------------------------------
# Accessor methods
# ---------------------------------------------------------------------
@pytest.mark.unit
def test_get_features_targets_tags(sample_collection: SampleCollection):
    """Ensure get_* methods return proper numpy data."""
    feats = sample_collection.get_features(
        fmt=DataFormat.DICT_NUMPY,
        include_domain_prefix=False,
        include_rep_suffix=False,
    )
    assert "voltage" in feats

    feats = sample_collection.get_features(
        fmt=DataFormat.DICT_NUMPY,
        include_domain_prefix=False,
        include_rep_suffix=True,
    )
    assert "voltage.raw" in feats

    v = feats["voltage.raw"]
    assert isinstance(v, np.ndarray)
    assert v.shape == (sample_collection.n_samples, 4)

    targs = sample_collection.get_targets(
        fmt=DataFormat.DICT_NUMPY,
        include_domain_prefix=False,
        include_rep_suffix=True,
    )
    assert "soh.raw" in targs
    y = targs["soh.raw"]
    assert y.shape == (sample_collection.n_samples,)

    tags = sample_collection.get_tags(
        fmt=DataFormat.DICT_NUMPY,
        include_domain_prefix=False,
        include_rep_suffix=True,
    )
    assert "cell_id.raw" in tags


@pytest.mark.unit
def test_get_rep_data_numpy(sample_collection: SampleCollection):
    """Directly retrieve a single representation as NumPy array."""
    arr = sample_collection._get_rep_data(
        domain="features",
        key="voltage",
        rep="raw",
        fmt=DataFormat.NUMPY,
    )
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (sample_collection.n_samples, 4)


@pytest.mark.unit
def test_domain_accessors_are_valid(sample_collection: SampleCollection):
    """Domain keys, dtypes, and shapes should be consistent."""
    fkeys = sample_collection.get_feature_keys()
    tkeys = sample_collection.get_target_keys()
    assert isinstance(fkeys, list)
    assert all(isinstance(k, str) for k in fkeys)

    fshapes = sample_collection.get_feature_shapes()
    tdtypes = sample_collection.get_target_dtypes()
    for shape in fshapes.values():
        assert isinstance(shape, tuple)
        assert all(isinstance(x, int) for x in shape)
    for dtype in tdtypes.values():
        assert isinstance(dtype, str)


@pytest.mark.unit
def test_get_feature_keys_with_prefix(sample_collection: SampleCollection):
    """Test get_feature_keys with domain prefix."""
    keys = sample_collection.get_feature_keys(
        include_domain_prefix=True,
        include_rep_suffix=False,
    )
    assert all(k.startswith("features.") for k in keys)


@pytest.mark.unit
def test_get_all_keys(sample_collection: SampleCollection):
    """Test get_all_keys method."""
    keys = sample_collection.get_all_keys(
        include_domain_prefix=True,
        include_rep_suffix=True,
    )
    assert any("features" in k for k in keys)
    assert any("targets" in k for k in keys)
    assert DOMAIN_SAMPLE_UUIDS in keys


@pytest.mark.unit
def test_get_sample_uuids(sample_collection: SampleCollection):
    """Test get_sample_uuids returns unique IDs."""
    uuids = sample_collection.get_sample_uuids(fmt=DataFormat.NUMPY)
    assert len(uuids) == sample_collection.n_samples
    assert len(set(uuids)) == len(uuids)


# ---------------------------------------------------------------------
# Binary tensor decoding
# ---------------------------------------------------------------------
@pytest.mark.unit
def test_binary_tensor_decoding():
    """Check high-rank tensors (>2D) encoded as binary are decoded correctly."""
    n = 3
    data = generate_dummy_data((n, 8, 8, 3), "float")
    features = {"image": data}
    targets = {"label": generate_dummy_data((n, 1), "float")}
    table = build_sample_schema_table(features=features, targets=targets, tags=None)
    coll = SampleCollection(table)
    arr = coll._get_rep_data("features", "image", "raw", fmt=DataFormat.NUMPY)
    assert arr.shape == data.shape
    assert np.allclose(arr, data, atol=1e-6)


# ---------------------------------------------------------------------
# Export and flattening
# ---------------------------------------------------------------------
@pytest.mark.unit
def test_to_dict_and_to_pandas(sample_collection: SampleCollection):
    """Ensure flattened dict and DataFrame conversions work."""
    dct = sample_collection.to_dict()
    assert isinstance(dct, dict)
    assert any(k.startswith("features.voltage") for k in dct)

    df = sample_collection.to_pandas()
    assert isinstance(df, pd.DataFrame)
    assert DOMAIN_FEATURES in {c.split(".")[0] for c in df.columns}
    assert df.shape[0] == sample_collection.n_samples


# ---------------------------------------------------------------------
# Save/load roundtrip
# ---------------------------------------------------------------------
@pytest.mark.unit
def test_arrow_save_load_roundtrip(tmp_path, sample_collection: SampleCollection):
    """Confirm Arrow IPC roundtrip retains metadata and schema."""
    path = sample_collection.save(tmp_path / "collection")
    coll2 = SampleCollection.load(path)
    assert coll2.table_version == SCHEMA_VERSION
    assert coll2.n_samples == sample_collection.n_samples


# ---------------------------------------------------------------------
# Shape + dtype consistency
# ---------------------------------------------------------------------
@pytest.mark.unit
def test_shape_and_dtype_from_metadata(sample_collection: SampleCollection):
    """Ensure metadata-driven shape/dtype retrieval works."""
    shape = sample_collection._get_rep_shape("features", "voltage", "raw")
    dtype = sample_collection._get_rep_dtype("features", "voltage", "raw")
    assert isinstance(shape, tuple)
    assert isinstance(dtype, str)
    assert "float" in dtype


# ---------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------
@pytest.mark.unit
def test_missing_required_domains():
    """Table missing features/targets should raise."""
    bad_table = pa.table({"bad": pa.array([1, 2, 3])})
    with pytest.raises(ValueError, match="Invalid column 'bad'"):
        SampleCollection(bad_table)


# ---------------------------------------------------------------------
# Representation mutation
# ---------------------------------------------------------------------
@pytest.mark.unit
def test_add_overwrite_delete_rep(sample_collection: SampleCollection):
    """Test add_rep, overwrite, and delete_rep workflow."""
    # Add transformed feature rep
    scaled_data = sample_collection._get_rep_data(
        domain="features",
        key="voltage",
        rep="raw",
        fmt=DataFormat.NUMPY,
    )
    scaled_data = scaled_data * 0.5
    sample_collection.add_rep(
        domain="features",
        key="voltage",
        rep="transformed",
        data=scaled_data,
        overwrite=False,
    )

    # Confirm it exists
    existing_reps = sample_collection._get_rep_keys(
        domain="features",
        key="voltage",
    )
    assert "transformed" in existing_reps
    assert "raw" in existing_reps

    # Overwrite
    zero_data = scaled_data * 0
    sample_collection.add_rep(
        domain="features",
        key="voltage",
        rep="transformed",
        data=zero_data,
        overwrite=True,
    )
    new_data = sample_collection._get_rep_data(
        domain="features",
        key="voltage",
        rep="transformed",
        fmt=DataFormat.NUMPY,
    )
    assert np.all(new_data.ravel() == 0)

    # Delete it
    sample_collection.delete_rep(
        domain="features",
        key="voltage",
        rep="transformed",
    )
    existing_reps = sample_collection._get_rep_keys(
        domain="features",
        key="voltage",
    )
    assert "transformed" not in existing_reps
    assert "raw" in existing_reps

    # Try deleting "raw"
    with pytest.raises(ValueError, match="cannot be deleted"):
        sample_collection.delete_rep(
            domain="features",
            key="voltage",
            rep="raw",
        )


@pytest.mark.unit
def test_add_rep_raw_raises(sample_collection: SampleCollection):
    """Test that adding 'raw' representation raises."""
    data = np.ones((sample_collection.n_samples, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="cannot be modified"):
        sample_collection.add_rep(
            domain="features",
            key="voltage",
            rep="raw",
            data=data,
        )


@pytest.mark.unit
def test_add_rep_wrong_samples_raises(sample_collection: SampleCollection):
    """Test add_rep raises for wrong number of samples."""
    data = np.ones((999, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="does not match"):
        sample_collection.add_rep(
            domain="features",
            key="voltage",
            rep="transformed",
            data=data,
        )


@pytest.mark.unit
def test_add_rep_non_numpy_raises(sample_collection: SampleCollection):
    """Test add_rep raises for non-numpy data."""
    with pytest.raises(TypeError, match="numpy array"):
        sample_collection.add_rep(
            domain="features",
            key="voltage",
            rep="transformed",
            data=[[1.0, 2.0]],
        )


# ---------------------------------------------------------------------
# Copy tests
# ---------------------------------------------------------------------
@pytest.mark.unit
def test_copy(sample_collection: SampleCollection):
    """Test copy method."""
    copy = sample_collection.copy()
    assert copy is not sample_collection
    assert copy.n_samples == sample_collection.n_samples


@pytest.mark.unit
def test_copy_raw_only(sample_collection: SampleCollection):
    """Test copy with raw_only=True."""
    # First add a transformed representation
    data = np.ones((sample_collection.n_samples, 4), dtype=np.float32)
    sample_collection.add_rep(
        domain="features",
        key="voltage",
        rep="transformed",
        data=data,
    )

    copy = sample_collection.copy(raw_only=True)
    # Should not contain transformed rep
    reps = copy._get_rep_keys(domain="features", key="voltage")
    assert "raw" in reps
    assert "transformed" not in reps


# ---------------------------------------------------------------------
# Equality tests
# ---------------------------------------------------------------------
@pytest.mark.unit
def test_equality_type_error(sample_collection: SampleCollection):
    """Test __eq__ raises for non-SampleCollection."""
    with pytest.raises(TypeError, match="Cannot compare"):
        _ = sample_collection == "not a collection"
