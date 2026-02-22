"""Unit tests for modularml.core.data.sample_data module."""

import numpy as np
import pytest

from modularml.core.data.sample_data import RoleData, SampleData, SampleShapes
from modularml.core.data.schema_constants import (
    DOMAIN_FEATURES,
    DOMAIN_TARGETS,
)
from modularml.utils.data.data_format import DataFormat


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------
@pytest.fixture
def simple_sample_data():
    """Create a simple SampleData with features and targets."""
    features = np.random.rand(4, 10).astype(np.float32)  # noqa: NPY002
    targets = np.random.rand(4, 1).astype(np.float32)  # noqa: NPY002
    return SampleData(features=features, targets=targets)


@pytest.fixture
def full_sample_data():
    """Create SampleData with all domains."""
    features = np.random.rand(4, 10).astype(np.float32)  # noqa: NPY002
    targets = np.random.rand(4, 1).astype(np.float32)  # noqa: NPY002
    tags = np.array(["A", "B", "C", "D"]).reshape(4, 1)
    uuids = np.array(["id1", "id2", "id3", "id4"]).reshape(4, 1)
    return SampleData(
        features=features,
        targets=targets,
        tags=tags,
        sample_uuids=uuids,
    )


@pytest.fixture
def output_sample_data():
    """Create SampleData with kind='output'."""
    features = np.random.rand(4, 5).astype(np.float32)  # noqa: NPY002
    return SampleData(features=features, kind="output")


# ---------------------------------------------------------------------
# SampleData initialization tests
# ---------------------------------------------------------------------
@pytest.mark.unit
def test_sample_data_init_with_kwargs(simple_sample_data):
    """Test SampleData initialization with keyword arguments."""
    assert simple_sample_data.features is not None
    assert simple_sample_data.targets is not None
    assert simple_sample_data.tags is None
    assert simple_sample_data.n_samples == 4


@pytest.mark.unit
def test_sample_data_init_with_dict():
    """Test SampleData initialization with data dict."""
    features = np.random.rand(4, 10)  # noqa: NPY002
    sd = SampleData(data={DOMAIN_FEATURES: features})
    assert sd.features is not None
    assert sd.targets is None


@pytest.mark.unit
def test_sample_data_shapes(simple_sample_data):
    """Test that shapes are correctly computed."""
    shapes = simple_sample_data.shapes
    assert isinstance(shapes, SampleShapes)
    assert shapes.features_shape == (10,)
    assert shapes.targets_shape == (1,)


@pytest.mark.unit
def test_sample_data_n_samples(simple_sample_data):
    """Test n_samples property."""
    assert simple_sample_data.n_samples == 4


# ---------------------------------------------------------------------
# SampleData domain access
# ---------------------------------------------------------------------
@pytest.mark.unit
def test_get_domain_data_features(simple_sample_data):
    """Test get_domain_data for features."""
    features = simple_sample_data.get_domain_data("features")
    assert features is not None
    assert features.shape == (4, 10)


@pytest.mark.unit
def test_get_domain_data_invalid_raises(simple_sample_data):
    """Test get_domain_data raises for invalid domain."""
    with pytest.raises(KeyError, match="Invalid domain"):
        simple_sample_data.get_domain_data("invalid")


@pytest.mark.unit
def test_outputs_property_input_raises(simple_sample_data):
    """Test outputs property raises for input kind."""
    with pytest.raises(AttributeError, match="only defined for"):
        _ = simple_sample_data.outputs


@pytest.mark.unit
def test_outputs_property_output_kind(output_sample_data):
    """Test outputs property works for output kind."""
    outputs = output_sample_data.outputs
    assert outputs is not None
    assert outputs.shape == (4, 5)


# ---------------------------------------------------------------------
# SampleData format conversion
# ---------------------------------------------------------------------
@pytest.mark.unit
def test_as_format_numpy(simple_sample_data):
    """Test in-place format conversion to numpy."""
    simple_sample_data.as_format(DataFormat.NUMPY)
    assert isinstance(simple_sample_data.features, np.ndarray)


@pytest.mark.unit
def test_to_format_numpy(simple_sample_data):
    """Test non-mutating format conversion returns new instance."""
    new_sd = simple_sample_data.to_format(DataFormat.NUMPY)
    assert new_sd is not simple_sample_data
    assert isinstance(new_sd.features, np.ndarray)


@pytest.mark.unit
def test_as_format_non_tensorlike_raises(simple_sample_data):
    """Test as_format raises for non-tensorlike format."""
    with pytest.raises(ValueError, match="tensor-like"):
        simple_sample_data.as_format(DataFormat.DICT_NUMPY)


# ---------------------------------------------------------------------
# SampleData concatenation
# ---------------------------------------------------------------------
@pytest.mark.unit
def test_concat_two_sample_data():
    """Test concatenating two SampleData instances."""
    sd1 = SampleData(features=np.ones((2, 5)), targets=np.ones((2, 1)))
    sd2 = SampleData(features=np.zeros((3, 5)), targets=np.zeros((3, 1)))
    result = SampleData.concat(sd1, sd2)
    assert result.features.shape == (5, 5)
    assert result.targets.shape == (5, 1)


@pytest.mark.unit
def test_concat_with(simple_sample_data):
    """Test concat_with method."""
    other = SampleData(
        features=np.random.rand(2, 10).astype(np.float32),  # noqa: NPY002
        targets=np.random.rand(2, 1).astype(np.float32),  # noqa: NPY002
    )
    result = simple_sample_data.concat_with(other)
    assert result.n_samples == 6


@pytest.mark.unit
def test_concat_type_error():
    """Test concat raises TypeError for non-SampleData."""
    sd = SampleData(features=np.ones((2, 5)))
    with pytest.raises(TypeError, match="Expected SampleData"):
        SampleData.concat(sd, "not a SampleData")


@pytest.mark.unit
def test_concat_mismatched_kinds_raises():
    """Test concat raises for mismatched kinds."""
    sd1 = SampleData(features=np.ones((2, 5)), kind="input")
    sd2 = SampleData(features=np.ones((2, 5)), kind="output")
    with pytest.raises(ValueError, match="differing kinds"):
        SampleData.concat(sd1, sd2)


@pytest.mark.unit
def test_concat_mismatched_shapes_raises():
    """Test concat raises for mismatched shapes."""
    sd1 = SampleData(features=np.ones((2, 5)), targets=np.ones((2, 1)))
    sd2 = SampleData(features=np.ones((2, 10)), targets=np.ones((2, 1)))
    with pytest.raises(ValueError, match="different domain shapes"):
        SampleData.concat(sd1, sd2)


# ---------------------------------------------------------------------
# SampleData repr
# ---------------------------------------------------------------------
@pytest.mark.unit
def test_repr(simple_sample_data):
    """Test __repr__ method."""
    repr_str = repr(simple_sample_data)
    assert "SampleData" in repr_str
    assert "n_samples=4" in repr_str


# ---------------------------------------------------------------------
# RoleData tests
# ---------------------------------------------------------------------
@pytest.mark.unit
def test_role_data_init():
    """Test RoleData initialization."""
    sd = SampleData(features=np.ones((4, 5)), targets=np.ones((4, 1)))
    rd = RoleData(data={"default": sd})
    assert rd.available_roles == ["default"]
    assert len(rd) == 1


@pytest.mark.unit
def test_role_data_empty_raises():
    """Test RoleData raises for empty data."""
    with pytest.raises(ValueError, match="at least one role"):
        RoleData(data={})


@pytest.mark.unit
def test_role_data_non_str_key_raises():
    """Test RoleData raises for non-string keys."""
    sd = SampleData(features=np.ones((4, 5)))
    with pytest.raises(TypeError, match="str"):
        RoleData(data={123: sd})


@pytest.mark.unit
def test_role_data_non_sample_data_value_raises():
    """Test RoleData raises for non-SampleData values."""
    with pytest.raises(TypeError, match="SampleData"):
        RoleData(data={"default": "not SampleData"})


@pytest.mark.unit
def test_role_data_getitem():
    """Test RoleData __getitem__."""
    sd = SampleData(features=np.ones((4, 5)))
    rd = RoleData(data={"default": sd})
    assert rd["default"] is sd


@pytest.mark.unit
def test_role_data_iter():
    """Test RoleData __iter__."""
    sd = SampleData(features=np.ones((4, 5)))
    rd = RoleData(data={"default": sd, "anchor": sd})
    roles = list(rd)
    assert "default" in roles
    assert "anchor" in roles


@pytest.mark.unit
def test_role_data_single_role_features():
    """Test single-role pass-through for features."""
    sd = SampleData(features=np.ones((4, 5)), targets=np.ones((4, 1)))
    rd = RoleData(data={"default": sd})
    assert rd.features is not None
    assert rd.features.shape == (4, 5)


@pytest.mark.unit
def test_role_data_multi_role_features_raises():
    """Test multi-role features raises RuntimeError."""
    sd = SampleData(features=np.ones((4, 5)))
    rd = RoleData(data={"anchor": sd, "pair": sd})
    with pytest.raises(RuntimeError, match="exactly one role"):
        _ = rd.features


@pytest.mark.unit
def test_role_data_get_data():
    """Test RoleData.get_data method."""
    sd = SampleData(features=np.ones((4, 5)), targets=np.ones((4, 1)))
    rd = RoleData(data={"default": sd})
    result = rd.get_data(role="default")
    assert isinstance(result, SampleData)


@pytest.mark.unit
def test_role_data_get_data_with_domain():
    """Test RoleData.get_data with domain selection."""
    sd = SampleData(features=np.ones((4, 5)), targets=np.ones((4, 1)))
    rd = RoleData(data={"default": sd})
    result = rd.get_data(role="default", domain="features")
    assert isinstance(result, np.ndarray)
    assert result.shape == (4, 5)


@pytest.mark.unit
def test_role_data_get_data_invalid_role_raises():
    """Test RoleData.get_data raises for invalid role."""
    sd = SampleData(features=np.ones((4, 5)))
    rd = RoleData(data={"default": sd})
    with pytest.raises(KeyError, match="not found"):
        rd.get_data(role="nonexistent")


@pytest.mark.unit
def test_role_data_getattr():
    """Test attribute-style role access."""
    sd = SampleData(features=np.ones((4, 5)))
    rd = RoleData(data={"anchor": sd})
    assert rd.anchor is sd


@pytest.mark.unit
def test_role_data_getattr_invalid_raises():
    """Test attribute access raises for invalid role."""
    sd = SampleData(features=np.ones((4, 5)))
    rd = RoleData(data={"default": sd})
    with pytest.raises(AttributeError, match="no role"):
        _ = rd.nonexistent


@pytest.mark.unit
def test_role_data_copy():
    """Test RoleData copy method."""
    sd = SampleData(features=np.ones((4, 5)), targets=np.ones((4, 1)))
    rd = RoleData(data={"default": sd})
    rd_copy = rd.copy()
    assert rd_copy is not rd
    assert rd_copy.available_roles == rd.available_roles


@pytest.mark.unit
def test_role_data_concat():
    """Test RoleData.concat method."""
    sd1 = SampleData(features=np.ones((2, 5)), targets=np.ones((2, 1)))
    sd2 = SampleData(features=np.zeros((3, 5)), targets=np.zeros((3, 1)))
    rd1 = RoleData(data={"default": sd1})
    rd2 = RoleData(data={"default": sd2})
    result = RoleData.concat(rd1, rd2)
    assert result["default"].n_samples == 5


@pytest.mark.unit
def test_role_data_repr():
    """Test RoleData __repr__."""
    sd = SampleData(features=np.ones((4, 5)))
    rd = RoleData(data={"default": sd})
    assert "RoleData" in repr(rd)


# ---------------------------------------------------------------------
# SampleShapes tests
# ---------------------------------------------------------------------
@pytest.mark.unit
def test_sample_shapes_init_with_shapes():
    """Test SampleShapes initialization with dict."""
    shapes = SampleShapes(
        shapes={
            DOMAIN_FEATURES: (10,),
            DOMAIN_TARGETS: (1,),
        },
    )
    assert shapes.features_shape == (10,)
    assert shapes.targets_shape == (1,)
    assert shapes.tags_shape is None


@pytest.mark.unit
def test_sample_shapes_init_with_kwargs():
    """Test SampleShapes initialization with keyword arguments."""
    shapes = SampleShapes(
        features_shape=(10,),
        targets_shape=(1,),
    )
    assert shapes.features_shape == (10,)
    assert shapes.targets_shape == (1,)


@pytest.mark.unit
def test_sample_shapes_getitem():
    """Test SampleShapes __getitem__."""
    shapes = SampleShapes(features_shape=(10,))
    assert shapes[DOMAIN_FEATURES] == (10,)


@pytest.mark.unit
def test_sample_shapes_equality():
    """Test SampleShapes equality comparison."""
    s1 = SampleShapes(features_shape=(10,), targets_shape=(1,))
    s2 = SampleShapes(features_shape=(10,), targets_shape=(1,))
    assert s1 == s2


@pytest.mark.unit
def test_sample_shapes_inequality():
    """Test SampleShapes inequality."""
    s1 = SampleShapes(features_shape=(10,))
    s2 = SampleShapes(features_shape=(20,))
    assert s1 != s2


@pytest.mark.unit
def test_sample_shapes_eq_type_error():
    """Test SampleShapes equality raises for non-SampleShapes."""
    s = SampleShapes(features_shape=(10,))
    with pytest.raises(TypeError, match="Cannot compare"):
        _ = s == "not a shape"


@pytest.mark.unit
def test_sample_shapes_outputs_shape_input_raises():
    """Test outputs_shape raises for input kind."""
    s = SampleShapes(features_shape=(10,), kind="input")
    with pytest.raises(AttributeError, match="only defined for"):
        _ = s.outputs_shape


@pytest.mark.unit
def test_sample_shapes_outputs_shape_output():
    """Test outputs_shape works for output kind."""
    s = SampleShapes(features_shape=(10,), kind="output")
    assert s.outputs_shape == (10,)


@pytest.mark.unit
def test_sample_shapes_repr():
    """Test SampleShapes __repr__."""
    s = SampleShapes(features_shape=(10,))
    assert "SampleShapes" in repr(s)
