"""Unit tests for modularml.core.data.sample_schema module."""

import pyarrow as pa
import pytest

from modularml.core.data.sample_schema import (
    SCHEMA_VERSION,
    SampleSchema,
    ensure_sample_id,
    validate_str_list,
)
from modularml.core.data.schema_constants import (
    DOMAIN_SAMPLE_UUIDS,
)


# ---------------------------------------------------------------------
# SampleSchema initialization tests
# ---------------------------------------------------------------------
@pytest.mark.unit
def test_sample_schema_init_empty():
    """Test SampleSchema with default (empty) values."""
    schema = SampleSchema()
    assert schema.features == {}
    assert schema.targets == {}
    assert schema.tags == {}


@pytest.mark.unit
def test_sample_schema_init_with_data():
    """Test SampleSchema with actual domain mappings."""
    schema = SampleSchema(
        features={"voltage": {"raw": pa.float64()}},
        targets={"soh": {"raw": pa.float32()}},
        tags={"cell_id": {"raw": pa.string()}},
    )
    assert "voltage" in schema.features
    assert "soh" in schema.targets
    assert "cell_id" in schema.tags


@pytest.mark.unit
def test_sample_schema_duplicate_keys_raises():
    """Test that duplicate keys across domains raise ValueError."""
    with pytest.raises(ValueError, match="unique across"):
        SampleSchema(
            features={"voltage": {"raw": pa.float64()}},
            targets={"voltage": {"raw": pa.float32()}},
        )


@pytest.mark.unit
def test_sample_schema_invalid_char_in_key_raises():
    """Test that keys with invalid characters raise ValueError."""
    with pytest.raises(ValueError, match="invalid characters"):
        SampleSchema(
            features={"volt.age": {"raw": pa.float64()}},
        )


@pytest.mark.unit
def test_sample_schema_reserved_key_raises():
    """Test that reserved keywords as keys raise ValueError."""
    with pytest.raises(ValueError, match="reserved keyword"):
        SampleSchema(
            features={"raw": {"raw": pa.float64()}},
        )


# ---------------------------------------------------------------------
# domain_keys / domain_types tests
# ---------------------------------------------------------------------
@pytest.mark.unit
def test_domain_keys():
    """Test domain_keys returns correct keys."""
    schema = SampleSchema(
        features={"voltage": {"raw": pa.float64()}, "current": {"raw": pa.float64()}},
        targets={"soh": {"raw": pa.float32()}},
        tags={},
    )
    assert sorted(schema.domain_keys("features")) == ["current", "voltage"]
    assert schema.domain_keys("targets") == ["soh"]
    assert schema.domain_keys("tags") == []


@pytest.mark.unit
def test_domain_keys_invalid_domain_raises():
    """Test domain_keys raises for unknown domain."""
    schema = SampleSchema()
    with pytest.raises(ValueError, match="Unknown domain"):
        schema.domain_keys("invalid")


@pytest.mark.unit
def test_domain_types():
    """Test domain_types returns correct type mappings."""
    schema = SampleSchema(
        features={"voltage": {"raw": pa.float64()}},
        targets={},
    )
    types = schema.domain_types("features")
    assert "voltage" in types
    assert "raw" in types["voltage"]


@pytest.mark.unit
def test_domain_types_invalid_domain_raises():
    """Test domain_types raises for unknown domain."""
    schema = SampleSchema()
    with pytest.raises(ValueError, match="Unknown domain"):
        schema.domain_types("invalid")


# ---------------------------------------------------------------------
# rep_keys / rep_types tests
# ---------------------------------------------------------------------
@pytest.mark.unit
def test_rep_keys():
    """Test rep_keys returns available representations."""
    schema = SampleSchema(
        features={"voltage": {"raw": pa.float64(), "transformed": pa.float64()}},
    )
    reps = schema.rep_keys("features", "voltage")
    assert "raw" in reps
    assert "transformed" in reps


@pytest.mark.unit
def test_rep_keys_missing_key_raises():
    """Test rep_keys raises for missing column key."""
    schema = SampleSchema(features={})
    with pytest.raises(KeyError, match="not found"):
        schema.rep_keys("features", "nonexistent")


@pytest.mark.unit
def test_rep_types():
    """Test rep_types returns correct DataType mapping."""
    schema = SampleSchema(
        features={"voltage": {"raw": pa.float64()}},
    )
    types = schema.rep_types("features", "voltage")
    assert "raw" in types
    assert types["raw"] == pa.float64()


@pytest.mark.unit
def test_rep_types_missing_key_raises():
    """Test rep_types raises for missing key."""
    schema = SampleSchema(features={})
    with pytest.raises(KeyError, match="not found"):
        schema.rep_types("features", "nonexistent")


# ---------------------------------------------------------------------
# from_table tests
# ---------------------------------------------------------------------
@pytest.mark.unit
def test_from_table():
    """Test SampleSchema.from_table infers schema from Arrow table."""
    table = pa.table(
        {
            "features.voltage.raw": pa.array([[1.0, 2.0], [3.0, 4.0]]),
            "targets.soh.raw": pa.array([0.95, 0.93]),
            "tags.cell_id.raw": pa.array(["A", "B"]),
            DOMAIN_SAMPLE_UUIDS: pa.array(["id1", "id2"]),
        },
    )
    schema = SampleSchema.from_table(table)
    assert "voltage" in schema.features
    assert "soh" in schema.targets
    assert "cell_id" in schema.tags


@pytest.mark.unit
def test_from_table_invalid_column_format_raises():
    """Test from_table raises for invalid column naming."""
    table = pa.table({"bad_column": pa.array([1, 2, 3])})
    with pytest.raises(ValueError, match="Invalid column"):
        SampleSchema.from_table(table)


@pytest.mark.unit
def test_from_table_unknown_domain_raises():
    """Test from_table raises for unknown domain prefix."""
    table = pa.table({"unknown.voltage.raw": pa.array([1.0, 2.0])})
    with pytest.raises(ValueError, match="Unknown domain"):
        SampleSchema.from_table(table)


# ---------------------------------------------------------------------
# ensure_sample_id tests
# ---------------------------------------------------------------------
@pytest.mark.unit
def test_ensure_sample_id_adds_column():
    """Test ensure_sample_id adds UUID column when missing."""
    table = pa.table({"features.x.raw": pa.array([1.0, 2.0])})
    result = ensure_sample_id(table)
    assert DOMAIN_SAMPLE_UUIDS in result.column_names
    ids = result[DOMAIN_SAMPLE_UUIDS].to_pylist()
    assert len(ids) == 2
    assert len(set(ids)) == 2  # unique


@pytest.mark.unit
def test_ensure_sample_id_preserves_existing():
    """Test ensure_sample_id preserves existing valid column."""
    table = pa.table(
        {
            "features.x.raw": pa.array([1.0, 2.0]),
            DOMAIN_SAMPLE_UUIDS: pa.array(["a", "b"]),
        },
    )
    result = ensure_sample_id(table)
    ids = result[DOMAIN_SAMPLE_UUIDS].to_pylist()
    assert ids == ["a", "b"]


@pytest.mark.unit
def test_ensure_sample_id_non_string_raises():
    """Test ensure_sample_id raises for non-string column type."""
    table = pa.table(
        {
            "features.x.raw": pa.array([1.0, 2.0]),
            DOMAIN_SAMPLE_UUIDS: pa.array([1, 2]),
        },
    )
    with pytest.raises(TypeError, match="string"):
        ensure_sample_id(table)


# ---------------------------------------------------------------------
# validate_str_list tests
# ---------------------------------------------------------------------
@pytest.mark.unit
def test_validate_str_list_valid():
    """Test validate_str_list passes for valid keys."""
    validate_str_list(["cell_id", "cycle_number", "soh"])


@pytest.mark.unit
def test_validate_str_list_duplicates_raises():
    """Test validate_str_list raises for duplicate keys."""
    with pytest.raises(ValueError, match="Duplicate"):
        validate_str_list(["cell_id", "cell_id"])


@pytest.mark.unit
def test_validate_str_list_invalid_chars_raises():
    """Test validate_str_list raises for keys with invalid characters."""
    with pytest.raises(ValueError, match="invalid characters"):
        validate_str_list(["cell.id"])


@pytest.mark.unit
def test_validate_str_list_reserved_name_raises():
    """Test validate_str_list raises for reserved keywords."""
    with pytest.raises(ValueError, match="reserved keyword"):
        validate_str_list(["sample_uuids"])


# ---------------------------------------------------------------------
# Schema version
# ---------------------------------------------------------------------
@pytest.mark.unit
def test_schema_version_is_string():
    """Test SCHEMA_VERSION is a valid version string."""
    assert isinstance(SCHEMA_VERSION, str)
    parts = SCHEMA_VERSION.split(".")
    assert len(parts) == 3
    assert all(p.isdigit() for p in parts)
