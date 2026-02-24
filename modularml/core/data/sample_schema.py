"""
Schema definition and validation for :class:`FeatureSet` sample data.

Provides the :class:`SampleSchema` dataclass for defining and validating the
column layout of a :class:`FeatureSet` across its three domains (features,
targets, tags), along with utilities for sample identity and key validation.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pyarrow as pa

from modularml.core.data.schema_constants import (
    DOMAIN_FEATURES,
    DOMAIN_SAMPLE_UUIDS,
    DOMAIN_TAGS,
    DOMAIN_TARGETS,
    INVALID_LABEL_CHARACTERS,
    REP_RAW,
    REP_TRANSFORMED,
)

if TYPE_CHECKING:
    from collections.abc import Mapping


SCHEMA_VERSION = "1.0.0"

# Metadata keys / prefixes embedded in Arrow tables
METADATA_PREFIX = "modularml"
METADATA_SCHEMA_VERSION_KEY = f"{METADATA_PREFIX}.version"
SHAPE_SUFFIX = "shape"
DTYPE_SUFFIX = "dtype"


@dataclass
class SampleSchema:
    """
    Defines and validates the schema of a :class:`FeatureSet`.

    Description:
        Each :class:`FeatureSet` is organized into three structured domains:
          - **features**: model inputs (e.g., voltage, current)
          - **targets**: supervised outputs (e.g., SOH, capacity)
          - **tags**: metadata or identifiers (e.g., cell_id, SOC)

        The schema acts as the contract that ensures consistent column names, \
        data types, and separation across these domains.

        Every Arrow table that follows this schema must also contain a \
        global identifier column, `DOMAIN_SAMPLE_UUIDS`, storing a unique \
        per-row ID string (UUID or hash).

    """

    # Mapping: domain -> column name -> column representation -> dtype
    features: Mapping[str, Mapping[str, pa.DataType]] = field(default_factory=dict)
    targets: Mapping[str, Mapping[str, pa.DataType]] = field(default_factory=dict)
    tags: Mapping[str, Mapping[str, pa.DataType]] = field(default_factory=dict)

    def __post_init__(self):
        """
        Validates and normalizes schema mappings after initialization.

        Ensures that:
          1. All mappings are dicts (not arbitrary Mappings).
          2. No duplicate column names appear across domains.
          3. Columns names do not contains invalid characters.
        """
        # Instantiate dicts
        self.features = {k: dict(v) for k, v in dict(self.features).items()}
        self.targets = {k: dict(v) for k, v in dict(self.targets).items()}
        self.tags = {k: dict(v) for k, v in dict(self.tags).items()}

        # Detect duplicate column names across domains
        duplicates = (
            (set(self.features) & set(self.targets))
            | (set(self.features) & set(self.tags))
            | (set(self.targets) & set(self.tags))
        )
        if duplicates:
            msg = f"Schema keys must be unique across features/targets/tags. Duplicated keys: {sorted(duplicates)}"
            raise ValueError(msg)

        # Check for invalid characters and reserved names
        all_keys = (
            set(self.features.keys()) | set(self.targets.keys()) | set(self.tags.keys())
        )
        validate_str_list(list(all_keys))

    # ================================================
    # Domain utility methods
    # ================================================
    def domain_keys(self, domain: str) -> list[str]:
        """
        Return the list of column names for a given domain.

        Args:
            domain (str): One of {"features", "targets", "tags"}.

        Returns:
            list[str]: Column names in that domain.

        Raises:
            ValueError: if the domain name is invalid.

        """
        domain = domain.lower()
        if domain == DOMAIN_FEATURES:
            return list(self.features.keys())
        if domain == DOMAIN_TARGETS:
            return list(self.targets.keys())
        if domain == DOMAIN_TAGS:
            return list(self.tags.keys())
        msg = f"Unknown domain '{domain}'. Expected one of: {DOMAIN_FEATURES}, {DOMAIN_TARGETS}, {DOMAIN_TAGS}"
        raise ValueError(msg)

    def domain_types(self, domain: str) -> dict[str, dict[str, pa.DataType]]:
        """
        Return the {column_name: DataType} mapping for a given domain.

        Args:
            domain (str): Domain name ("features", "targets", "tags").

        Returns:
            dict[str, dict[str, pa.DataType]]:
                Mapping of column names to representation to Arrow data types.

        """
        domain = domain.lower()
        if domain == DOMAIN_FEATURES:
            return self.features
        if domain == DOMAIN_TARGETS:
            return self.targets
        if domain == DOMAIN_TAGS:
            return self.tags
        msg = f"Unknown domain '{domain}'. Expected one of: {DOMAIN_FEATURES}, {DOMAIN_TARGETS}, {DOMAIN_TAGS}"
        raise ValueError(msg)

    def rep_keys(self, domain: str, key: str) -> list[str]:
        """
        Return available representations for a given column key.

        Args:
            domain (str): Domain name ("features", "targets", "tags").
            key (str): Column name in specified domain.

        Returns:
            list[str]: Representation names (e.g., "raw")

        """
        dom_types = self.domain_types(domain)
        if key not in dom_types:
            msg = f"Column '{key}' not found in domain '{domain}'"
            raise KeyError(msg)
        return list(dom_types[key].keys())

    def rep_types(self, domain: str, key: str) -> dict[str, pa.DataType]:
        """
        Return the {rep_name: DataType} mapping for a given domain and column.

        Args:
            domain (str): Domain name ("features", "targets", "tags").
            key (str): Column name in specified domain.

        Returns:
            Mapping of representations to Arrow data types.

        """
        dom_types = self.domain_types(domain)
        if key not in dom_types:
            msg = f"Column '{key}' not found in domain '{domain}'"
            raise KeyError(msg)
        return dom_types[key]

    # ================================================
    # Flat schema inference
    # ================================================
    @classmethod
    def from_table(cls, table: pa.Table) -> SampleSchema:
        """
        Infer :class:`SampleSchema` from a flat Arrow table.

        Args:
            table (pa.Table): Arrow table with columns following the naming
                convention ``<domain>.<key>.<representation>``.

        Returns:
            SampleSchema: Inferred schema instance.

        Expected column naming:
            "<domain>.<key>.<representation>"

        Example:
            features.voltage.raw
            features.voltage.transformed
            targets.soh.raw
            tags.cell_id.raw
            sample_id

        """
        features: dict[str, dict[str, pa.DataType]] = {}
        targets: dict[str, dict[str, pa.DataType]] = {}
        tags: dict[str, dict[str, pa.DataType]] = {}

        for col in table.schema.names:
            if col == DOMAIN_SAMPLE_UUIDS:
                continue

            parts = col.split(".")
            if len(parts) != 3:
                msg = f"Invalid column '{col}'. Expected '<domain>.<key>.<representations>' format."
                raise ValueError(msg)

            domain, key, rep = parts
            dtype = table.schema.field(col).type

            if domain == DOMAIN_FEATURES:
                features.setdefault(key, {})[rep] = dtype
            elif domain == DOMAIN_TARGETS:
                targets.setdefault(key, {})[rep] = dtype
            elif domain == DOMAIN_TAGS:
                tags.setdefault(key, {})[rep] = dtype
            else:
                msg = f"Unknown domain '{domain}' in column '{col}'"
                raise ValueError(msg)

        return cls(features=features, targets=targets, tags=tags)


def ensure_sample_id(table: pa.Table) -> pa.Table:
    """
    Ensure that the Arrow table contains a unique `DOMAIN_SAMPLE_UUIDS` column.

    Description:
        - If the column exists, validates that it is of type string.
        - If it does not exist, generates a new UUID string for each row.
        - This column is used for traceability across subsets and views.

    Args:
        table (pa.Table): Input Arrow table.

    Returns:
        pa.Table: A table guaranteed to contain a valid DOMAIN_SAMPLE_UUIDS.

    """
    if DOMAIN_SAMPLE_UUIDS in table.column_names:
        col = table[DOMAIN_SAMPLE_UUIDS]
        if not pa.types.is_string(col.type):
            msg = f"'{DOMAIN_SAMPLE_UUIDS}' column must be of type string, got {col.type}."
            raise TypeError(msg)
        return table

    sample_ids = pa.array(
        [str(uuid.uuid4()) for _ in range(table.num_rows)],
        type=pa.string(),
    )
    return table.append_column(DOMAIN_SAMPLE_UUIDS, sample_ids)


def validate_str_list(keys: list[str]):
    """
    Validate a list of string keys for naming consistency and safety.

    Description:
        Ensures that all provided keys meet naming requirements for use in
        FeatureSet, SampleCollection, or related schema contexts. This function
        enforces:
          - Uniqueness of all keys.
          - Absence of invalid characters (see `INVALID_LABEL_CHARACTERS`).
          - Exclusion of reserved schema keywords (e.g., `DOMAIN_SAMPLE_UUIDS`,
            `DOMAIN_FEATURES`, etc.).
          - Exclusion of internal metadata prefixes/postfixes used in column
            naming conventions.

    Args:
        keys (list[str]):
            List of strings to validate.

    Raises:
        ValueError:
            If any of the following conditions occur:
              - Duplicate keys are detected.
              - Keys contain invalid characters.
              - Keys use reserved schema names or internal label conventions.

    """
    # Detect duplicate elements in keys
    if len(set(keys)) != len(keys):
        msg = "Duplicate elements exist in `keys`."
        raise ValueError(msg)

    # Check invalid characters
    invalid_chars = tuple(INVALID_LABEL_CHARACTERS)
    invalid_keys: list[str] = []
    for k in keys:
        if any(k.find(ch) != -1 for ch in invalid_chars):
            invalid_keys.append(k)
    if invalid_keys:
        msg = (
            f"The following keys contain invalid characters: {', '.join(invalid_keys)}. "
            f"Keys cannot contain any of: {list(INVALID_LABEL_CHARACTERS)}"
        )
        raise ValueError(msg)

    # Ensure reserved names are not used
    for res_key in (
        DOMAIN_SAMPLE_UUIDS,
        DOMAIN_FEATURES,
        DOMAIN_TARGETS,
        DOMAIN_TAGS,
        REP_RAW,
        REP_TRANSFORMED,
        METADATA_PREFIX,
        DTYPE_SUFFIX,
        SHAPE_SUFFIX,
    ):
        if res_key in keys:
            msg = f"`{res_key}` is a reserved keyword and cannot be used."
            raise ValueError(msg)
