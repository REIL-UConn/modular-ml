from __future__ import annotations

import ast
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa

from modularml.core.data.sample_schema import (
    DTYPE_SUFFIX,
    METADATA_PREFIX,
    METADATA_SCHEMA_VERSION_KEY,
    SCHEMA_VERSION,
    SHAPE_SUFFIX,
    SampleSchema,
)
from modularml.core.data.schema_constants import (
    DOMAIN_FEATURES,
    DOMAIN_SAMPLE_ID,
    DOMAIN_TAGS,
    DOMAIN_TARGETS,
    REP_RAW,
)
from modularml.utils.data.conversion import (
    convert_dict_to_format,
    convert_to_format,
    stack_nested_numpy,
)
from modularml.utils.data.data_format import DataFormat
from modularml.utils.data.formatting import ensure_list
from modularml.utils.data.pyarrow_data import (
    _ensure_domain_prefix,
    _ensure_rep_suffix,
    _remove_domain_prefix,
    _remove_rep_suffix,
    get_dtype_of_pyarrow_array,
    get_shape_of_pyarrow_array,
    numpy_to_sample_schema_column,
)


@dataclass
class SampleCollection:
    """
    Immutable, Arrow-backed container for sample-level data in ModularML.

    Description:
        `SampleCollection` is the canonical in-memory data container used
        throughout ModularML to store and access sample-level data across
        all domains:

            - features
            - targets
            - tags
            - sample identifiers

        Data are stored internally as a `pyarrow.Table` whose column schema
        follows the `SampleSchema` contract:

            <domain>.<key>.<representation>

        The collection guarantees the presence of a globally unique
        `sample_id` column for row-level traceability across filtering,
        batching, sampling, model execution, and export.

        The underlying Arrow table is immutable. Any mutation (e.g. adding
        a representation) results in a new table being created internally.

    Args:
        table (pa.Table):
            Arrow table containing all sample data.
        schema (SampleSchema | None):
            Optional schema describing domain, key, and representation layout.
            If omitted, the schema is inferred from the table metadata.

    """

    table: pa.Table
    schema: SampleSchema | None = None

    # ================================================
    # Initialization
    # ================================================
    def __post_init__(self):
        """
        Finalize initialization and validate internal consistency.

        Description:
            Performs the following steps in order:

                1. Ensures a valid, unique `sample_id` column exists.
                2. Infers a `SampleSchema` from the table if none is provided.
                3. Validates column uniqueness and schema integrity.
                4. Embeds per-column shape and dtype metadata into the table schema.

            This method is invoked automatically after dataclass construction.
        """
        self._ensure_sample_id()
        if self.schema is None:
            self.schema = SampleSchema.from_table(self.table)
        self._validate_schema()
        self._embed_metadata()

    def _ensure_sample_id(self):
        """
        Ensure the presence of a unique string-based `sample_id` column.

        Description:
            - If the column is missing, a new UUID4 string is generated per row.
            - If the column exists:
                - Its type is validated to be string-compatible.
                - Duplicate values are detected and replaced with new UUIDs.

            This method guarantees that every row in the collection can be
            uniquely identified across all downstream operations.
        """
        colnames = set(self.table.column_names)

        # Case 1: Add new column if missing
        if DOMAIN_SAMPLE_ID not in colnames:
            n = self.table.num_rows
            ids = pa.array([str(uuid.uuid4()) for _ in range(n)], type=pa.string())
            self.table = self.table.append_column(DOMAIN_SAMPLE_ID, ids)
            return

        # Case 2: Validate existing column
        col = self.table[DOMAIN_SAMPLE_ID]
        if not pa.types.is_string(col.type):
            msg = f"'{DOMAIN_SAMPLE_ID}' column must be of type string, got {col.type}."
            raise TypeError(msg)

        # Extract and check uniqueness
        ids = col.to_pandas().astype(str)
        n_unique = ids.nunique(dropna=False)
        if n_unique < len(ids):
            # Regenerate to ensure unique identifiers
            unique_ids = pa.array(
                [str(uuid.uuid4()) for _ in range(len(ids))],
                type=pa.string(),
            )
            self.table = self.table.set_column(
                self.table.schema.get_field_index(DOMAIN_SAMPLE_ID),
                DOMAIN_SAMPLE_ID,
                unique_ids,
            )

    def _validate_schema(self):
        """
        Validate basic structural integrity of the Arrow table.

        Description:
            Ensures that:
                - All column names are unique.
                - No duplicate fields exist in the Arrow schema.

            Domain-level validation (required domains, representations, etc.)
            is handled by `SampleSchema`.
        """
        unique_cols = set(self.table.column_names)

        # Check for duplicate/non-unique columns
        if len(unique_cols) != len(self.table.column_names):
            unique, counts = np.unique(
                np.asarray(self.table.column_names),
                return_counts=True,
            )
            dups = unique[counts > 1]
            msg = f"Detected duplicate column names: {dups}"
            raise ValueError(msg)

        # Required column validation is done in SampleSchema

    def _embed_metadata(self) -> None:
        """
        Embed ModularML metadata into the Arrow table schema.

        Description:
            Writes the following metadata entries:

                - Table schema version.
                - Per-column shape metadata.
                - Per-column dtype metadata.

            Metadata keys follow the convention:

                modularml.sample.<domain>.<key>.<rep>.<suffix>

            This metadata is used to enable fast reconstruction of tensor
            shapes and dtypes without scanning the Arrow buffers.

        Notes:
            This operation replaces the table's schema metadata in-place.

        """
        meta = dict(self.table.schema.metadata or {})

        # Table version
        meta[METADATA_SCHEMA_VERSION_KEY.encode()] = SCHEMA_VERSION.encode()

        # Domain shapes and dtypes
        for domain in [DOMAIN_FEATURES, DOMAIN_TARGETS] + (
            [DOMAIN_TAGS] if self.has_tags else []
        ):
            # Get shapes and dtypes
            d_shapes: dict[str, tuple[int, ...]] = self._get_domain_shapes(
                domain=domain,
                include_domain_prefix=False,
                include_rep_suffix=True,
            )
            d_dtypes: dict[str, str] = self._get_domain_dtypes(
                domain=domain,
                include_domain_prefix=False,
                include_rep_suffix=True,
            )
            if d_shapes.keys() != d_dtypes.keys():
                msg = f"`{domain}` keys differ between shapes and data types."

            # Update meta data
            for k in d_shapes:
                for s_dic, suffix in zip(
                    (d_shapes, d_dtypes),
                    (SHAPE_SUFFIX, DTYPE_SUFFIX),
                    strict=True,
                ):
                    meta_key = (f"{METADATA_PREFIX}.{domain}.{k}.{suffix}").encode()
                    meta_val = str(s_dic[k]).encode()
                    if meta_key in meta and meta_val != meta[meta_key]:
                        msg = (
                            f"Meta data already exist for key `{meta_key}` but it "
                            f"does not match expected value: {meta[meta_key]} "
                            f"!= {meta_val}."
                        )
                        raise ValueError(msg)
                    meta[meta_key] = meta_val

        # Update table schema
        self.table = self.table.replace_schema_metadata(meta)

    def __eq__(self, other):
        if not isinstance(other, SampleCollection):
            msg = f"Cannot compare equality between SampleCollection and {type(other)}"
            raise TypeError(msg)

        return self.table.equals(other.table)

    __hash__ = None

    # ================================================
    # Helpers
    # ================================================
    def _format_col_str(
        self,
        *,
        domain: str,
        key: str,
        rep: str | None,
        include_domain_prefix: bool,
        include_rep_suffix: bool,
    ) -> str:
        """
        Construct a formatted column identifier string.

        Description:
            Builds a column name using the canonical components:

                - domain
                - key
                - representation

            The inclusion of the domain prefix and representation suffix
            is controlled explicitly via boolean flags.

        Args:
            domain (str):
                Domain name (e.g. "features", "targets", "tags").
            key (str):
                Logical column key.
            rep (str | None):
                Representation name, if applicable.
            include_domain_prefix (bool):
                Whether to include the domain prefix.
            include_rep_suffix (bool):
                Whether to include the representation suffix.

        Returns:
            str:
                Formatted column identifier.

        """
        name = key
        if include_domain_prefix:
            name = f"{domain}.{name}"
        if include_rep_suffix and rep is not None:
            name = f"{name}.{rep}"
        return name

    def _get_domain_keys(
        self,
        domain: str,
        *,
        include_rep_suffix: bool,
        include_domain_prefix: bool,
    ) -> list[str]:
        """
        Retrieve logical column keys for a domain.

        Description:
            Returns column identifiers for a domain, optionally including
            representation suffixes and domain prefixes.

            This method is the internal backbone for all public key-access
            APIs and guarantees consistent formatting behavior.

        Returns:
            list[str]:
                Sorted list of column identifiers.

        """
        out = []
        for key in self.schema.domain_keys(domain):
            reps = self.schema.rep_keys(domain, key)
            if include_rep_suffix:
                for rep in reps:
                    out.append(
                        self._format_col_str(
                            domain=domain,
                            key=key,
                            rep=rep,
                            include_domain_prefix=include_domain_prefix,
                            include_rep_suffix=True,
                        ),
                    )
            else:
                out.append(
                    self._format_col_str(
                        domain=domain,
                        key=key,
                        rep=None,
                        include_domain_prefix=include_domain_prefix,
                        include_rep_suffix=False,
                    ),
                )
        return sorted(out)

    def _get_domain_shapes(
        self,
        domain: str,
        *,
        include_rep_suffix: bool,
        include_domain_prefix: bool,
    ) -> dict[str, tuple[int, ...]]:
        """
        Retrieve per-representation tensor shapes for a domain.

        Description:
            For each key and representation in the domain, returns the
            stored or inferred tensor shape excluding the sample dimension.

        Returns:
            dict[str, tuple[int, ...]]:
                Mapping from formatted column identifiers to shapes.

        """
        out = {}
        for key in self.schema.domain_keys(domain):
            for rep in self.schema.rep_keys(domain, key):
                name = self._format_col_str(
                    domain=domain,
                    key=key,
                    rep=rep if include_rep_suffix else None,
                    include_domain_prefix=include_domain_prefix,
                    include_rep_suffix=include_rep_suffix,
                )
                out[name] = self._get_rep_shape(domain, key, rep)
        return out

    def _get_domain_dtypes(
        self,
        domain: str,
        *,
        include_rep_suffix: bool,
        include_domain_prefix: bool,
    ) -> dict[str, tuple[int, ...]]:
        """
        Retrieve per-representation data types for a domain.

        Description:
            Returns the logical dtype for each representation, either from
            stored metadata or inferred directly from Arrow buffers.

        Returns:
            dict[str, str]:
                Mapping from formatted column identifiers to dtype strings.

        """
        out = {}
        for key in self.schema.domain_keys(domain):
            for rep in self.schema.rep_keys(domain, key):
                name = self._format_col_str(
                    domain=domain,
                    key=key,
                    rep=rep if include_rep_suffix else None,
                    include_domain_prefix=include_domain_prefix,
                    include_rep_suffix=include_rep_suffix,
                )
                out[name] = self._get_rep_dtype(domain, key, rep)
        return out

    def _get_domain_data(
        self,
        domain: str,
        fmt: DataFormat = DataFormat.DICT_NUMPY,
        *,
        keys: str | list[str] | None = None,
        rep: str | None = None,
        include_rep_suffix: bool = False,
        include_domain_prefix: bool = False,
    ):
        """
        Retrieve domain data in a chosen format.

        Args:
            domain (str):
                domain (str): One of {"features", "targets", "tags"}.
            fmt (DataFormat): Desired output format (see :class:`DataFormat`). \
                Defaults to a single dictionary of numpy arrays.
            keys (str | list[str] | None): Optional subset of domain keys to return. \
                If None, all feature keys are returned. Defaults to None.
            rep (str, optional): The representation (e.g., "raw" or "transformed") of the domain keys to \
                return. If None, all representations are returned and `include_rep_suffix` is set to True. \
                If specfiied, `keys` must all have matching representations. Defaults to None.
            include_rep_suffix (bool): Whether to include the representation suffix in the \
                domain keys (e.g., "voltage" or "voltage.raw"). Defaults to False.
            include_domain_prefix (bool): Wether to include the domain prefix in the \
                domain keys (e.g., "voltage" or "features.voltage"). Defaults to False.

        Returns:
            Domain data in the requested format.

        """
        keys = ensure_list(keys)
        domain_keys = self._get_domain_keys(
            domain=domain,
            include_domain_prefix=False,
            include_rep_suffix=False,
        )
        invalid = set(keys).difference(set(domain_keys))
        if invalid:
            msg = f"The following keys do not exist in the `{domain}` domain: {invalid}"
            raise ValueError(msg)
        if rep is None:
            include_rep_suffix = True

        res = {}
        for k in keys or domain_keys:
            # Get specified (or all) representations for this column
            reps = ensure_list(rep) if rep else self._get_rep_keys(domain=domain, key=k)
            for r in reps:
                f_key = self._format_col_str(
                    domain=domain,
                    key=k,
                    rep=r,
                    include_domain_prefix=include_domain_prefix,
                    include_rep_suffix=include_rep_suffix,
                )
                res[f_key] = self._get_rep_data(
                    domain=domain,
                    key=k,
                    rep=r,
                    fmt=DataFormat.NUMPY,
                )

        if fmt == DataFormat.DICT_NUMPY:
            return res
        return convert_dict_to_format(res, fmt=fmt, mode="stack", axis=1)

    def _get_rep_keys(self, domain: str, key: str) -> list[str]:
        """
        Return all available representations for a domain key.

        Args:
            domain (str):
                Domain name.
            key (str):
                Logical column key.

        Returns:
            list[str]:
                Representation names (e.g. ["raw", "transformed"]).

        """
        if domain not in (DOMAIN_FEATURES, DOMAIN_TARGETS, DOMAIN_TAGS):
            msg = f"Invalid domain '{domain}'"
            raise ValueError(msg)
        return list(self.schema.rep_keys(domain, key))

    def _get_rep_shape(self, domain: str, key: str, rep: str) -> tuple[int, ...]:
        """
        Obtain the shape of a specific column and representation.

        Args:
            domain (str):
                One of {"features", "targets", "tags"}.
            key (str):
                Column name with in the specified domain.
            rep (str):
                Column representation (e.g., "raw" or "transformed")

        Returns:
            tuple[int, ...]: Data shape for specific column and representation.

        """
        col_name = self._format_col_str(
            domain=domain,
            key=key,
            rep=rep,
            include_domain_prefix=True,
            include_rep_suffix=True,
        )

        # 1. Check meta-data
        meta = self.table.schema.metadata or {}
        meta_key = f"{METADATA_PREFIX}.{col_name}.{SHAPE_SUFFIX}"
        shape = meta.get(meta_key.encode())
        if shape:
            return tuple(ast.literal_eval(shape.decode()))

        # 2. Infer shapes directly from stored data (slow)
        rep_arr: pa.Array = self.table[col_name].combine_chunks()
        return get_shape_of_pyarrow_array(rep_arr, include_nrows=False)

    def _get_rep_dtype(self, domain: str, key: str, rep: str) -> str:
        """
        Obtain the data type of a specific column and representation.

        Description:
            Attempts to read data types from table metadata. If not found, \
            falls back to inferring dtypes directly from stored data.

        Args:
            domain (str): One of {"features", "targets", "tags"}.
            key (str): Column name with in the specified domain.
            rep (str): Column representation (e.g., "raw" or "transformed")

        Returns:
            Data type string (e.g., "float32", "int64") for specific column and representation.

        """
        col_name = self._format_col_str(
            domain=domain,
            key=key,
            rep=rep,
            include_domain_prefix=True,
            include_rep_suffix=True,
        )

        # 1. Check meta-data
        meta = self.table.schema.metadata or {}
        meta_key = f"{METADATA_PREFIX}.{col_name}.{DTYPE_SUFFIX}"
        dtype = meta.get(meta_key.encode())
        if dtype:
            return dtype.decode()

        # 2. Infer data type directly from stored data (slow)
        rep_arr: pa.Array = self.table[col_name].combine_chunks()
        return get_dtype_of_pyarrow_array(rep_arr)

    def _get_rep_data(
        self,
        domain: str,
        key: str,
        rep: str,
        *,
        fmt: DataFormat | None = None,
    ):
        """
        Retrieve a single column of data in the chosen format.

        Args:
            domain (str): One of {"features", "targets", "tags"}.
            key (str): Column name with in the specified domain.
            rep (str): Column representation (e.g., "raw" or "transformed")
            fmt (DataFormat, optional): Desired output format (see :class:`DataFormat`). \
                If None, the PyArrow view (pa.ListArray) is returned. Defaults to None.

        Returns:
            Column representation data in the requested format.

        """
        col_name = self._format_col_str(
            domain=domain,
            key=key,
            rep=rep,
            include_domain_prefix=True,
            include_rep_suffix=True,
        )

        # Access the representation column as an Arrow array
        if col_name not in self.table.column_names:
            msg = f"Column '{col_name}' not found in PyArrow table."
            raise KeyError(msg)
        pa_arr: pa.Array = self.table[col_name].combine_chunks()

        # Return raw pyarrow array is no format is specified
        if fmt is None:
            return pa_arr

        # Retrieve metadata for decoding
        shape = self._get_rep_shape(domain=domain, key=key, rep=rep)
        dtype_str = self._get_rep_dtype(domain=domain, key=key, rep=rep)

        # Handle binary-encoded tensors
        if pa.types.is_fixed_size_binary(pa_arr.type):
            try:
                np_dtype = np.dtype(dtype_str)
            except Exception as e:
                msg = f"Invalid dtype string '{dtype_str}' in metadata."
                raise ValueError(msg) from e

            flat_size = int(np.prod(shape))
            n_rows = len(pa_arr)

            # Decode each row's binary buffer to an ndarray of the correct dtype/shape
            decoded = np.empty((n_rows, *tuple(shape)), dtype=np_dtype)
            for i in range(n_rows):
                buf = (
                    pa_arr[i].as_buffer()
                    if hasattr(pa_arr[i], "as_buffer")
                    else pa_arr[i]
                )
                if buf is None:
                    decoded[i] = np.zeros(shape, dtype=np_dtype)
                    continue
                row_bytes = np.frombuffer(buf, dtype=np_dtype, count=flat_size)
                decoded[i] = row_bytes.reshape(shape)

            return convert_to_format(decoded, fmt=fmt)

        # Otherwise PyArrow's `to_numpy` for list-like arrays (1D/2D)
        # .to_numpy() returns an object-array, we need to convert to proper shape and dtype
        data = pa_arr.to_numpy(zero_copy_only=False)
        np_arr = stack_nested_numpy(data, shape)
        # Drop trailing singleton dimensions (e.g., (N,1) to (N,))
        while np_arr.ndim > 1 and np_arr.shape[-1] == 1:
            np_arr = np_arr.squeeze(-1)
        return convert_to_format(np_arr, fmt=fmt)

    # ================================================
    # Core properties
    # ================================================
    @property
    def table_version(self) -> str | None:
        """
        Retrieve the ModularML schema version stored in the table metadata.

        Returns:
            Version string if present; otherwise ``None``.

        """
        meta = self.table.schema.metadata or {}
        val = meta.get(METADATA_SCHEMA_VERSION_KEY.encode())
        return val.decode() if val else None

    @property
    def available_domains(self) -> list[str]:
        return [DOMAIN_FEATURES, DOMAIN_TARGETS, DOMAIN_TAGS, DOMAIN_SAMPLE_ID]

    @property
    def n_samples(self) -> int:
        """Total number of rows (samples) in the Arrow table."""
        return self.table.num_rows

    @property
    def has_tags(self) -> bool:
        """Return True if tags domain exists and is non-empty."""
        return len(self.get_tag_keys()) > 0

    # ================================================
    # Column name accessors
    # ================================================
    def get_feature_keys(
        self,
        *,
        include_rep_suffix: bool = False,
        include_domain_prefix: bool = False,
    ) -> list[str]:
        """
        Retrieve feature column names in this SampleCollection.

        Description:
            Returns keys for the specified domain with fully
            explicit control over key formatting. No implicit suffix or
            prefix behavior is applied.

        Args:
            include_rep_suffix (bool):
                Whether to include representation suffixes (e.g., "raw")
            include_domain_prefix (bool):
                Whether to include domain prefixes (e.g., "features")

        Returns:
            List of all feature column names.

        """
        return self._get_domain_keys(
            DOMAIN_FEATURES,
            include_rep_suffix=include_rep_suffix,
            include_domain_prefix=include_domain_prefix,
        )

    def get_target_keys(
        self,
        *,
        include_rep_suffix: bool = False,
        include_domain_prefix: bool = False,
    ) -> list[str]:
        """
        Retrieve target column names in this SampleCollection.

        Description:
            Returns keys for the specified domain with fully
            explicit control over key formatting. No implicit suffix or
            prefix behavior is applied.

        Args:
            include_rep_suffix (bool):
                Whether to include representation suffixes (e.g., "raw")
            include_domain_prefix (bool):
                Whether to include domain prefixes (e.g., "targets")

        Returns:
            List of all target column names.

        """
        return self._get_domain_keys(
            DOMAIN_TARGETS,
            include_rep_suffix=include_rep_suffix,
            include_domain_prefix=include_domain_prefix,
        )

    def get_tag_keys(
        self,
        *,
        include_rep_suffix: bool = False,
        include_domain_prefix: bool = False,
    ) -> list[str]:
        """
        Retrieve tag column names in this SampleCollection.

        Description:
            Returns keys for the specified domain with fully
            explicit control over key formatting. No implicit suffix or
            prefix behavior is applied.

        Args:
            include_rep_suffix (bool):
                Whether to include representation suffixes (e.g., "raw")
            include_domain_prefix (bool):
                Whether to include domain prefixes (e.g., "tags")

        Returns:
            List of all tag column names.

        """
        return self._get_domain_keys(
            DOMAIN_TAGS,
            include_rep_suffix=include_rep_suffix,
            include_domain_prefix=include_domain_prefix,
        )

    def get_all_keys(
        self,
        *,
        include_rep_suffix: bool = True,
        include_domain_prefix: bool = True,
    ) -> list[str]:
        """
        Get all column names in the SampleCollection.

        Args:
            include_rep_suffix (bool):
                Whether to include the representation suffix in the keys
                (e.g., "voltage" or "voltage.raw"). Defaults to True.
            include_domain_prefix (bool): Wether to include the domain prefix in the
                keys (e.g., "cell_id" or "tags.cell_id"). Defaults to True.

        Returns:
            list[str]:
                All unique columns in the SampleCollection.

        """
        keys = []
        for domain in (DOMAIN_FEATURES, DOMAIN_TARGETS, DOMAIN_TAGS):
            if domain == DOMAIN_TAGS and not self.has_tags:
                continue
            d_cols = self._get_domain_keys(
                domain,
                include_rep_suffix=include_rep_suffix,
                include_domain_prefix=include_domain_prefix,
            )
            keys.extend(sorted(d_cols))

        if include_domain_prefix:
            keys.append(DOMAIN_SAMPLE_ID)

        return keys

    # ================================================
    # Column shape accessors
    # ================================================
    def get_feature_shapes(
        self,
        *,
        include_rep_suffix: bool = True,
        include_domain_prefix: bool = False,
    ) -> dict[str, tuple[int, ...]]:
        """
        Mapping of feature columns to their array shapes.

        Description:
            Returns a dictionary describing the shape of each feature representation
            in the FeatureSet.
            Keys are formatted depending on the bool flags and values are tuples
            representing the array shape (e.g., `(101,)`).
            The shape does not include the number of samples dimension.

        Args:
            include_rep_suffix (bool):
                Whether to include representation suffixes (e.g., "raw")
            include_domain_prefix (bool):
                Whether to include domain prefixes (e.g., "features")

        Returns:
            dict[str, tuple[int, ...]]:
                Mapping of feature column representation names to their shapes.

        """
        return self._get_domain_shapes(
            DOMAIN_FEATURES,
            include_rep_suffix=include_rep_suffix,
            include_domain_prefix=include_domain_prefix,
        )

    def get_target_shapes(
        self,
        *,
        include_rep_suffix: bool = True,
        include_domain_prefix: bool = False,
    ) -> dict[str, tuple[int, ...]]:
        """
        Mapping of target columns to their array shapes.

        Description:
            Returns a dictionary describing the shape of each target representation
            in the FeatureSet.
            Keys are formatted depending on the bool flags and values are tuples
            representing the array shape (e.g., `(101,)`).
            The shape does not include the number of samples dimension.

        Args:
            include_rep_suffix (bool):
                Whether to include representation suffixes (e.g., "raw")
            include_domain_prefix (bool):
                Whether to include domain prefixes (e.g., "targets")

        Returns:
            dict[str, tuple[int, ...]]:
                Mapping of target column representation names to their shapes.

        """
        return self._get_domain_shapes(
            DOMAIN_TARGETS,
            include_rep_suffix=include_rep_suffix,
            include_domain_prefix=include_domain_prefix,
        )

    def get_tag_shapes(
        self,
        *,
        include_rep_suffix: bool = True,
        include_domain_prefix: bool = False,
    ) -> dict[str, tuple[int, ...]]:
        """
        Mapping of tag columns to their array shapes.

        Description:
            Returns a dictionary describing the shape of each tag representation
            in the FeatureSet.
            Keys are formatted depending on the bool flags and values are tuples
            representing the array shape (e.g., `(101,)`).
            The shape does not include the number of samples dimension.

        Args:
            include_rep_suffix (bool):
                Whether to include representation suffixes (e.g., "raw")
            include_domain_prefix (bool):
                Whether to include domain prefixes (e.g., "tags")

        Returns:
            dict[str, tuple[int, ...]]:
                Mapping of tag column representation names to their shapes.

        """
        return self._get_domain_shapes(
            DOMAIN_TAGS,
            include_rep_suffix=include_rep_suffix,
            include_domain_prefix=include_domain_prefix,
        )

    # ================================================
    # Column data-type accessors
    # ================================================
    def get_feature_dtypes(
        self,
        *,
        include_rep_suffix: bool = True,
        include_domain_prefix: bool = False,
    ) -> dict[str, str]:
        """
        Mapping of feature columns to their data types.

        Description:
            Returns a dictionary describing the data type of each feature representation.
            Keys are formatted depending on the bool flags and values are string-based
            data-types (e.g., "float32").

        Args:
            include_rep_suffix (bool):
                Whether to include representation suffixes (e.g., "raw")
            include_domain_prefix (bool):
                Whether to include domain prefixes (e.g., "features")

        Returns:
            dict[str, str]:
                Mapping of feature column names to their data-types.

        """
        return self._get_domain_dtypes(
            DOMAIN_FEATURES,
            include_rep_suffix=include_rep_suffix,
            include_domain_prefix=include_domain_prefix,
        )

    def get_target_dtypes(
        self,
        *,
        include_rep_suffix: bool = True,
        include_domain_prefix: bool = False,
    ) -> dict[str, str]:
        """
        Mapping of targets columns to their data types.

        Description:
            Returns a dictionary describing the data type of each targets representation.
            Keys are formatted depending on the bool flags and values are string-based
            data-types (e.g., "float32").

        Args:
            include_rep_suffix (bool):
                Whether to include representation suffixes (e.g., "raw")
            include_domain_prefix (bool):
                Whether to include domain prefixes (e.g., "targetss")

        Returns:
            dict[str, str]:
                Mapping of targets column names to their data-types.

        """
        return self._get_domain_dtypes(
            DOMAIN_TARGETS,
            include_rep_suffix=include_rep_suffix,
            include_domain_prefix=include_domain_prefix,
        )

    def get_tag_dtypes(
        self,
        *,
        include_rep_suffix: bool = True,
        include_domain_prefix: bool = False,
    ) -> dict[str, str]:
        """
        Mapping of tag columns to their data types.

        Description:
            Returns a dictionary describing the data type of each tag representation.
            Keys are formatted depending on the bool flags and values are string-based
            data-types (e.g., "float32").

        Args:
            include_rep_suffix (bool):
                Whether to include representation suffixes (e.g., "raw")
            include_domain_prefix (bool):
                Whether to include domain prefixes (e.g., "tags")

        Returns:
            dict[str, str]:
                Mapping of tag column names to their data-types.

        """
        return self._get_domain_dtypes(
            DOMAIN_TAGS,
            include_rep_suffix=include_rep_suffix,
            include_domain_prefix=include_domain_prefix,
        )

    # ================================================
    # Domain data accessors
    # ================================================
    def get_columns(
        self,
        *,
        columns: list[str],
        fmt: DataFormat = DataFormat.DICT_NUMPY,
        include_domain_prefix: bool = True,
        include_rep_suffix: bool = True,
    ) -> Any:
        """
        Retrieve column data in a chosen format.

        Args:
            columns (str | list[str] | None):
                Fully-qualified column names to include
                (e.g. "features.voltage.raw"). These must exactly match existing
                columns in this collection.

            fmt (DataFormat):
                Desired output format (see :class:`DataFormat`).
                Defaults to a single dictionary of numpy arrays.

            include_domain_prefix (bool):
                Whether to include domain prefixes (e.g., "tags").
                Automatically included if multiple domains are included in the
                selected columns. Defaults to True.

            include_rep_suffix (bool):
                Whether to include representation suffixes (e.g., "raw").
                Automatically included if multiple representations are included in
                the selected columns. Defaults to True.

        """
        # Enforce domain/rep in naming if more than 1 in selected columns
        all_ds = []
        all_rs = []
        for col in columns:
            domain, _, rep = col.split(".", maxsplit=2)
            all_ds.append(domain)
            all_rs.append(rep)
        if len(set(all_ds)) > 1:
            include_domain_prefix = True
        if len(set(all_rs)) > 1:
            include_rep_suffix = True

        data = {}
        for col in columns:
            if col == DOMAIN_SAMPLE_ID:
                data[col] = self.get_sample_uuids(fmt=DataFormat.NUMPY)
                continue

            domain, key, rep = col.split(".", maxsplit=2)
            col_key = (
                _ensure_domain_prefix(col, domain)
                if include_domain_prefix
                else _remove_domain_prefix(col, domain)
            )
            col_key = (
                _ensure_rep_suffix(col_key, rep)
                if include_rep_suffix
                else _remove_rep_suffix(col_key, rep)
            )

            data[col_key] = self._get_rep_data(
                domain=domain,
                key=key,
                rep=rep,
                fmt=DataFormat.NUMPY,
            )

        if fmt == DataFormat.DICT_NUMPY:
            return data

        return convert_dict_to_format(data, fmt=fmt, mode="stack", axis=1)

    def get_features(
        self,
        fmt: DataFormat = DataFormat.DICT_NUMPY,
        *,
        keys: str | list[str] | None = None,
        rep: str | None = REP_RAW,
        include_rep_suffix: bool = False,
        include_domain_prefix: bool = False,
    ):
        """
        Retrieve feature data in a chosen format.

        Args:
            fmt (DataFormat):
                Desired output format (see :class:`DataFormat`).
                Defaults to a single dictionary of numpy arrays.
            keys (str | list[str] | None):
                Optional subset of feature keys to return.
                If None, all feature keys are returned. Defaults to None.
            rep (str, optional):
                The representation (e.g., "raw" or "transformed") of the feature keys to
                return. If None, all representations are returned and `include_rep_suffix` is set to True.
                If specfied, `keys` must all have matching representations. Defaults to REP_RAW.
            include_rep_suffix (bool):
                Whether to include the representation suffix in the
                feature keys (e.g., "voltage" or "voltage.raw"). Defaults to False.
            include_domain_prefix (bool):
                Whether to include the domain prefix in the
                feature keys (e.g., "voltage" or "features.voltage"). Defaults to False.

        Returns:
            Feature data in the requested format.

        """
        return self._get_domain_data(
            domain=DOMAIN_FEATURES,
            fmt=fmt,
            keys=keys,
            rep=rep,
            include_rep_suffix=include_rep_suffix,
            include_domain_prefix=include_domain_prefix,
        )

    def get_targets(
        self,
        fmt: DataFormat = DataFormat.DICT_NUMPY,
        *,
        keys: str | list[str] | None = None,
        rep: str | None = REP_RAW,
        include_rep_suffix: bool = False,
        include_domain_prefix: bool = False,
    ):
        """
        Retrieve target data in a chosen format.

        Args:
            fmt (DataFormat):
                Desired output format (see :class:`DataFormat`).
                Defaults to a single dictionary of numpy arrays.
            keys (str | list[str] | None):
                Optional subset of target keys to return.
                If None, all target keys are returned. Defaults to None.
            rep (str, optional):
                The representation (e.g., "raw" or "transformed") of the target keys to
                return. If None, all representations are returned and `include_rep_suffix` is set to True.
                If specfied, `keys` must all have matching representations. Defaults to REP_RAW.
            include_rep_suffix (bool):
                Whether to include the representation suffix in the
                target keys (e.g., "voltage" or "voltage.raw"). Defaults to False.
            include_domain_prefix (bool):
                Whether to include the domain prefix in the
                target keys (e.g., "voltage" or "targets.voltage"). Defaults to False.

        Returns:
            Feature data in the requested format.

        """
        return self._get_domain_data(
            domain=DOMAIN_TARGETS,
            fmt=fmt,
            keys=keys,
            rep=rep,
            include_rep_suffix=include_rep_suffix,
            include_domain_prefix=include_domain_prefix,
        )

    def get_tags(
        self,
        fmt: DataFormat = DataFormat.DICT_NUMPY,
        *,
        keys: str | list[str] | None = None,
        rep: str | None = REP_RAW,
        include_rep_suffix: bool = False,
        include_domain_prefix: bool = False,
    ):
        """
        Retrieve tag data in a chosen format.

        Args:
            fmt (DataFormat):
                Desired output format (see :class:`DataFormat`).
                Defaults to a single dictionary of numpy arrays.
            keys (str | list[str] | None):
                Optional subset of tag keys to return.
                If None, all tag keys are returned. Defaults to None.
            rep (str, optional):
                The representation (e.g., "raw" or "transformed") of the tag keys to
                return. If None, all representations are returned and `include_rep_suffix` is set to True.
                If specfied, `keys` must all have matching representations. Defaults to REP_RAW.
            include_rep_suffix (bool):
                Whether to include the representation suffix in the
                tag keys (e.g., "voltage" or "voltage.raw"). Defaults to False.
            include_domain_prefix (bool):
                Whether to include the domain prefix in the
                tag keys (e.g., "voltage" or "tags.voltage"). Defaults to False.

        Returns:
            Feature data in the requested format.

        """
        return self._get_domain_data(
            domain=DOMAIN_TAGS,
            fmt=fmt,
            keys=keys,
            rep=rep,
            include_rep_suffix=include_rep_suffix,
            include_domain_prefix=include_domain_prefix,
        )

    def get_sample_uuids(self, fmt: DataFormat = DataFormat.NUMPY):
        """
        Retrieve sample UUIDs in this collection.

        Args:
            fmt (DataFormat):
                Desired output format (see :class:`DataFormat`).
                Defaults to a single dictionary of numpy arrays.

        Returns:
            Sample UUIDs in the requested format.

        """
        # Access the sample_id column as an Arrow array
        pa_arr = self.table.column(DOMAIN_SAMPLE_ID).combine_chunks()

        # Return raw pyarrow array is no format is specified
        if fmt is None:
            return pa_arr

        # Otherwise PyArrow's `to_numpy` for list-like arrays (1D/2D)
        # .to_numpy() returns an object-array, we need to convert to proper shape and dtype
        data = pa_arr.to_numpy(zero_copy_only=False)
        np_arr = stack_nested_numpy(data, (1,))
        return convert_to_format(np_arr, fmt=fmt)

    # ================================================
    # Controlled Mutation
    # ================================================
    # PyArrow tables are immutable, meaning any changes require a full
    # rebuild. This should be done very sparingly.
    # Current rules are:
    #   - Only representations of existing domain keys can be added/modified. Examples:
    #       - You can add/modify a "transformed" representation of an existing key
    #       - You can't add an entirely new key
    #   - The "raw" (default) representation cannot be modified
    #   - Meta data must be rebuilt for any changed columns
    #   - Any new representations must have the same number of samples
    #   - Mutations to the `table` attribute must happen inplace
    def add_rep(
        self,
        domain: str,
        key: str,
        rep: str,
        data: np.ndarray,
        *,
        overwrite: bool = False,
    ) -> None:
        """
        Add or overwrite a representation of an existing feature/target/tag key.

        Description:
            New representations can be added to an existing key. The `"raw"` (default) \
            representation cannot be overridden.

        Args:
            domain (str):
                One of {"features", "targets", "tags"}.
            key (str):
                Existing key within the domain (e.g., "voltage").
            rep (str):
                Representation name to add or overwrite (e.g., "transformed").
            data (np.ndarray):
                NumPy array of shape (n_samples, ...).
            overwrite (bool, optional):
                Whether to replace existing representation data if present. Defaults to False.

        Raises:
            ValueError: If the key does not exist or representation already exists (and overwrite=False).
            TypeError: If data is not a numpy array.

        """
        # Validate domain/key
        if key not in self._get_domain_keys(
            domain,
            include_domain_prefix=False,
            include_rep_suffix=False,
        ):
            msg = f"Cannot add representation for unknown key '{key}' in domain '{domain}'."
            raise ValueError(msg)

        # Check for valid representation
        if rep == REP_RAW:
            msg = f"The default representation ('{REP_RAW}') cannot be modified."
            raise ValueError(msg)

        # Check type and shape of data
        if not isinstance(data, np.ndarray):
            msg = f"Data must be a numpy array. Received: {type(data)}"
            raise TypeError(msg)
        if data.shape[0] != self.n_samples:
            msg = f"Data length {data.shape[0]} does not match number of samples {self.n_samples}."
            raise ValueError(msg)

        # Build Arrow array for the new representation column
        col_name = self._format_col_str(
            domain=domain,
            key=key,
            rep=rep,
            include_domain_prefix=True,
            include_rep_suffix=True,
        )
        _, new_arr, new_meta = numpy_to_sample_schema_column(
            name=key,
            data=data,
            rep=rep,
        )

        # Insert / replace column in table
        if col_name in self.table.column_names:
            if not overwrite:
                msg = f"Representation '{rep}' already exists for '{key}'. Set `overwrite=True` to modify."
                raise ValueError(msg)
            idx = self.table.schema.get_field_index(col_name)
            self.table = self.table.set_column(idx, col_name, new_arr)
        else:
            self.table = self.table.append_column(col_name, new_arr)

        # Update schema: register this representation's dtype
        if domain == DOMAIN_FEATURES:
            self.schema.features.setdefault(key, {})[rep] = new_arr.type
        elif domain == DOMAIN_TARGETS:
            self.schema.targets.setdefault(key, {})[rep] = new_arr.type
        else:
            self.schema.tags.setdefault(key, {})[rep] = new_arr.type

        # Update metadata
        meta = dict(self.table.schema.metadata or {})
        meta.update(new_meta)
        self.table = self.table.replace_schema_metadata(meta)

    def delete_rep(
        self,
        domain: str,
        key: str,
        rep: str,
    ) -> None:
        """
        Delete an existing representation of a feature/target/tag key.

        Description:
            The `"raw"` (default) representation cannot be deleted.

        Args:
            domain (str):
                One of {"features", "targets", "tags"}.
            key (str):
                Existing key within the domain (e.g., "voltage").
            rep (str):
                Representation name to add or overwrite (e.g., "transformed").

        Raises:
            ValueError: If the key does not exist or representation cannot be deleted.

        """
        # Validate domain/key
        if key not in self._get_domain_keys(
            domain,
            include_domain_prefix=False,
            include_rep_suffix=False,
        ):
            msg = f"Cannot delete representation for unknown key '{key}' in domain '{domain}'."
            raise ValueError(msg)

        # Check for valid representation
        if rep == REP_RAW:
            msg = f"The default representation ('{REP_RAW}') cannot be deleted."
            raise ValueError(msg)

        col_name = self._format_col_str(
            domain=domain,
            key=key,
            rep=rep,
            include_domain_prefix=True,
            include_rep_suffix=True,
        )
        if col_name not in self.table.column_names:
            msg = f"Column `{col_name}` does not exist in table."
            raise KeyError(msg)

        # Remove the column from the table
        idx = self.table.schema.get_field_index(col_name)
        self.table = self.table.remove_column(idx)

        # Update schema: drop this representation
        if domain == DOMAIN_FEATURES:
            self.schema.features[key].pop(rep, None)
        elif domain == DOMAIN_TARGETS:
            self.schema.targets[key].pop(rep, None)
        else:
            self.schema.tags[key].pop(rep, None)

        # Remove metadata entries for this representation
        meta = dict(self.table.schema.metadata or {})
        to_delete = [
            k
            for k in meta
            if k.decode().startswith(f"{METADATA_PREFIX}.{domain}.{key}.{rep}.")
        ]
        for k in to_delete:
            del meta[k]
        self.table = self.table.replace_schema_metadata(meta)

    # ================================================
    # Export / conversion
    # ================================================
    def copy(
        self,
        *,
        raw_only: bool = False,
        deep: bool = False,
    ) -> SampleCollection:
        """
        Create a copy of this SampleCollection.

        Description:
            Returns a new SampleCollection instance with controlled
            duplication of the underlying PyArrow table.

            - If raw_only=True, only REP_RAW representations (and sample_id)
              are copied. All transformed representations are excluded.
            - If deep=True, Arrow buffers are duplicated via pandas roundtrip.
              Otherwise, zero-copy buffer sharing is used.

        Args:
            raw_only (bool, optional):
                If True, include only REP_RAW columns and sample_id.
                Defaults to False.

            deep (bool, optional):
                If True, perform a deep copy of Arrow buffers.
                If False, underlying Arrow buffers are shared (zero-copy).
                Defaults to False.

        Returns:
            SampleCollection:
                A new SampleCollection instance.

        """
        # Determine columns to copy, and build new table
        if raw_only:
            cols = []
            for domain in self.available_domains:
                if domain == DOMAIN_SAMPLE_ID:
                    cols.append(DOMAIN_SAMPLE_ID)
                    continue
                if (domain == DOMAIN_TAGS) and not self.has_tags:
                    continue

                # Record all REP_RAW columns
                for key in self.schema.domain_keys(domain=domain):
                    if REP_RAW in self.schema.rep_keys(
                        domain=domain,
                        key=key,
                    ):
                        col_name = self._format_col_str(
                            domain=domain,
                            key=key,
                            rep=REP_RAW,
                            include_domain_prefix=True,
                            include_rep_suffix=True,
                        )
                        cols.append(col_name)

            new_table = self.table.select(cols)

        else:
            new_table = self.table

        # Deep copy (if requested)
        if deep:
            new_table = pa.Table.from_pandas(new_table.to_pandas())

        # Build new sample collection
        return SampleCollection(table=new_table)

    def select(
        self,
        columns: str | list[str],
    ) -> SampleCollection:
        """
        Selects only the specified columns and returns a new SampleCollection.

        Args:
            columns (str | list[str]):
                Columns to include in the returned SampleCollection.

        Returns:
            SampleCollection:
                The returned collection shares internal buffers with the original.
                It is not a deep copy.

        """
        columns = ensure_list(columns)
        return SampleCollection(table=self.table.select(columns))

    def to_dict(self) -> dict[str, np.ndarray]:
        """
        Convert the entire SampleCollection into a flattened dict of NumPy arrays.

        Description:
            Returns a unified dict containing all domains (features, targets, \
            and tags) as columns. Each structured domain (e.g., features struct) \
            is flattened into individual keys named after their field keys.

        Returns:
            dict[str, np.ndarray]:
                A flattened dict containing all domains and the sample_id column.

        """
        # Gather domain data (flattened)
        all_data: dict[str, np.ndarray] = {}
        for domain in [DOMAIN_FEATURES, DOMAIN_TARGETS] + (
            [DOMAIN_TAGS] if self.has_tags else []
        ):
            d_res = self._get_domain_data(
                domain=domain,
                fmt=DataFormat.DICT_NUMPY,
                keys=None,
                rep=None,
                include_domain_prefix=True,
                include_rep_suffix=True,
            )
            all_data |= d_res

        # Always include sample_id column (guaranteed unique)
        uuids = self.get_sample_uuids(fmt=DataFormat.NUMPY)
        all_data[DOMAIN_SAMPLE_ID] = stack_nested_numpy(uuids, shape=())

        return all_data

    def to_pandas(
        self,
    ) -> pd.DataFrame:
        """
        Convert the entire SampleCollection into a flattened pandas DataFrame.

        Description:
            Returns a unified DataFrame containing all domains (features, targets, \
            and tags) as columns. Each structured domain (e.g., features struct) \
            is flattened into individual columns named after their field keys.

        Returns:
            pd.DataFrame:
                A flattened DataFrame containing all domains and the sample_id column.

        """
        # Gather domain data (flattened)
        # Each element may be a multi-dim array, need to separate 1st dimension with list()
        reshaped_data = {k: list(v) for k, v in self.to_dict().items()}
        return pd.DataFrame(reshaped_data)

    def save(self, path: str | Path) -> Path:
        """
        Save SampleCollection to a single file using Arrow IPC (Feather v2).

        Description:
            This method fully fully preserves schema, domain struct columns, \
            representations, metadata (shapes/dtypes), and sample_id column.
            The `load` class method should be used for safe round-trip loading.

        Args:
            path (str | Path): Destination file path (e.g., "dataset.arrow").

        Returns:
            Path: Where collection was saved.

        """
        path = Path(path).with_suffix(".arrow")
        with pa.OSFile(str(path), "wb") as sink:
            writer = pa.ipc.new_file(sink, self.table.schema)
            writer.write(self.table)
            writer.close()
        return path

    @classmethod
    def load(cls, path: str | Path) -> SampleCollection:
        """
        Load a SampleCollection previously saved with `.save()`.

        Description:
            All metadata, representations, domains, and sample_id column \
            are restored exactly as originally saved.

        Args:
            path (str | Path): File path to load from.

        Returns:
            SampleCollection: A fully reconstructed instance.

        """
        path = Path(path)
        with pa.OSFile(str(path), "rb") as source:
            reader = pa.ipc.open_file(source)
            table = reader.read_all()

        # Schema is inferred back from metadata inside table
        return cls(table=table)
