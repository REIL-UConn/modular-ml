"""PyArrow conversion helpers for ModularML sample domains."""

from __future__ import annotations

import ast
import fnmatch
import hashlib
from collections import defaultdict
from typing import Literal

import numpy as np
import pyarrow as pa

from modularml.core.data.sample_schema import (
    DTYPE_SUFFIX,
    METADATA_PREFIX,
    SHAPE_SUFFIX,
)
from modularml.core.data.schema_constants import (
    DOMAIN_FEATURES,
    DOMAIN_SAMPLE_UUIDS,
    DOMAIN_TAGS,
    DOMAIN_TARGETS,
    REP_RAW,
    REP_TRANSFORMED,
)
from modularml.utils.data.formatting import ensure_list
from modularml.utils.data.shape_utils import ensure_tuple_shape, get_shape
from modularml.utils.logging.warnings import warn


def resolve_column_selectors(
    *,
    all_columns: list[str],
    columns: str | list[str] | None = None,
    features: str | list[str] | None = None,
    targets: str | list[str] | None = None,
    tags: str | list[str] | None = None,
    rep: str | None = None,
    include_all_if_empty: bool = False,
) -> dict[str, set[str]]:
    """
    Resolve column selectors into fully-qualified column names.

    Args:
        all_columns (list[str]):
            All available columns in the table being filtered.

        columns (str | list[str] | None):
            Fully-qualified column names to include (e.g., `features.voltage.raw`).
            Must have an exact match in `all_columns`.

        features (str | list[str] | None):
            Feature-domain selectors. Accepts exact names or wildcard (`*`) patterns.
            Domain prefix `features.` may be omitted.

        targets (str | list[str] | None):
            Same as `features`, but for the targets domain.

        tags (str | list[str] | None):
            Same as `features`, but for the tags domain.

        rep (str | None):
            Default representation suffix to apply when a selector provides no representation.
            Does *not* overwrite explicit representations in `columns`.

        include_all_if_empty (bool):
            If True, domains with no explicitly selected columns will include
            all available columns from that domain. Defaults to False.

    Returns:
        dict[str, set[str]]:
            Mapping of domain to fully-qualified column names
            (e.g., `{"features": {"features.voltage.raw"}}`).

    Raises:
        KeyError: If a requested column or pattern does not match any column.
        ValueError: If a selector has an invalid format or mismatched domain.

    """
    # Build final column selection (organized by domain)
    selected: dict[str, set[str]] = {
        DOMAIN_FEATURES: set(),
        DOMAIN_TARGETS: set(),
        DOMAIN_TAGS: set(),
    }

    # 1. Extract columns defined via `columns` argument
    for col in ensure_list(columns):
        if col not in all_columns:
            msg = f"Columns '{col}' does not exist. Available columns: {all_columns}"
            raise KeyError(msg)
        selected[col.split(".")[0]].add(col)

    # 2. Domain-based arguments
    domain_inputs = {
        DOMAIN_FEATURES: ensure_list(features),
        DOMAIN_TARGETS: ensure_list(targets),
        DOMAIN_TAGS: ensure_list(tags),
    }
    for domain, selectors in domain_inputs.items():
        for sel in selectors:
            # Normalize to domain.key.rep (rep may be wildcard or missing)
            full = _ensure_domain_prefix(sel, domain)
            parts = full.split(".")

            if len(parts) not in (2, 3):
                msg = f"Invalid selector '{sel}'. Expected 'key', 'key.rep', or wildcard form."
                raise ValueError(msg)
            d, k, r = parts[0], parts[1], (parts[2] if len(parts) == 3 else None)

            # Apply default rep if missing
            if r is None and rep is not None:
                r = rep
            pattern = f"{d}.{k}.{r or '*'}"
            matched = fnmatch.filter(all_columns, pattern)
            if not matched:
                msg = f"No entries in `all_columns` match selector '{sel}' (expanded to '{pattern}')."
                raise KeyError(msg)

            # Add to selected
            for m in matched:
                selected[domain].add(m)

    # 3. Fill empty domain, if specified
    if include_all_if_empty:
        for d, cols in selected.items():
            if not cols:
                selected[d].update(c for c in all_columns if c.startswith(f"{d}."))  # noqa: PLR1733

            # Raise warning if multiple representations of the same columns are being used
            key_to_reps: dict[str, set[str]] = defaultdict(set)
            for c in cols:
                _, k, r = c.split(".")
                key_to_reps[k].add(r)
            multi_rep_keys = {k: list(v) for k, v in key_to_reps.items() if len(v) > 1}
            if multi_rep_keys:
                msg = (
                    f"Multiple representations selected for the same {d} column(s): "
                    + ", ".join(f"{k} -> {reps}" for k, reps in multi_rep_keys.items())
                    + ". This may lead to unintended model inputs if not handled explicitly."
                )
                warn(msg, category=UserWarning, stacklevel=2)

    return selected


def _remove_domain_prefix(element: str, domain: str) -> str:
    """
    Remove the domain prefix from `element`, if present.

    Args:
        element (str): Fully-qualified column name.
        domain (str): Domain prefix to remove (e.g., `features`).

    Returns:
        str: Column name without the domain prefix.

    """
    prefix = f"{domain}."
    return (
        element[element.rindex(prefix) + len(prefix) :]
        if prefix in element
        else element
    )


def _ensure_domain_prefix(element: str, domain: str) -> str:
    """
    Ensure the domain prefix is present on `element`.

    Args:
        element (str): Column selector to normalize.
        domain (str): Expected domain name.

    Returns:
        str: Selector guaranteed to start with `domain`.

    Raises:
        ValueError: If `element` already references a different domain.

    """
    prefix = f"{domain}."

    all_domains = [DOMAIN_FEATURES, DOMAIN_TARGETS, DOMAIN_TAGS]
    invalid_domains = [y for y in all_domains if y != domain]
    invalid_prefixes = [f"{x}." for x in invalid_domains]

    if element.startswith(prefix):
        return element
    if any(element.startswith(f"{x}.") for x in invalid_prefixes):
        msg = f"Domain mistmatch: '{element}' starts with domain other than '{domain}'"
        raise ValueError(msg)

    return prefix + element


def _remove_rep_suffix(element: str, rep: str) -> str:
    """
    Remove the representation suffix from `element`, if present.

    Args:
        element (str): Fully-qualified column name.
        rep (str): Representation suffix to remove.

    Returns:
        str: Column without the specified suffix.

    """
    suffix = f".{rep}"
    return element[: element.rindex(suffix)] if suffix in element else element


def _ensure_rep_suffix(element: str, rep: str) -> str:
    """
    Ensure the representation suffix exists on `element`.

    Args:
        element (str): Column selector to normalize.
        rep (str): Representation suffix to enforce.

    Returns:
        str: Column guaranteed to end with the suffix.

    """
    suffix = f".{rep}"
    return element if element.endswith(suffix) else element + suffix


def flatten_schema(schema: pa.Schema, prefix: str = "", separator: str = "."):
    """
    Return flattened column paths for a (possibly nested) Arrow schema.

    Args:
        schema (pa.Schema): Schema to flatten.
        prefix (str, optional): Prefix applied to every nested path. Defaults to "".
        separator (str, optional): Separator between nested levels. Defaults to ".".

    Returns:
        list[str]: Flattened column paths including nested field names.

    """
    cols = []
    for field in schema:
        name = f"{prefix}{field.name}"

        if pa.types.is_struct(field.type):
            # Recurse into struct fields
            child_schema = pa.schema(field.type)
            cols.extend(flatten_schema(child_schema, prefix=name + separator))
        else:
            cols.append(name)

    return cols


def normalize_numpy_dtype_for_pyarrow_dtype(dtype) -> str:
    """
    Normalize dtype across NumPy and Arrow for consistent metadata storage.

    Returns a semantic type string like 'float32', 'int64', 'string', etc.

    Args:
        dtype (str | np.dtype): NumPy dtype or dtype string to normalize.

    Returns:
        str: Semantic dtype string describing the element width.

    """
    if isinstance(dtype, str):
        dtype = np.dtype(dtype)

    if np.issubdtype(dtype, np.floating):
        return f"float{dtype.itemsize * 8}"
    if np.issubdtype(dtype, np.integer):
        return f"int{dtype.itemsize * 8}"
    if np.issubdtype(dtype, np.bool_):
        return "bool"
    if np.issubdtype(dtype, np.str_) or np.issubdtype(dtype, np.object_):
        return "string"
    # fallback for unknown / complex types
    return str(pa.from_numpy_dtype(dtype))


def get_shape_of_pyarrow_array(
    arr: pa.Array,
    *,
    include_nrows: bool = True,
) -> tuple[int, ...]:
    """
    Infer the shape of a PyArrow array using sampled rows.

    Args:
        arr (pa.Array): Array whose element shape should be inferred.
        include_nrows (bool, optional): Include the row dimension in the returned shape.

    Returns:
        tuple[int, ...]: Best-effort shape for the provided array.

    """
    # Get num rows
    n_rows = len(arr)
    if n_rows == 0:
        return ()

    detected: tuple[int, ...] | None = None
    for i in range(min(n_rows, 10)):  # only sample first 10 rows for speed
        if not arr[i].is_valid:
            continue
        s = get_shape(arr[i].as_py())
        s = ensure_tuple_shape(
            shape=s.get("__shape__") if isinstance(s, dict) else s,
            min_len=1,
            max_len=None,
            allow_null_shape=False,
        )
        if detected is None:
            detected = s
        elif detected != s:
            detected = None
            break
    if include_nrows:
        return (n_rows, *detected) if detected else (n_rows,)

    return detected if detected else ()


def get_dtype_of_pyarrow_array(arr: pa.Array) -> str:
    """
    Infer array data type directly from contents.

    Args:
        arr (pa.Array): Array whose dtype should be described.

    Returns:
        str: Element data type string (e.g., `float32`, `int64`, `bytes`).

    """

    def arrow_dtype_string(dtype: pa.DataType) -> str:
        """
        Return a precise string representation of a PyArrow DataType.

        Preserves bit width (e.g. 'float32', 'int64').
        """
        if pa.types.is_binary(dtype):
            return "binary"
        if pa.types.is_string(dtype):
            return "string"
        if pa.types.is_boolean(dtype):
            return "bool"
        if pa.types.is_floating(dtype):
            return f"float{dtype.bit_width}"
        if pa.types.is_integer(dtype):
            return f"{'int' if dtype.bit_width > 0 else 'uint'}{dtype.bit_width}"
        return str(dtype)

    t = arr.type

    # Case 1: nested list (tensor-like)
    if pa.types.is_list(t):
        base = t
        while pa.types.is_list(base):
            base = base.value_type
        return base.to_pandas_dtype().__name__

    # Case 2: binary blobs
    if pa.types.is_binary(t) or pa.types.is_fixed_size_binary(t):
        return "bytes"

    # Case 3: primitive (int, float, string, etc.)
    return arrow_dtype_string(t)


def make_nested_list_type(base_type: pa.DataType, depth: int) -> pa.DataType:
    """
    Recursively build a nested PyArrow ListType (e.g., list<list<...<float32>>>).

    Args:
        base_type (pa.DataType): Base element type for the innermost tensor.
        depth (int): Number of nested list levels to create.

    Returns:
        pa.DataType: Nested list type with the specified depth.

    """
    typ = base_type
    for _ in range(depth):
        typ = pa.list_(typ)
    return typ


def numpy_to_sample_schema_column(
    name: str,
    data: np.ndarray,
    *,
    rep: Literal["raw", "transformed"] = "raw",
    dtype: pa.DataType | None = None,
    store_shape_metadata: bool = True,
    max_list_depth: int = 2,
) -> tuple[str, pa.StructArray, dict[bytes, bytes]]:
    """
    Convert a NumPy array into a PyArrow column suitable for :class:`SampleCollection`.

    This function automatically chooses between:
      - Nested list arrays (for rank <= `max_list_depth`)
      - Binary blobs with shape metadata (for higher-rank tensors)

    Args:
        name (str): Column name.
        data (np.ndarray): NumPy array of shape `(N, ...)`.
        rep (Literal["raw", "transformed"]): Representation name used in the schema.
        dtype (pa.DataType | None): Optional element dtype. Defaults to `pa.from_numpy_dtype(data.dtype)`.
        store_shape_metadata (bool): Attach shape metadata for reconstruction.
        max_list_depth (int): Maximum nesting depth for list-based encoding.

    Returns:
        tuple[str, pa.StructArray, dict[bytes, bytes]]: Column name, Arrow array, and metadata dict.

    Raises:
        TypeError: If `data` is not a NumPy array.
        ValueError: If `data` is empty or scalar.

    """
    if rep not in [REP_RAW, REP_TRANSFORMED]:
        msg = f"`rep` must be one of: {[REP_RAW, REP_TRANSFORMED]}"
        raise ValueError(msg)
    if not isinstance(data, np.ndarray):
        msg = f"Expected NumPy array, got {type(data)}"
        raise TypeError(msg)

    if data.ndim == 0:
        msg = (
            "Cannot store scalar arrays as tensor columns (need at least 1 dimension)."
        )
        raise ValueError(msg)

    if dtype is None:
        dtype = pa.from_numpy_dtype(data.dtype)

    # Drop trailing singleton dimensions (e.g., (N,1) to (N,))
    if data.ndim > 1 and data.shape[-1] == 1:
        data = data.squeeze(-1)

    n_samples = data.shape[0]
    shape = data.shape[1:]

    # Case 1: small tensors (1D, 2D) -> store as nested list
    if data.ndim <= max_list_depth:
        # Build recursive list type matching rank
        element_type = pa.from_numpy_dtype(data.dtype)
        list_type = make_nested_list_type(element_type, depth=data.ndim - 1)
        array = pa.array(data.tolist(), type=list_type)

    # Case 2: high-rank tensors -> store as binary
    else:
        # Convert each row (sample) to bytes
        element_size = (
            data[0].nbytes
            if isinstance(data[0], np.ndarray)
            else np.prod(shape) * data.itemsize
        )
        array = pa.array(
            [data[i].tobytes() for i in range(n_samples)],
            type=pa.binary(element_size),
        )

    # Metadata for shape reconstruction
    metadata = {}
    if store_shape_metadata:
        meta_dtype = normalize_numpy_dtype_for_pyarrow_dtype(data.dtype).encode()
        metadata = {
            f"{name}.{rep}.{SHAPE_SUFFIX}".encode(): str(shape).encode(),
            f"{name}.{rep}.{DTYPE_SUFFIX}".encode(): meta_dtype,
        }

    return name, array, metadata


def pyarrow_array_to_numpy(
    array: pa.Array,
    *,
    shape: tuple[int, ...] | None = None,
    dtype: np.dtype | None = None,
    metadata: dict[bytes, bytes] | None = None,
) -> np.ndarray:
    """
    Reconstruct a NumPy array from an Arrow tensor column.

    Supports both list-array and binary encodings.

    Args:
        array (pa.Array): PyArrow array (list, nested list, or binary).
        shape (tuple[int, ...] | None): Optional shape override for binary encodings.
        dtype (np.dtype | None): Optional dtype override.
        metadata (dict[bytes, bytes] | None): Schema metadata for dtype/shape lookup.

    Returns:
        np.ndarray: Reconstructed tensor.

    Raises:
        ValueError: If shape information is missing for binary encodings.
        TypeError: If the Arrow type is unsupported.

    """
    # Infer dtype automatically
    inferred_dtype = None

    if pa.types.is_list(array.type):
        value_type = array.type
        # Recursively descend to base element type (for nested lists)
        while pa.types.is_list(value_type):
            value_type = value_type.value_type
        inferred_dtype = value_type.to_pandas_dtype()

    elif pa.types.is_binary(array.type) or pa.types.is_fixed_size_binary(array.type):
        # Try from metadata
        if metadata is not None:
            for key in metadata:
                if key.endswith(b".dtype"):
                    inferred_dtype = np.dtype(metadata[key].decode())
                    break

    # Fallback to explicit dtype or np.float32
    dtype = dtype or inferred_dtype or np.float32

    # Decode data
    if pa.types.is_list(array.type):
        # Nested list representation
        pydata = array.to_pylist()
        return np.array(pydata, dtype=dtype)

    if pa.types.is_binary(array.type) or pa.types.is_fixed_size_binary(array.type):
        # Infer shape from metadata if not provided
        if shape is None and metadata is not None:
            for key in metadata:
                if key.endswith(b".shape"):
                    shape = ast.literal_eval(metadata[key].decode())
                    break
        if shape is None:
            raise ValueError("Shape must be provided for binary-encoded tensors.")

        n_rows = len(array)
        # Convert each element back into a NumPy array
        return np.stack(
            [
                np.frombuffer(array[i].as_buffer(), dtype=dtype).reshape(shape)
                for i in range(n_rows)
            ],
            axis=0,
        )

    msg = f"Unsupported Arrow tensor column type: {array.type}"
    raise TypeError(msg)


def build_sample_schema_table(
    *,
    features: dict[str, np.ndarray],
    targets: dict[str, np.ndarray],
    tags: dict[str, np.ndarray] | None = None,
) -> pa.Table:
    """
    Build a metadata-preserving PyArrow table compatible with :class:`SampleCollection`.

    Description:
        Converts three dictionaries of NumPy arrays (`features`, `targets`, and `tags`)
        into a structured Arrow table. Each domain becomes a Struct column whose fields
        are per-column structs containing a single representation field: `raw`.

        Example layout:
            features: struct<
                voltage: struct<raw: list<float32>>,
                current: struct<raw: list<float32>>
            >

        Each leaf array is stored as a PyArrow array, and accompanying metadata
        (shape, dtype) is attached to the table schema for full reconstruction.

        Metadata keys follow the namespaced convention:
            - `features.voltage.raw.shape`
            - `features.voltage.raw.dtype`

    Args:
        features (dict[str, np.ndarray]): Feature columns keyed by name.
        targets (dict[str, np.ndarray]): Target columns keyed by name.
        tags (dict[str, np.ndarray] | None): Optional tag columns keyed by name.

    Returns:
        pa.Table: PyArrow table with structured domain columns and metadata.

    """

    def _encode_domain(
        domain_name: str,
        domain_dict: dict[str, np.ndarray],
        meta_prefix: str | None = None,
    ):
        """
        Convert a domain dict into Arrow columns and metadata.

        Args:
            domain_name (str): Domain prefix (e.g., `features`).
            domain_dict (dict[str, np.ndarray]): Mapping of column names to arrays.
            meta_prefix (str | None): Optional metadata key prefix.

        Returns:
            tuple[dict[str, pa.Array], dict[bytes, bytes]]: Column mapping and metadata.

        """
        cols: dict[str, pa.Array] = {}
        metadata = {}

        for key, np_arr in domain_dict.items():
            name, arrow_arr, meta = numpy_to_sample_schema_column(
                name=key,
                data=np_arr,
                rep=REP_RAW,
            )

            # Full col name
            col_name = f"{domain_name}.{name}.{REP_RAW}"
            cols[col_name] = arrow_arr

            # Prefix metadata keys if provided
            if meta_prefix:
                for k, v in meta.items():
                    # Handle both bytes and string keys gracefully
                    base = k.decode() if isinstance(k, bytes) else str(k)
                    full = f"{meta_prefix}.{base}"
                    metadata[full.encode()] = v
            else:
                metadata.update(meta)

        return cols, metadata

    # Assume all domains have same length
    n_rows = len(next(iter(features.values())))

    # Build each domain
    feature_cols, feature_meta = _encode_domain(
        domain_name=DOMAIN_FEATURES,
        domain_dict=features,
        meta_prefix=f"{METADATA_PREFIX}.{DOMAIN_FEATURES}",
    )
    target_cols, target_meta = _encode_domain(
        domain_name=DOMAIN_TARGETS,
        domain_dict=targets,
        meta_prefix=f"{METADATA_PREFIX}.{DOMAIN_TARGETS}",
    )
    tag_cols, tag_meta = _encode_domain(
        domain_name=DOMAIN_TAGS,
        domain_dict=tags or {},
        meta_prefix=f"{METADATA_PREFIX}.{DOMAIN_TAGS}",
    )

    # Build flat column dict
    all_cols = {**feature_cols, **target_cols, **tag_cols}

    # Ensure table has at least 1 column
    if len(all_cols) == 0:
        # Create an "empty" placeholder column of correct row count
        all_cols = {DOMAIN_SAMPLE_UUIDS: pa.array([None] * n_rows, type=pa.string())}

    # Build table
    table = pa.table(all_cols)

    # Merge and attach metadata
    all_metadata = {**feature_meta, **target_meta, **tag_meta}
    if table.schema.metadata:
        all_metadata.update(table.schema.metadata)
    table = table.replace_schema_metadata(all_metadata)

    return table


def hash_pyarrow_table(table: pa.Table) -> str:
    """
    Compute a deterministic SHA-256 digest for a PyArrow table's contents.

    Args:
        table (pa.Table): Table to hash.

    Returns:
        str: Hexadecimal digest of the table contents.

    """
    t = table.combine_chunks()
    h = hashlib.sha256()

    for b in t.to_batches():
        for c in b.columns:
            h.update(c.to_pylist().__repr__().encode())

    return h.hexdigest()
