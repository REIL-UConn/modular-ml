"""Shape inference utilities shared across ModularML data utilities."""

from collections.abc import Iterable
from typing import Any

import numpy as np

from modularml.utils.environment.optional_imports import (
    check_pandas,
    check_tensorflow,
    check_torch,
)

ShapeLike = tuple[int, ...] | dict[str, Any]


def shapes_similar_except_singleton(
    shape_a: tuple[int, ...],
    shape_b: tuple[int, ...],
) -> bool:
    """
    Returns True if two shapes are equal except for singleton (size-1) dimensions.

    Examples:
        (32, 1, 4) ≈ (32, 4)        → True
        (1, 64, 1, 1) ≈ (64,)       → True
        (16, 8, 4) ≈ (8, 16, 4)     → False  (order mismatch)
        (32, 2, 4) ≈ (32, 4)        → False  (non-singleton mismatch)

    Args:
        shape_a (tuple[int, ...]): First shape.
        shape_b (tuple[int, ...]): Second shape.

    Returns:
        bool: True if the shapes are equivalent ignoring singleton dimensions.

    """
    # Remove singleton dimensions
    reduced_a = tuple(dim for dim in shape_a if dim != 1)
    reduced_b = tuple(dim for dim in shape_b if dim != 1)

    return reduced_a == reduced_b


def get_shape(x) -> ShapeLike:
    """
    Infer the structural shape of an object in a framework-agnostic way.

    This function provides a unified interface for inspecting the "shape"
    (dimensionality and consistency of elements) of data across a variety
    of Python and machine learning data types.

    Supported types:
    - **Scalars**: int, float, bool, str, bytes -> returns ()
    - **NumPy arrays** (`np.ndarray`) -> returns tuple of ints
    - **Torch tensors** (`torch.Tensor`) -> returns tuple of ints
    - **TensorFlow tensors** (`tf.Tensor`) -> returns tuple of ints
      (with `-1` for unknown dimensions)
    - **Pandas Series** (`pd.Series`) -> returns (n_rows, *element_shape)
    - **Pandas DataFrames** (`pd.DataFrame`) -> returns a nested dict:
        {
            "__shape__": (n_rows, n_cols),
            "columns": {
                col_name: ShapeLike,  # shape per column (recursively computed)
            }
        }
    - **Sequences** (`list`, `tuple`) -> returns:
        * (len, *element_shape) if all elements share the same shape
        * (len,) otherwise
        * For nested non-tuple shapes (e.g., list of dicts), returns:
            {"__shape__": (len,), "element": ShapeLike}
    - **Generic objects with `.shape` attribute** -> returns tuple(s)

    Args:
        x (Any): Object whose structural shape should be inferred.

    Returns:
        ShapeLike: Tuple or nested mapping describing the shape.

    Conventions:
        - Scalars -> ()
        - Empty sequence -> (0,)
        - Unknown or variable-length inner dimension -> -1
        - Tabular/nested data -> dict with "__shape__" and recursive entries

    Raises:
        TypeError: If the object's type is unsupported and its shape cannot be inferred.

    Examples:
        >>> get_shape(42)
        ()
        >>> get_shape(np.zeros((3, 4)))
        (3, 4)
        >>> get_shape(torch.zeros(2, 5))
        (2, 5)
        >>> get_shape([[1, 2], [3, 4]])
        (2, 2)
        >>> get_shape(pd.Series([[1, 2], [3, 4], [5, 6]]))
        (3, 2)
        >>> get_shape(pd.DataFrame({"a": [1, 2], "b": [[1, 2, 3], [4, 5, 6]]}))
        {'__shape__': (2, 2), 'columns': {'a': (), 'b': (3,)}}

    """
    torch = check_torch()
    tf = check_tensorflow()
    pd = check_pandas()

    # --- None or scalars ---
    if x is None or isinstance(x, (int, float, complex, bool, str, bytes)):
        return ()

    # --- NumPy ---
    if isinstance(x, np.ndarray):
        return tuple(map(int, x.shape))

    # --- Torch ---
    if torch is not None and isinstance(x, torch.Tensor):
        return tuple(map(int, x.shape))

    # --- TensorFlow ---
    if tf is not None and isinstance(x, tf.Tensor):
        # tf.TensorShape behaves like tuple but may have None dims
        return tuple(int(d) if d is not None else -1 for d in x.shape)

    # --- Pandas ---
    if pd is not None and isinstance(x, (pd.Series, pd.DataFrame)):
        if isinstance(x, pd.Series):
            first = x.iloc[0]
            inner_shape = get_shape(first)
            return (len(x), *inner_shape)

        if isinstance(x, pd.DataFrame):
            all_shapes = {}
            for c in x.columns:
                first = x[c].iloc[0]
                all_shapes[c] = get_shape(first)
            return {"__shape__": (len(x), len(x.columns)), "columns": all_shapes}

    if isinstance(x, dict):
        all_shapes = {}
        for k, v in x.items():
            all_shapes[k] = get_shape(v)

        return {"__shape__": (len(x),), "keys": all_shapes}

    # --- Python sequences ---
    if isinstance(x, (list, tuple)):
        if not x:
            return (0,)
        first_shape = get_shape(x[0])
        # only append if all elements share same shape
        if all(get_shape(e) == first_shape for e in x[1:]):
            if isinstance(first_shape, tuple):
                return (len(x), *first_shape)
            return {"__shape__": (len(x),), "element": first_shape}
        return (len(x),)

    # --- Objects with .shape attribute (fallback) ---
    if hasattr(x, "shape"):
        s = x.shape
        if isinstance(s, (tuple, list)):
            return tuple(s)
        try:
            return tuple(s())  # callable shape
        except TypeError:
            return (s,) if isinstance(s, int) else ()

    msg = f"Unable to infer shape for object with type: {type(x)}"
    raise TypeError(msg)


def ensure_tuple_shape(
    shape: Iterable,
    *,
    min_len: int = 1,
    max_len: int | None = None,
    allow_null_shape: bool = False,
) -> tuple[int, ...] | None:
    """
    Formats the given shape iterable into a tuple of integers.

    Description:
        The iterable (eg: `[101, 5]`) is cast to a tuple of integers.
        If the tuple has fewer elements than `min_len`, singleton dimensions
        are inserted at the beginning.
        If the tuple has more elements than `max_len`, removal of singleton
        dimensions is attempted.
        If the length of the tuple is still more than `max_len`, an error
        is raised.

    Args:
        shape (Iterable): Shape values.
        min_len (int): Minimum number of elements in the returned tuple.
        max_len (int | None): Maximum number of elements allowed in the tuple.
        allow_null_shape (bool): Return `None` when the input is `None`.

    Returns:
        tuple[int, ...] | None: Normalized shape tuple or `None`.

    Raises:
        TypeError: If `shape` is not iterable.
        ValueError: If dimensions are negative or cannot satisfy `max_len`.

    """
    if shape is None:
        if allow_null_shape:
            return None
        raise ValueError("Shape cannot be None.")

    # Convert to tuple[int]
    try:
        shape_tuple = tuple(int(x) for x in shape)
    except Exception as exc:
        msg = f"Shape must be an iterable of integers. Received: {shape}"
        raise TypeError(msg) from exc

    # Validate non-negative dimensions
    if any(d < 0 for d in shape_tuple):
        msg = f"Shape dimensions must be non-negative. Received: {shape_tuple}"
        raise ValueError(msg)

    # Pad with leading singleton dimensions if too short
    if len(shape_tuple) < min_len:
        shape_tuple = (1,) * (min_len - len(shape_tuple)) + shape_tuple

    # Reduce singleton dimensions if too long
    if max_len is not None and len(shape_tuple) > max_len:
        # Remove singleton dims from the left first (least informative)
        reduced = list(shape_tuple)
        i = 0
        while len(reduced) > max_len and i < len(reduced):
            if reduced[i] == 1:
                reduced.pop(i)
            else:
                i += 1

        if len(reduced) > max_len:
            msg = (
                f"Cannot reduce shape {shape_tuple} to max_len={max_len} "
                "without removing non-singleton dimensions."
            )
            raise ValueError(msg)

        shape_tuple = tuple(reduced)

    return shape_tuple
