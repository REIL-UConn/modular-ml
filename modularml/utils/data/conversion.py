"""Data conversion helpers shared across ModularML utilities and backends."""

from typing import Any, Literal

import numpy as np
import pandas as pd

from modularml.utils.data.data_format import (
    DataFormat,
    format_requires_compatible_shapes,
    normalize_format,
)
from modularml.utils.environment.optional_imports import (
    check_tensorflow,
    check_torch,
    ensure_tensorflow,
    ensure_torch,
)
from modularml.utils.errors.error_handling import ErrorMode
from modularml.utils.logging.warnings import warn
from modularml.utils.nn.backend import Backend, infer_backend


def flatten_to_2d(arr: np.ndarray, merged_axes: int | tuple[int]):
    """
    Flatten an N-D array into 2D by *merging* a set of axes.

    Description:
        The axes in `merged_axes` are multiplied together to form one \
        dimension of the 2D output; all remaining axes form the other dimension.
        If axis 0 is included, merged axes become the first dimension \
        (samples); otherwise they become the second dimension (features).

    Args:
        arr (np.ndarray): Input N-D array.
        merged_axes (int | tuple[int]): Axes whose sizes are merged into a single dimension.

    Returns:
        tuple[np.ndarray, dict[str, tuple]]:
            - Flattened 2D array of shape `(A, B)`.
            - Metadata for reversing the operation (`original_shape`, `merged_axes`).

    Example:
        X.shape  # (1000, 3, 16, 16)
        Y = flatten_to_2d(X, (2, 3))
        Y.shape  # (3000, 256)

    """
    arr = np.asarray(arr)
    original_shape = arr.shape

    # Axes to merge and resulting size
    merged_axes = (merged_axes,) if isinstance(merged_axes, int) else tuple(merged_axes)
    merged_size = int(np.prod([original_shape[i] for i in merged_axes]))

    # Axes to flatten
    flattened_axes = tuple(i for i in range(arr.ndim) if i not in merged_axes)
    flattened_size = int(np.prod([original_shape[i] for i in flattened_axes]))

    # Permute merged axes:
    #   - It should be first if 0 in merged_axes (merge sample dim)
    #   - Otherwise it should be last (merge feature shape)
    if 0 in merged_axes:
        perm = merged_axes + flattened_axes
        final_shape = (merged_size, flattened_size)
    else:
        perm = flattened_axes + merged_axes
        final_shape = (flattened_size, merged_size)

    arr_permutated = arr.transpose(perm)
    arr_flat = arr_permutated.reshape(final_shape)

    # Store metadata required to reverse the transform
    meta = {
        "original_shape": original_shape,
        "merged_axes": merged_axes,
    }

    return arr_flat, meta


def unflatten_from_2d(flat: np.ndarray, meta: dict):
    """
    Restore the original N-D array from a 2D matrix flattened by `flatten_to_2d`.

    Description:
        Reconstructs the original shape using metadata describing which axes \
        were merged and how they were ordered during flattening.

    Args:
        flat (np.ndarray): 2D flattened array.
        meta (dict): Metadata from :func:`flatten_to_2d` describing the original axes.

    Returns:
        np.ndarray: Restored N-D array with its original shape.

    """
    original_shape = meta["original_shape"]
    ndim = len(original_shape)

    # Axes that were merged
    merged_axes = meta["merged_axes"]
    merged_orig_shape = tuple(original_shape[i] for i in merged_axes)

    # Axes that were flattened
    flatten_axes = tuple(i for i in range(ndim) if i not in merged_axes)
    flatten_orig_shape = tuple(original_shape[i] for i in flatten_axes)

    # Reshape (undo collapse in proper order)
    if 0 in merged_axes:
        exp_shape = merged_orig_shape + flatten_orig_shape
        perm = merged_axes + flatten_axes
    else:
        exp_shape = flatten_orig_shape + merged_orig_shape
        perm = flatten_axes + merged_axes

    arr_expanded = flat.reshape(exp_shape)
    inv_perm = np.argsort(perm)
    arr_orig = arr_expanded.transpose(inv_perm)

    return arr_orig


def align_ranks(arr1: Any, arr2: Any, backend: Backend | None = None) -> tuple:
    """
    Align the *ranks* (number of dimensions) of two array-like objects.

    This function ensures both arrays have the same number of dimensions by inserting singleton
    axes (using backend-safe operations like `unsqueeze` or `expand_dims`). It does **not**
    permute or reshape non-singleton dimensions, and it preserves autograd in all supported
    backends.

    The operation is symmetric; the array with the smaller rank is modified so that both
    arrays end up with the same shape, as long as they are compatible under singleton
    expansion. If both arrays already have the same rank and shape, they are returned
    unchanged.

    Examples:
        >>> import numpy as np
        >>> a = np.random.random(size=(32, 1, 100))
        >>> b = np.random.random(size=(32, 100))
        >>> a.shape, b.shape
        ((32, 1, 100), (32, 100))
        >>> a2, b2 = align_ranks(a, b)
        >>> a2.shape, b2.shape
        ((32, 1, 100), (32, 1, 100))

    Constraints:
        - Only singleton (size-1) dimensions are added or removed.
        - Arrays with mismatched non-singleton sizes (e.g. (32,1,4) vs (32,1))
          will raise a ValueError.
        - The total number of elements (`prod(shape)`) must match.
        - Does not detach tensors or break computational graphs.

    Args:
        arr1 (Any):
            First array-like object (must expose `.shape`).
        arr2 (Any):
            Second array-like object (must expose `.shape`).
        backend (Backend | None):
            Backend identifier determining which framework (Torch, TensorFlow, NumPy)
            operations are used. If None, backend is inferred for each array.

    Returns:
        tuple[Any, Any]: Pair `(arr1_aligned, arr2_aligned)` with matching shapes.

    Raises:
        ValueError:
            - If arrays have equal rank but different shapes.
            - If shapes cannot be aligned by inserting/removing only singleton dimensions.
        TypeError:
            If an unsupported backend is provided.
        ImportError:
            If the specified backend is unavailable (e.g., PyTorch not installed).
        RuntimeError:
            If alignment fails despite matching total element counts.

    Notes:
        - This function is intended for internal use in loss computation or tensor comparison,
          where user-defined models may add trailing singleton axes (e.g., via Conv1D or Linear).
        - Only the array with the smaller rank is modified; the other is returned unchanged.
        - Autograd gradients are preserved (no `.detach()`, `.numpy()`, or graph breakage).

    """
    from math import prod

    torch = check_torch()
    tf = check_tensorflow()

    def _modify_arr(arr, singleton_idxs: list[int], arr_backend: Backend | None = None):
        """
        Insert singleton axes into `arr` using backend-appropriate operations.

        Args:
            arr (Any): Tensor or array to reshape.
            singleton_idxs (list[int]): Positions where singleton axes should be inserted.
            arr_backend (Backend | None): Backend override. Defaults to inference from `arr`.

        Returns:
            Any: Reshaped tensor compatible with the backend.

        """
        if arr_backend is None:
            arr_backend = infer_backend(arr)

        if arr_backend == Backend.TORCH:
            if torch is None:
                msg = f"PyTorch is not installed but a backend of {Backend.TORCH} was used."
                raise ImportError(msg)
            arr = convert_to_format(arr, fmt=DataFormat.TORCH)
            for idx in singleton_idxs:
                arr = arr.unsqueeze(idx)
            return arr
        if arr_backend == Backend.TENSORFLOW:
            if tf is None:
                msg = f"TensorFlow is not installed but a backend of {Backend.TENSORFLOW} was used."
                raise ImportError(msg)
            arr = convert_to_format(arr, fmt=DataFormat.TENSORFLOW)
            for idx in singleton_idxs:
                arr = tf.expand_dims(arr, axis=idx)
            return arr
        if arr_backend in (Backend.SCIKIT, Backend.NONE):
            arr = convert_to_format(arr, fmt=DataFormat.NUMPY)
            for idx in singleton_idxs:
                arr = np.expand_dims(arr, axis=idx)
            return arr
        msg = f"Unsupported backend: {arr_backend}"
        raise TypeError(msg)

    shape_1, shape_2 = arr1.shape, arr2.shape

    # If already aligned, return or reshape
    if len(shape_1) == len(shape_2):
        if shape_1 != shape_2:
            raise ValueError("Arrays are the same rank but not in the same order.")
        return arr1, arr2

    # Ranks cannot be matched via singleton additions/subtractions
    if prod(shape_1) != prod(shape_2):
        msg = f"Arrays cannot be aligned via addition or removal of singletons: {shape_1} & {shape_2}"
        raise ValueError(msg)

    # Use highest rank as reference (ref), other as comparison (cmp)
    ref, cmp = shape_2, shape_1
    if len(shape_1) > len(shape_2):
        ref, cmp = shape_1, shape_2

    # For each item in ref, see if cmp matches:
    cmp_assignments = -1 * np.ones(
        shape=len(ref),
    )  # each element corresponds to index in ref
    for i in range(len(cmp)):
        for j in range(i, len(ref)):
            if cmp[i] == ref[j]:
                cmp_assignments[j] = j
                break

    # The index of each -1 element is where a singleton needs to be inserted
    idxs = np.argwhere(cmp_assignments == -1).reshape(-1).tolist()
    if len(shape_1) > len(shape_2):  # arr1 is ref, modify arr2
        arr2 = _modify_arr(arr2, singleton_idxs=idxs, arr_backend=backend)
    else:  # arr2 is ref, modify arr1
        arr1 = _modify_arr(arr1, singleton_idxs=idxs, arr_backend=backend)

    if arr1.shape != arr2.shape:
        raise RuntimeError("Failed to match ranks.")

    return arr1, arr2


def stack_nested_numpy(obj: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """
    Recursively stack nested NumPy object arrays into a dense array of shape (n_samples, *shape).

    Args:
        obj (np.ndarray): Object-dtype array containing nested arrays.
        shape (tuple[int, ...]): Expected inner shape per sample, excluding `n_samples`.

    Returns:
        np.ndarray: Dense NumPy array of shape `(n_samples, *shape)`.

    Notes:
        - Assumes homogenous shapes across all samples.
        - Never converts to Python lists.
        - Recurses only for object-dtype subarrays.

    """
    if not isinstance(obj, np.ndarray):
        msg = f"Expected np.ndarray, got {type(obj)}"
        raise TypeError(msg)
    if obj.dtype != object:
        # Already dense numeric array
        return obj.reshape((len(obj), *shape)) if shape else obj

    # Base case: final depth, inner arrays are numeric
    if len(shape) == 1 or all(
        not isinstance(sub, np.ndarray) or sub.dtype != object for sub in obj
    ):
        return np.stack(obj, axis=0)

    # Recursive case: still object arrays inside
    inner_shape = shape[1:]
    stacked_inner = [stack_nested_numpy(sub, inner_shape) for sub in obj]
    return np.stack(stacked_inner, axis=0)


def merge_dict_of_arrays_to_numpy(
    data: dict[str, np.ndarray],
    *,
    mode: Literal["auto", "stack", "concat", "flatten"] = "auto",
    axis: int | None = None,
    align_singletons: bool = True,
) -> np.ndarray:
    """
    Merge a dictionary of NumPy arrays into a single unified NumPy array.

    The behavior depends on the specified `mode`:
        - **"stack"**:   Adds a new axis for the dictionary keys (default axis=0).
                         e.g. [(10, 4), (10, 4)] → (2, 10, 4)
        - **"concat"**:  Concatenates along an existing axis (default axis=0).
                         e.g. [(10, 4), (10, 4)] → (20, 4)
        - **"flatten"**: Flattens and concatenates across features.
                         e.g. [(10, 4), (10, 3)] → (10, 7)
        - **"auto"**:    Attempts stack → concat(axis=0) → concat(axis=-1) → flatten.

    Args:
        data (dict[str, np.ndarray]): Mapping of key to array-like objects to merge.
        mode (Literal["auto", "stack", "concat", "flatten"]): Merge strategy. Defaults to "auto".
        axis (int | None): Axis for concatenation or stacking. Defaults depend on `mode`.
        align_singletons (bool): Add singleton dimensions to align ranks when True.

    Returns:
        np.ndarray: The merged array.

    Raises:
        ValueError: If arrays cannot be merged according to the selected mode.
        RuntimeError: If "auto" mode fails to find a valid merge strategy.

    """

    def _flatten(arrays: list[np.ndarray], axis: int | None = None) -> np.ndarray:
        """
        Flatten all arrays and then concatenate along the feature axis.

        Behavior:
            - If `axis is None`: flatten *everything* into a 1D vector.
              e.g. [(2, 3), (2, 2)] -> (10,)
            - If `axis` is specified: only flatten *after* that axis.
              e.g. [(10, 3, 2), (10, 4, 1)], axis=0 -> (10, 14)

        Args:
            arrays (list[np.ndarray]): Arrays to flatten.
            axis (int | None): Axis preserved before flattening. If None, fully flatten.

        Returns:
            np.ndarray: Concatenated flattened array.

        """
        if not arrays:
            raise ValueError("Cannot flatten an empty list of arrays.")
        if axis is None:
            # Full flatten: produce a single 1D vector
            return np.hstack([a.flatten() for a in arrays])
        # Partial flatten: preserve leading axes up to the specified one
        flattened = [a.reshape(*(a.shape[: axis + 1]), -1) for a in arrays]
        return np.hstack(flattened)

    def _concat(arrays: list[np.ndarray], axis: int) -> np.ndarray:
        """
        Concatenate all arrays along the given axis.

        Raises ValueError if dimensions (other than the target axis) are incompatible.

        Args:
            arrays (list[np.ndarray]): Arrays to concatenate.
            axis (int): Axis along which to concatenate.

        Returns:
            np.ndarray: Concatenated output.

        """
        if not arrays:
            raise ValueError("Cannot concatenate an empty list of arrays.")
        return np.concatenate(arrays, axis=axis)

    def _stack(arrays: list[np.ndarray], axis: int) -> np.ndarray:
        """
        Stack arrays along a *new* dimension (the dict-key axis).

        Raises ValueError if arrays differ in shape.

        Args:
            arrays (list[np.ndarray]): Arrays to stack.
            axis (int): Axis index for the new dimension.

        Returns:
            np.ndarray: Stacked output array.

        """
        if not arrays:
            raise ValueError("Cannot stack an empty list of arrays.")
        return np.stack(arrays, axis=axis)

    # 1. Normalize all arrays and shapes
    np_dict = {k: to_numpy(v) for k, v in data.items()}
    arrs = list(np_dict.values())
    if not arrs:
        raise ValueError("No arrays provided for merging.")

    # 2. Align ranks, if needed & specified
    arr_shapes = [a.shape for a in arrs]
    ranks = [len(s) for s in arr_shapes]
    ranks_match = len(set(ranks)) == 1
    # Align rank to reference
    if (
        align_singletons
        and not (mode == "flatten" and axis is None)
        and not ranks_match
    ):
        try:
            aligned_arrs = []
            # Use highest rank array as reference shape to match
            ref_idx = np.argmax(ranks)
            for i in range(len(arrs)):
                if i == ref_idx:
                    aligned_arrs.append(arrs[i])
                else:
                    # align_ranks(ref, cmp) -> returns (ref_aligned, cmp_aligned)
                    _, arr_cmp = align_ranks(arrs[ref_idx], arrs[i])
                    aligned_arrs.append(arr_cmp)
            # Reset arrs and shapes
            arrs = aligned_arrs
            arr_shapes = [a.shape for a in arrs]
        except ValueError as e:
            msg = f"Failed to align ranks: {e}. Skipping."
            warn(msg, category=UserWarning, stacklevel=2)

    # 3. Determine how to combine
    if mode == "auto":
        # Order of calls: stack (axis=0) -> concat (axis=0) -> concat (axis=-1) -> flatten
        errors = []
        for fn, ax in [
            (_stack, axis or 0),
            (_concat, axis or 0),
            (_concat, axis or -1),
            (_flatten, axis),
        ]:
            try:
                return fn(arrs) if ax is None else fn(arrs, ax)
            except ValueError as e:  # noqa: PERF203
                # Store error context for debugging
                errors.append(f"{fn.__name__} failed (axis={ax}): {e}")
        raise RuntimeError("Failed to merge arrays:\n" + "\n".join(errors))

    if mode == "flatten":
        return _flatten(arrays=arrs, axis=axis)
    if mode == "concat":
        return _concat(arrays=arrs, axis=axis or 0)
    if mode == "stack":
        return _stack(arrays=arrs, axis=axis or 0)
    msg = f"Unsupported value of 'mode': {mode}"
    raise ValueError(msg)


def convert_dict_to_format(
    data: dict[str, Any],
    *,
    fmt: str | DataFormat,
    errors: ErrorMode = ErrorMode.RAISE,
    mode: Literal["auto", "stack", "concat", "flatten"] = "auto",
    axis: int | None = None,
    align_singletons: bool = True,
) -> Any:
    """
    Convert a dictionary of arrays, lists, or tensors into a unified data format.

    This function provides a high-level conversion layer for mapping
    a dictionary of heterogeneous data (NumPy arrays, tensors, or lists)
    into one of the supported target formats in `DataFormat`.
    Depending on the selected `fmt`, the function may either:
        - Preserve the dictionary structure (DICT_* formats), or
        - Merge the data into a single unified array or tensor (NUMPY, TORCH, TENSORFLOW).

    When merging into array/tensor formats, shape compatibility is automatically
    validated and can optionally be aligned via singleton expansion.
    The merge behavior is controlled by `mode`, which determines how individual
    arrays are combined (stacked, concatenated, or flattened).

    ---
    Supported Format Behaviors

    **Dictionary-like outputs**
      - `DICT`: Convert all entries to native Python lists/scalars.
      - `DICT_LIST`: Each key maps to a Python list.
      - `DICT_NUMPY`: Each key maps to a NumPy array.
      - `DICT_TORCH`: Each key maps to a Torch tensor.
      - `DICT_TENSORFLOW`: Each key maps to a TensorFlow tensor.

    **Tabular output**
      - `PANDAS`: Returns a pandas DataFrame with each key as a column.

    **Unified array/tensor outputs**
      - `NUMPY`, `TORCH`, or `TENSORFLOW`:
        Merge all dictionary values into a single multi-dimensional array/tensor
        according to `mode` and `axis`. Example:
          - `mode="stack"` -> adds new axis for dict keys -> shape (n_keys, ...)
          - `mode="concat"` -> concatenates along existing axis -> shape (..., sum(features))
          - `mode="flatten"` -> flattens and concatenates all features into one dimension.
        If array shapes differ in rank, singleton dimensions can be aligned
        when `align_singletons=True`.

    **List output**
      - `LIST`: Converts to a list-of-lists, zipping all entries by sample index.

    Args:
        data (dict[str, Any]): Dictionary mapping feature names (str) to array-like \
            objects (NumPy arrays, lists, Torch tensors, etc.).
        fmt (DataFormat | str): Target data format identifier. Must be one of the \
            supported `DataFormat` enums or equivalent string values (e.g., "numpy", \
            "torch", "pandas", etc.).
        errors (ErrorMode): Error handling behavior during element-wise conversion:
            - `ErrorMode.RAISE`: Raise on incompatible or unconvertible types.
            - `ErrorMode.COERCE`: Attempt forced conversion where possible.
            - `ErrorMode.IGNORE`: Skip conversion errors and preserve original objects.
        mode (Literal["auto", "stack", "concat", "flatten"]): Merge strategy for \
            array/tensor outputs (ignored for dict-like formats):
            - `"stack"`: Add a new axis representing dict keys.
            - `"concat"`: Concatenate arrays along an existing axis.
            - `"flatten"`: Flatten and concatenate across all features.
            - `"auto"`: Attempt stack -> concat(axis=0) -> concat(axis=-1) -> flatten.
        axis (int | None): Axis index for concatenation or stacking. Defaults to 0 for \
            most operations, or -1 for feature concatenation.
        align_singletons (bool): If True, automatically expands singleton dimensions to \
            align shapes before merging arrays of different ranks.

    Returns:
        Any: The converted object in the requested format. Possible return types include:
            - `dict` (for DICT_* formats)
            - `pandas.DataFrame` (for PANDAS)
            - `numpy.ndarray` (for NUMPY)
            - `torch.Tensor` (for TORCH)
            - `tensorflow.Tensor` (for TENSORFLOW)
            - `list[list]` (for LIST)

    Raises:
        ImportError:
            If the selected target format requires a backend (Torch/TensorFlow)
            that is not installed.
        ValueError:
            If `fmt` is unsupported or arrays cannot be merged under the given mode.
        TypeError:
            If any input cannot be coerced into the target type when `errors="raise"`.

    Examples:
        >>> data = {"a": np.ones((10, 2)), "b": np.zeros((10, 3))}
        >>> convert_dict_to_format(data, fmt="numpy", mode="concat").shape
        (10, 5)

        >>> data = {"a": np.ones((10, 3)), "b": np.zeros((10, 3))}
        >>> convert_dict_to_format(data, fmt="torch", mode="stack", axis=1).shape
        (10, 2, 3)

    """
    torch = check_torch()
    tf = check_tensorflow()

    fmt = normalize_format(fmt)

    # Dictionary-like formats
    if fmt == DataFormat.DICT:
        return to_python(data)
    if fmt == DataFormat.DICT_LIST:
        return {k: to_list(v, errors=errors) for k, v in data.items()}
    if fmt == DataFormat.DICT_NUMPY:
        return {k: to_numpy(v, errors=errors) for k, v in data.items()}
    if fmt == DataFormat.DICT_TORCH:
        return {k: to_torch(v, errors=errors) for k, v in data.items()}
    if fmt == DataFormat.DICT_TENSORFLOW:
        return {k: to_tensorflow(v, errors=errors) for k, v in data.items()}

    # DataFrame
    if fmt == DataFormat.PANDAS:
        return pd.DataFrame({k: to_list(v, errors=errors) for k, v in data.items()})

    # Unified array formats (NUMPY, TORCH, TENSORFLOW)
    if format_requires_compatible_shapes(fmt):
        np_merged = merge_dict_of_arrays_to_numpy(
            data=data,
            mode=mode,
            axis=axis,
            align_singletons=align_singletons,
        )

        if fmt == DataFormat.NUMPY:
            return np_merged
        if fmt == DataFormat.TORCH:
            if torch is None:
                raise ImportError("PyTorch is not installed.")
            return torch.tensor(np_merged, dtype=torch.float32)
        if fmt == DataFormat.TENSORFLOW:
            if tf is None:
                raise ImportError("TensorFlow is not installed.")
            return tf.convert_to_tensor(np_merged, dtype=tf.float32)

    # List
    if fmt == DataFormat.LIST:
        return [
            list(row)
            for row in zip(
                *[to_list(v, errors=errors) for v in data.values()],
                strict=True,
            )
        ]

    msg = f"Unsupported data format: {fmt}"
    raise ValueError(msg)


def to_python(obj):
    """
    Recursively converts an object into its native Python equivalent.

    Supported conversions:
    - NumPy scalars    -> Python scalars
    - NumPy arrays     -> Python lists
    - PyTorch tensors  -> Python scalars or lists
    - TensorFlow tensors -> Python scalars or lists
    - Dicts, tuples, and lists -> Recursively converted

    Args:
        obj (Any): Object to convert.

    Returns:
        Any: Python-native representation.

    """
    torch = check_torch()
    tf = check_tensorflow()

    # NumPy
    if isinstance(obj, np.generic):  # np.int64, np.float64, etc.
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    # Pandas
    if isinstance(obj, pd.Series):
        return to_python(obj.values)

    # PyTorch
    if torch is not None and isinstance(obj, torch.Tensor):
        # Move to CPU, detach from graph if needed, convert to list or scalar
        if obj.ndim == 0:
            return obj.item()
        return obj.detach().cpu().tolist()

    # TensorFlow
    if tf is not None and isinstance(obj, tf.Tensor):
        # Use .numpy() safely, then convert like numpy arrays
        np_obj = obj.numpy()
        if np_obj.ndim == 0:
            return np_obj.item()
        return np_obj.tolist()

    # Containers
    if isinstance(obj, dict):
        return {k: to_python(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return type(obj)(to_python(v) for v in obj)

    # Base case
    return obj


def to_list(obj: Any, errors: ErrorMode = ErrorMode.RAISE):
    """
    Converts any object into a Python list.

    Args:
        obj (Any): Object to convert.
        errors (ErrorMode): How to handle non-listable objects.

    Returns:
        list[Any] | Any: Converted list or original object when ignored.

    """
    # If we're ignoring incompatible types, leave dicts unchanged directly
    if errors == ErrorMode.IGNORE and isinstance(obj, dict):
        return obj

    py_obj = to_python(obj)

    # If it's already a list or tuple, convert directly
    if isinstance(py_obj, list | tuple | np.ndarray):
        return list(py_obj)

    # If it's a scalar, decide based on `errors`
    if np.isscalar(py_obj):
        return [py_obj]

    # Dicts aren't naturally convertible to lists
    if isinstance(py_obj, dict):
        if errors == ErrorMode.RAISE:
            raise TypeError("Cannot convert dict to list. Use DICT format instead.")
        if errors == ErrorMode.COERCE:
            # Convert dict values into a list of values
            return list(py_obj.values())
        if errors == ErrorMode.IGNORE:
            return py_obj

    # Fallback: try NumPy coercion if possible
    try:
        return np.asarray(py_obj).tolist()
    except Exception as e:
        if errors == ErrorMode.RAISE:
            msg = f"Cannot convert object of type {type(py_obj)} to list."
            raise TypeError(msg) from e
        if errors == ErrorMode.IGNORE:
            return py_obj
        if errors == ErrorMode.COERCE:
            return [py_obj]


def to_numpy(
    obj: Any,
    errors: ErrorMode = ErrorMode.RAISE,
    *,
    _top_level: bool = True,
) -> np.ndarray:
    """
    Recursively converts any object into a NumPy array.

    Nested lists/tuples are traversed so every sub-sequence
    is converted to np.ndarray where appropriate. Scalars are left
    as plain Python types inside the structure; only the *outermost*
    call wraps the final scalar into a 0-D array.

    Args:
        obj (Any): Object to convert.
        errors (ErrorMode): Error handling mode (RAISE, COERCE, IGNORE).
        _top_level (bool): Internal flag to track recursion depth.

    Returns:
        np.ndarray | Any: NumPy array or original object when ignored.

    """
    # If it's already a numpy array, just return
    if isinstance(obj, np.ndarray):
        return obj

    py_obj = to_python(obj)

    # Dicts must use DICT_NUMPY format unless coerced
    if isinstance(py_obj, dict):
        if errors == ErrorMode.RAISE:
            raise TypeError(
                "Cannot convert dict directly to NumPy array. Use DICT_NUMPY instead.",
            )
        if errors == ErrorMode.COERCE:
            return np.array(list(py_obj.values()))
        if errors == ErrorMode.IGNORE:
            return py_obj

    # Sequences (lists, tuples) -> convert directly
    if isinstance(py_obj, list | tuple):
        try:
            # Recursively convert every element to numpy before stacking
            converted = [
                to_numpy(item, errors=errors, _top_level=False) for item in py_obj
            ]
            return np.array(
                converted,
                dtype=object
                if any(isinstance(c, np.ndarray) and c.ndim == 0 for c in converted)
                else None,
            )
        except Exception as e:
            if errors == ErrorMode.RAISE:
                msg = f"Cannot convert nested sequence of type {type(py_obj)} to NumPy array."
                raise TypeError(msg) from e
            if errors == ErrorMode.IGNORE:
                return py_obj
            if errors == ErrorMode.COERCE:
                return np.array([py_obj])

    # Scalars -> wrap into a 0-D array
    if np.isscalar(py_obj):
        if _top_level:
            return np.asarray(py_obj)
        return py_obj  # leave scalars within a nest object unchanged

    # Unsupported type
    if errors == ErrorMode.RAISE:
        msg = f"Cannot convert object of type {type(py_obj)} to NumPy array."
        raise TypeError(msg)
    if errors == ErrorMode.IGNORE:
        return py_obj
    if errors == ErrorMode.COERCE:
        return np.array([py_obj])
    return None


def to_torch(obj: Any, errors: ErrorMode = ErrorMode.RAISE):
    """
    Convert any object into a PyTorch tensor.

    Args:
        obj (Any): Object to convert.
        errors (ErrorMode): Error handling strategy.

    Returns:
        torch.Tensor | Any: Torch tensor or original object when ignored.

    """
    torch = ensure_torch()

    # If it's already a Torch Tensor, just return
    if isinstance(obj, torch.Tensor):
        return obj

    py_obj = to_python(obj)
    try:
        return torch.as_tensor(np.asarray(py_obj), dtype=torch.float32)
    except Exception as e:
        if errors == ErrorMode.RAISE:
            msg = f"Cannot convert object of type {type(py_obj)} to Torch tensor."
            raise TypeError(msg) from e
        if errors == ErrorMode.IGNORE:
            return py_obj
        if errors == ErrorMode.COERCE:
            return torch.as_tensor(np.asarray([py_obj]), dtype=torch.float32)


def to_tensorflow(obj: Any, errors: ErrorMode = ErrorMode.RAISE):
    """
    Convert any object into a TensorFlow tensor.

    Args:
        obj (Any): Object to convert.
        errors (ErrorMode): Error handling strategy.

    Returns:
        tensorflow.Tensor | Any: TensorFlow tensor or original object when ignored.

    """
    tf = ensure_tensorflow()

    # If it's already a Tensforflow Tensor, just return
    if isinstance(obj, tf.Tensor):
        return obj

    py_obj = to_python(obj)
    try:
        return tf.convert_to_tensor(np.asarray(py_obj), dtype=tf.float32)
    except Exception as e:
        if errors == ErrorMode.RAISE:
            msg = f"Cannot convert object of type {type(py_obj)} to TensorFlow tensor."
            raise TypeError(msg) from e
        if errors == ErrorMode.IGNORE:
            return py_obj
        if errors == ErrorMode.COERCE:
            return tf.convert_to_tensor(np.asarray([py_obj]), dtype=tf.float32)


def convert_to_format(
    data: Any,
    fmt: str | DataFormat,
    errors: ErrorMode = ErrorMode.RAISE,
) -> Any:
    """
    Convert a data object into the specified format.

    Args:
        data (Any): Dicts, arrays, lists, scalars, or tensors.
        fmt (str | DataFormat): Target data format to convert into.
        errors (ErrorMode): Strategy for handling incompatible types.

    Returns:
        Any: Converted object.

    """
    torch = check_torch()
    tf = check_tensorflow()

    fmt = normalize_format(fmt)
    if isinstance(data, dict):
        raise TypeError(
            "Use the `convert_dict_to_format` method for conversion of dict-like data.",
        )

    if fmt == DataFormat.NUMPY:
        return to_numpy(data, errors=errors)

    if fmt == DataFormat.LIST:
        return to_list(data, errors=errors)

    if fmt == DataFormat.TORCH:
        if torch is None:
            raise ImportError("PyTorch is not installed.")
        return to_torch(data, errors=errors)

    if fmt == DataFormat.TENSORFLOW:
        if tf is None:
            raise ImportError("TensorFlow is not installed.")
        return to_tensorflow(data, errors=errors)

    msg = f"Unsupported data format: {fmt}"
    raise ValueError(msg)
