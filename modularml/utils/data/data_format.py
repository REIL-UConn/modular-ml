"""Helpers for normalizing ModularML data format identifiers."""

from enum import Enum

import numpy as np

from modularml.utils.environment.optional_imports import check_tensorflow, check_torch
from modularml.utils.nn.backend import Backend


class DataFormat(Enum):
    PANDAS = "pandas"
    NUMPY = "numpy"
    DICT = "dict"
    DICT_NUMPY = "dict_numpy"
    DICT_LIST = "dict_list"
    DICT_TORCH = "dict_torch"
    DICT_TENSORFLOW = "dict_tensorflow"
    LIST = "list"
    TORCH = "torch.tensor"
    TENSORFLOW = "tensorflow.tensor"


_FORMAT_ALIASES = {
    "pandas": DataFormat.PANDAS,
    "pd": DataFormat.PANDAS,
    "df": DataFormat.PANDAS,
    "numpy": DataFormat.NUMPY,
    "np": DataFormat.NUMPY,
    "dict": DataFormat.DICT,
    "dict_numpy": DataFormat.DICT_NUMPY,
    "dict_list": DataFormat.DICT_LIST,
    "dict_torch": DataFormat.DICT_TORCH,
    "dict_tensorflow": DataFormat.DICT_TENSORFLOW,
    "list": DataFormat.LIST,
    "torch": DataFormat.TORCH,
    "torch.tensor": DataFormat.TORCH,
    "tf": DataFormat.TENSORFLOW,
    "tensorflow": DataFormat.TENSORFLOW,
    "tensorflow.tensor": DataFormat.TENSORFLOW,
}

_TENSORLIKE_FORMATS = (
    DataFormat.NUMPY,
    DataFormat.TORCH,
    DataFormat.TENSORFLOW,
)


def normalize_format(fmt: str | DataFormat) -> DataFormat:
    """
    Normalize a string or :class:`DataFormat` value into a :class:`DataFormat`.

    Args:
        fmt (str | DataFormat): Format identifier or enum instance.

    Returns:
        DataFormat: Canonical enum value.

    Raises:
        ValueError: If `fmt` cannot be resolved to a known format.

    """
    if isinstance(fmt, DataFormat):
        return fmt
    fmt = fmt.lower()
    if fmt not in _FORMAT_ALIASES:
        msg = f"Unknown data format: {fmt}"
        raise ValueError(msg)
    return _FORMAT_ALIASES[fmt]


def format_requires_compatible_shapes(fmt: DataFormat) -> bool:
    """
    Return True if the specified :class:`DataFormat` requires compatible shapes.

    Args:
        fmt (DataFormat): Format to inspect.

    Returns:
        bool: True if tensors must share shapes (NumPy, Torch, TensorFlow).

    """
    return fmt in [DataFormat.NUMPY, DataFormat.TORCH, DataFormat.TENSORFLOW]


def format_is_tensorlike(fmt: DataFormat) -> bool:
    """
    Return True if the specified :class:`DataFormat` yields tensor-like results.

    Args:
        fmt (DataFormat): Format to inspect.

    Returns:
        bool: True when `fmt` is a tensor format.

    """
    fmt = normalize_format(fmt)
    return fmt in _TENSORLIKE_FORMATS


def infer_data_type(obj) -> str:
    """
    Infer a coarse data type string for the provided object.

    Args:
        obj (Any): Object whose underlying data type should be determined.

    Returns:
        str: One of "float", "int", "bool", "string", or "unknown".

    """
    torch = check_torch()
    tf = check_tensorflow()

    # Torch tensor
    if torch is not None and isinstance(obj, torch.Tensor):
        dt = obj.dtype
        if dt.is_floating_point:
            return "float"
        if dt in (torch.int8, torch.int16, torch.int32, torch.int64):
            return "int"
        if dt == torch.bool:
            return "bool"
        return "unknown"

    # TensorFlow tensor
    if tf is not None and isinstance(obj, tf.Tensor):
        dt = obj.dtype
        if dt.is_floating:
            return "float"
        if dt.is_integer:
            return "int"
        if dt.is_bool:
            return "bool"
        if dt == tf.string:
            return "string"
        return "unknown"

    # NumPy
    if isinstance(obj, np.ndarray) or np.isscalar(obj):
        if np.issubdtype(np.asarray(obj).dtype, np.floating):
            return "float"
        if np.issubdtype(np.asarray(obj).dtype, np.integer):
            return "int"
        if np.issubdtype(np.asarray(obj).dtype, np.bool_):
            return "bool"
        if np.issubdtype(np.asarray(obj).dtype, np.str_) or np.issubdtype(
            np.asarray(obj).dtype,
            np.object_,
        ):
            return "string"
        return "unknown"

    # Python primitives
    if isinstance(obj, bool):
        return "bool"
    if isinstance(obj, int):
        return "int"
    if isinstance(obj, float):
        return "float"
    if isinstance(obj, str):
        return "string"

    return "unknown"


def get_data_format_for_backend(backend: str | Backend) -> DataFormat:
    """
    Map a backend identifier to its default :class:`DataFormat`.

    Args:
        backend (str | Backend): Backend name or enum instance.

    Returns:
        DataFormat: Default data format for the backend.

    Raises:
        ValueError: If the backend is unsupported.

    """
    if isinstance(backend, str):
        backend = Backend(backend)

    if backend == Backend.TORCH:
        return DataFormat.TORCH
    if backend == Backend.TENSORFLOW:
        return DataFormat.TENSORFLOW
    if backend in [Backend.SCIKIT, Backend.NUMPY, Backend.NONE]:
        return DataFormat.NUMPY
    msg = f"Unsupported backend: {backend}"
    raise ValueError(msg)
