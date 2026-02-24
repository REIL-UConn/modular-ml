"""Comparison helpers for array, tensor, and nested Python structures."""

import numpy as np

from modularml.utils.environment.optional_imports import check_tensorflow, check_torch
from modularml.utils.nn.backend import Backend

torch = check_torch()
tf = check_tensorflow()


def arrays_equal(a, b, *, rtol=1e-6, atol=1e-8):
    """
    Return True if two NumPy arrays share the same shape, dtype, and values.

    Args:
        a (np.ndarray): First array to compare.
        b (np.ndarray): Second array to compare.
        rtol (float, optional): Relative tolerance for :func:`numpy.allclose`. Defaults to 1e-6.
        atol (float, optional): Absolute tolerance for :func:`numpy.allclose`. Defaults to 1e-8.

    Returns:
        bool: True if arrays match under the specified tolerances.

    """
    if a.shape != b.shape:
        return False
    if a.dtype != b.dtype:
        return False
    return np.allclose(a, b, rtol=rtol, atol=atol)


def deep_equal(a, b):
    """
    Recursively check equality of nested structures, tensors, and primitives.

    Args:
        a (Any): First structure to compare.
        b (Any): Second structure to compare.

    Returns:
        bool: True if all nested elements compare equal.

    """
    if a is b:
        return True

    # Arrays & tensors
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        return arrays_equal(a, b, rtol=1e-6, atol=1e-8)
    if (
        torch is not None
        and isinstance(a, torch.Tensor)
        and isinstance(b, torch.Tensor)
    ):
        return arrays_equal(
            a.detach().cpu().numpy(),
            b.detach().cpu().numpy(),
            rtol=1e-6,
            atol=1e-8,
        )
    if tf is not None and isinstance(a, tf.Tensor) and isinstance(b, tf.Tensor):
        return arrays_equal(a.numpy(), b.numpy(), rtol=1e-6, atol=1e-8)

    # Primitive types
    if isinstance(a, (int, float, str, bool)) or a is None:
        return a == b

    # Dictionary
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a.keys()) != set(b.keys()):
            return False
        return all(deep_equal(a[k], b[k]) for k in a)

    # List or tuple
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return False
        return all(deep_equal(x, y) for x, y in zip(a, b, strict=True))

    # Classes
    if isinstance(a, type) and isinstance(b, type):
        return (
            (a.__qualname__ == b.__qualname__)
            and (
                getattr(
                    a,
                    "__module___",
                    None,
                )
                == getattr(
                    b,
                    "__module___",
                    None,
                )
            )
            and (
                getattr(
                    a,
                    "__file__",
                    None,
                )
                == getattr(
                    b,
                    "__file__",
                    None,
                )
            )
        )

    # Fallback
    return a == b


def tensors_are_equal(a, b, *, tol: float = 1e-9, strict_backend_match: bool = True):
    """
    Compare two tensor-like elements for equality across supported backends.

    Args:
        a (Any): First tensor-like object to compare.
        b (Any): Second tensor-like object to compare.
        tol (float, optional): Absolute tolerance for :func:`numpy.allclose`. Defaults to 1e-9.
        strict_backend_match (bool, optional): Require matching backends before comparing values.

    Returns:
        bool: True if tensors match backend (when requested), shape, and numeric values.

    """

    # Detect backends
    def detect_backend(x):
        try:
            import numpy as np

            if isinstance(x, np.ndarray):
                return Backend.SCIKIT
        except Exception:  # noqa: BLE001, S110
            pass

        try:
            import torch

            if isinstance(x, torch.Tensor):
                return Backend.TORCH
        except Exception:  # noqa: BLE001, S110
            pass

        try:
            import tensorflow as tf

            if isinstance(x, (tf.Tensor, tf.Variable)):
                return Backend.TENSORFLOW
        except Exception:  # noqa: BLE001, S110
            pass

        # Allow Python scalars, lists, tuples -> treat as numpy
        if isinstance(x, (int, float, complex, list, tuple)):
            return Backend.SCIKIT

        return type(x).__name__

    backend_a = detect_backend(a)
    backend_b = detect_backend(b)

    # 2. Strict backend check
    if strict_backend_match and backend_a != backend_b:
        return False

    # 3. Convert both to NumPy for value comparison
    from modularml.utils.data.conversion import convert_to_format
    from modularml.utils.data.data_format import DataFormat

    a_np = convert_to_format(a, fmt=DataFormat.NUMPY)
    b_np = convert_to_format(b, fmt=DataFormat.NUMPY)

    # 4. Shape check
    if a_np.shape != b_np.shape:
        return False

    # 5. Numeric equality check
    # Handle NaN cases explicitly: NaN should not equal NaN
    if np.any(np.isnan(a_np)) or np.any(np.isnan(b_np)):
        return False

    return np.allclose(a_np, b_np, atol=tol, rtol=0.0)
