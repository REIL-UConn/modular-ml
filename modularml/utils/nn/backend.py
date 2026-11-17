"""Backend normalization helpers and inference logic."""

from enum import Enum
from typing import Any

from modularml.utils.environment.optional_imports import (
    check_jax,
    check_tensorflow,
    check_torch,
)


class Backend(str, Enum):
    """
    Enum representing supported model backends in ModularML.

    Attributes:
        TORCH (str): PyTorch backend (torch.nn.Module, torch.Tensor).
        TENSORFLOW (str): TensorFlow backend (tf.keras.Model, tf.Tensor).
        JAX (str): JAX arrays, functional models, and Optax optimizers.
        SCIKIT (str): scikit-learn backend (BaseEstimator).
        NONE (str): No backend detected or unsupported object type.

    """

    JAX = "jax"
    TORCH = "torch"
    TENSORFLOW = "tensorflow"
    NUMPY = "numpy"
    SCIKIT = "scikit"
    NONE = "none"

    def __repr__(self) -> str:
        return self.value


def is_valid_backend(value: str | Backend):
    """
    Return True if `value` can be normalized to a :class:`Backend`.

    Args:
        value (str | Backend): Backend identifier to validate.

    Returns:
        bool: True if normalization succeeds.

    """
    try:
        _ = normalize_backend(value=value)
    except ValueError:
        return False
    return True


def normalize_backend(value: str | Backend):
    """
    Normalize a backend string or enum value to :class:`Backend`.

    Args:
        value (str | Backend): Backend identifier to normalize.

    Returns:
        Backend: Normalized backend enum.

    Raises:
        ValueError: If the value cannot be normalized.

    """
    if isinstance(value, Backend):
        return value
    if isinstance(value, str):
        value = value.lower().strip("backend").strip(".")
        if value in ["torch", "pytorch"]:
            return Backend.TORCH
        if value in ["tensorflow", "tf", "keras"]:
            return Backend.TENSORFLOW
        if value in ["scikit", "sklearn"]:
            return Backend.SCIKIT
        if value in ["numpy", "np"]:
            return Backend.SCIKIT
        return Backend(value)

    msg = (
        f"Unsupported Backend value: {value}. "
        f"Expected one of: ['torch', 'tensorflow', 'scikit', 'numpy', 'jax']."
    )
    raise ValueError(msg)


def backend_requires_optimizer(backend: Backend) -> bool:
    """
    Determines whether a given backend requires an optimizer for training.

    Args:
        backend (Backend): The backend to check.

    Returns:
        bool: True if the backend requires an optimizer (e.g., PyTorch, TensorFlow),
              False otherwise (e.g., scikit-learn).

    """
    return backend in [Backend.TORCH, Backend.TENSORFLOW, Backend.JAX]


def infer_backend(obj_or_cls: Any) -> Backend:
    """
    Infers the backend associated with a given object or class.

    Supports inference from:
    - Data objects (e.g., torch.Tensor, tf.Tensor)
    - Model instances and classes
    - Optimizer instances and classes

    Args:
        obj_or_cls (Any): The object/class to inspect.

    Returns:
        Backend: The inferred backend enum value.

    """

    def _safe_issubclass(obj_or_cls, bases):
        try:
            return isinstance(obj_or_cls, type) and issubclass(obj_or_cls, bases)
        except TypeError:
            return False

    # ================================================
    # PyTorch
    # ================================================
    torch = check_torch()
    if torch is not None and (
        isinstance(
            obj_or_cls,
            (torch.Tensor, torch.nn.Module, torch.optim.Optimizer),
        )
        or (
            isinstance(obj_or_cls, type)
            and _safe_issubclass(obj_or_cls, (torch.nn.Module, torch.optim.Optimizer))
        )
    ):
        return Backend.TORCH

    # ================================================
    # Tensorflow
    # ================================================
    tf = check_tensorflow()
    if tf is not None and (
        isinstance(
            obj_or_cls,
            (
                tf.Tensor,
                tf.keras.Model,
                tf.keras.losses.Loss,
                tf.keras.optimizers.Optimizer,
            ),
        )
        or (
            isinstance(obj_or_cls, type)
            and _safe_issubclass(
                obj_or_cls,
                (tf.keras.Model, tf.keras.losses.Loss, tf.keras.optimizers.Optimizer),
            )
        )
    ):
        return Backend.TENSORFLOW

    # ================================================
    # JAX arrays, tracers, Flax modules, and Optax functions.
    jax = check_jax()
    if jax is not None and isinstance(obj_or_cls, (jax.Array, jax.core.Tracer)):
        return Backend.JAX
    cls = obj_or_cls if isinstance(obj_or_cls, type) else type(obj_or_cls)
    if any(base.__module__.startswith("flax.linen.") for base in cls.__mro__):
        return Backend.JAX
    module = getattr(obj_or_cls, "__module__", type(obj_or_cls).__module__)
    if module.startswith(("jax.", "jaxlib.", "flax.", "optax.")):
        return Backend.JAX

    # NumPy
    # ================================================
    try:
        import numpy as np

        if isinstance(obj_or_cls, np.ndarray) or (
            isinstance(obj_or_cls, type) and _safe_issubclass(obj_or_cls, np.ndarray)
        ):
            return Backend.NUMPY
    except ImportError:
        pass

    # ================================================
    # Sklearn
    # ================================================
    try:
        from sklearn.base import BaseEstimator

        if isinstance(obj_or_cls, (BaseEstimator, np.ndarray)) or (
            isinstance(obj_or_cls, type) and _safe_issubclass(obj_or_cls, BaseEstimator)
        ):
            return Backend.SCIKIT
    except ImportError:
        pass

    return Backend.NONE
