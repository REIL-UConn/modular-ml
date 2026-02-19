"""Activation resolution helpers for backend-specific layers."""

from typing import Any

from modularml.utils.environment.optional_imports import ensure_tensorflow, ensure_torch
from modularml.utils.errors.exceptions import ActivationError, BackendNotSupportedError
from modularml.utils.nn.backend import Backend


def resolve_activation(
    activation: str | Any,
    backend: Backend | str | None = None,
):
    """
    Resolves an activation specification into a backend-specific activation layer or function.

    This function accepts either:
      - A string activation name (e.g., "relu", "tanh") and a backend specifier.
      - A callable or instantiated activation class (e.g., `torch.nn.ReLU`, `torch.nn.ReLU()`,
        `tf.keras.layers.ReLU`, `tf.keras.layers.ReLU()`), which will be returned directly.

    Args:
        activation (str | Any):
            Activation to resolve. Can be a name string, a class, or an instance.
        backend (Backend | str | None):
            Backend identifier for string activations.
            Required if `activation` is a string. Ignored otherwise.
            Supported values:
              - "torch" or `Backend.TORCH`
              - "tensorflow" or `Backend.TENSORFLOW`

    Returns:
        torch.nn.Module | tf.keras.layers.Layer | callable:
            Instantiated activation layer or the input activation object itself
            if already valid.

    Raises:
        BackendNotSupportedError:
            If the backend is unsupported for a given string activation.
        ActivationError:
            If the activation name is invalid for the selected backend.
        ValueError:
            If `activation` is a string but `backend` is not provided.

    Examples:
    ```python
        >>> resolve_activation("relu", backend="torch")
        ReLU()

        >>> resolve_activation(torch.nn.Tanh())
        Tanh()

        >>> resolve_activation(tf.keras.layers.ReLU)
        <class 'keras.layers.activation.relu.ReLU'>
    ```

    Notes:
        - Passing instantiated objects (`torch.nn.ReLU()`, `tf.keras.layers.ReLU()`)
          will return them unchanged.
        - Passing uninstantiated layer classes will return the class itself
          (no instantiation).
        - For name strings, a new layer instance is created based on the backend.

    """
    # Case 1 — already a class or instance (Torch or TF layer)
    if not isinstance(activation, str):
        # Allow torch.nn.Module, tf.keras.layers.Layer, or callable classes
        if callable(activation):
            return activation
        msg = f"Unsupported activation type: {type(activation)}"
        raise ActivationError(msg)

    # Case 2 — string name
    if backend is None:
        raise ValueError(
            "Backend must be provided when activation is specified by name (string).",
        )

    return resolve_activation_by_name(activation, backend)


def resolve_activation_by_name(name: str, backend: Backend | str):
    """
    Resolve an activation function name into a backend-specific activation layer instance.

    This utility dynamically constructs and returns the appropriate activation layer
    from PyTorch or TensorFlow, depending on the specified backend. It supports a common
    set of standard activations across both frameworks.

    Supported activation names:
        - "relu"
        - "leaky_relu"
        - "sigmoid"
        - "tanh"
        - "gelu"
        - "elu"

    Args:
        name (str): Name of the activation function (case-insensitive).
        backend (Backend | str): Backend identifier, either as a `Backend` \
            enum or a string. Supported values:
            - "torch" or `Backend.TORCH`
            - "tensorflow" or `Backend.TENSORFLOW`

    Returns:
        torch.nn.Module | tf.keras.layers.Layer: Instantiated activation layer \
            corresponding to the given name and backend.

    Raises:
        BackendNotSupportedError: If the provided backend is not supported.
        ActivationError: If the activation name is not recognized for the \
            selected backend.

    Examples:
    ```python
        >>> resolve_activation_by_name("relu", "torch")
        ReLU()

        >>> resolve_activation_by_name("tanh", "tensorflow")
        <keras.src.layers.activation.tanh.Tanh object at ...>
    ```

    Notes:
        - Each call returns a **new layer instance**, not a shared singleton.
        - Backend detection is case-insensitive.
        - For TensorFlow, activations are returned as `tf.keras.layers.Layer`
          objects; for PyTorch, as `torch.nn.Module` layers.

    """
    name = name.lower()
    if isinstance(backend, str):
        backend = Backend(backend)

    avail_acts = {}
    if backend == Backend.TORCH:
        torch = ensure_torch()
        avail_acts = {
            "relu": torch.nn.ReLU(),
            "leaky_relu": torch.nn.LeakyReLU(),
            "sigmoid": torch.nn.Sigmoid(),
            "tanh": torch.nn.Tanh(),
            "gelu": torch.nn.GELU(),
            "elu": torch.nn.ELU(),
        }
    elif backend == Backend.TENSORFLOW:
        tf = ensure_tensorflow()
        avail_acts = {
            "relu": tf.keras.layers.ReLU(),
            "leaky_relu": tf.keras.layers.LeakyReLU(),
            "sigmoid": tf.keras.layers.Activation("sigmoid"),
            "tanh": tf.keras.layers.Activation("tanh"),
            "gelu": tf.keras.layers.Activation("gelu"),
            "elu": tf.keras.layers.ELU(),
        }
    else:
        raise BackendNotSupportedError(backend=backend, method="Activation._resolve()")

    act = avail_acts.get(name)
    if act is None:
        msg = f"Unknown activation name (`{name}`) for `{backend}` backend.Available activations: {avail_acts.keys()}"
        raise ActivationError(msg)
    return act
