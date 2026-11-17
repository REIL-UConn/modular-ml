"""Helpers for optional third-party dependencies."""

from functools import cache


@cache
def check_jax():
    """
    Attempt to import JAX, returning None if missing.

    The result is cached: a failed import is not stored by Python, so without
    caching every call would re-scan `sys.path` for the missing package.
    """
    try:
        import jax
    except ImportError:
        return None
    return jax


def ensure_jax():
    """Import JAX, raising an actionable error if unavailable."""
    jax = check_jax()
    if jax is None:
        raise ImportError("JAX is required. Install modularml[jax] to continue.")
    return jax


def ensure_optax():
    """Import Optax for JAX optimizers and named losses."""
    try:
        import optax
    except ImportError as exc:
        raise ImportError(
            "Optax is required. Install modularml[jax] to continue.",
        ) from exc
    return optax


def ensure_torch():
    """Import torch, raising a helpful error if unavailable."""
    try:
        import torch
    except ImportError as exc:
        msg = "torch is required. Please install torch to continue."
        raise ImportError(msg) from exc
    return torch


@cache
def check_torch():
    """Attempt to import torch, returning None if missing (result is cached)."""
    try:
        import torch
    except ImportError:
        return None
    return torch


def ensure_tensorflow():
    """Import tensorflow, raising a helpful error if unavailable."""
    try:
        import tensorflow as tf
    except ImportError as exc:
        msg = "tensorflow is required. Please install tensorflow to continue."
        raise ImportError(msg) from exc
    return tf


@cache
def check_tensorflow():
    """Attempt to import tensorflow, returning None if missing (result is cached)."""
    try:
        import tensorflow as tf
    except ImportError:
        return None
    return tf
