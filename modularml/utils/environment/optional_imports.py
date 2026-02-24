"""Helpers for optional third-party dependencies."""


def ensure_torch():
    """Import torch, raising a helpful error if unavailable."""
    try:
        import torch
    except ImportError as exc:
        msg = "torch is required. Please install torch to continue."
        raise ImportError(msg) from exc
    return torch


def check_torch():
    """Attempt to import torch, returning None if missing."""
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


def check_tensorflow():
    """Attempt to import tensorflow, returning None if missing."""
    try:
        import tensorflow as tf
    except ImportError:
        return None
    return tf
