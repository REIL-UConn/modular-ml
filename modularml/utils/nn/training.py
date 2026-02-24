"""Training helpers for preserving graph state and detaching tensors."""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np

from modularml.utils.environment.optional_imports import check_tensorflow, check_torch

if TYPE_CHECKING:
    from modularml.core.topology.model_graph import ModelGraph

TensorLike: TypeAlias = Any


@contextmanager
def preserve_frozen_state(model_graph: ModelGraph):
    """
    Context manager that restores frozen nodes after mutation.

    Args:
        model_graph (ModelGraph): Graph whose frozen nodes should be preserved.

    """
    # Node IDs of frozen nodes prior to context execution
    frozen_before = set(model_graph.frozen_nodes.keys())
    try:
        yield
    finally:
        # Node IDs of frozen nodes after context execution
        frozen_after = set(model_graph.frozen_nodes.keys())
        if frozen_after != frozen_before:
            model_graph.unfreeze(nodes=frozen_after)
            model_graph.freeze(nodes=frozen_before)


def detach_tensor(t: TensorLike):
    """
    Detach a tensor from its computation graph, backend-agnostic.

    Args:
        t (TensorLike): Tensor or array to detach.

    Returns:
        TensorLike: Detached copy when supported, otherwise the original object.

    """
    torch = check_torch()
    tf = check_tensorflow()

    # PyTorch
    if (torch is not None) and isinstance(t, torch.Tensor):
        return t.detach().clone()

    # Tensorflow
    if (tf is not None) and isinstance(t, tf.Tensor):
        return tf.identity(t)

    # NumPy
    if isinstance(t, np.ndarray):
        return t.copy()

    return t
