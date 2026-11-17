"""Base class for models with explicit JAX parameter pytrees."""

from __future__ import annotations

from typing import Any

import numpy as np

from modularml.core.models.base_model import BaseModel
from modularml.utils.environment.optional_imports import ensure_jax
from modularml.utils.nn.backend import Backend


class JaxBaseModel(BaseModel):
    """
    Implement build/forward using JAX and store trainable leaves in ``params``.

    Subclasses must keep forward computations differentiable and must not mutate
    parameters during a forward pass. Non-trainable state belongs outside params.
    """

    def __init__(self, *, backend=Backend.JAX, **kwargs):
        ensure_jax()
        super().__init__(backend=backend, **kwargs)
        self.params: Any = None
        self._initial_params: Any = None

    def get_weights(self) -> dict[str, Any]:
        """Copy parameter leaves to NumPy for portable serialization."""
        jax = ensure_jax()
        return {
            "params": jax.tree_util.tree_map(
                lambda x: np.array(x, copy=True),
                self.params,
            ),
        }

    def set_weights(self, weights: dict[str, Any]) -> None:
        """Restore a parameter pytree without converting through Python lists."""
        jax = ensure_jax()
        self.params = jax.tree_util.tree_map(jax.numpy.asarray, weights["params"])

    def reset_weights(self) -> None:
        """Restore the parameters captured at build time."""
        if self._initial_params is None:
            raise RuntimeError("Build the JAX model before resetting weights.")
        self.set_weights({"params": self._initial_params})
