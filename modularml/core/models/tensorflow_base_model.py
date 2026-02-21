"""Base classes for native TensorFlow models in ModularML."""

from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING, Any

from modularml.core.models.base_model import BaseModel
from modularml.utils.environment.optional_imports import ensure_tensorflow
from modularml.utils.nn.backend import Backend

if TYPE_CHECKING:
    import numpy as np


class TensorflowBaseModel(BaseModel, ABC):
    """
    Base class for ModularML-native TensorFlow/Keras models.

    Description:
        Designed for framework-owned Keras implementations. User-defined
        :class:`tf.keras.Model` objects can subclass this base, though
        :class:`TensorflowModelWrapper` is typically simpler.

    """

    def __init__(self, **init_args: Any):
        """Initialize TensorFlow dependencies and call :class:`BaseModel`."""
        _ = ensure_tensorflow()

        # BaseModel handles backend + built flag
        _ = init_args.pop("backend", None)
        super().__init__(backend=Backend.TENSORFLOW, **init_args)

    # ================================================
    # Model Weights (Stateful)
    # ================================================
    def get_weights(self) -> dict[str, np.ndarray]:
        """Return model weights as numpy arrays keyed by variable name."""
        if not self.is_built:
            return {}
        # Subclasses must expose a `model` attribute holding the Keras model
        model = self._get_keras_model()
        return {var.name: var.numpy() for var in model.variables}

    def set_weights(self, weights: dict[str, np.ndarray]) -> None:
        """Restore numpy-based weights produced by :meth:`get_weights`."""
        if not weights:
            return
        model = self._get_keras_model()
        var_map = {var.name: var for var in model.variables}
        for name, value in weights.items():
            if name not in var_map:
                msg = (
                    f"Variable `{name}` not found in model. "
                    f"Available: {list(var_map.keys())}"
                )
                raise ValueError(msg)
            var_map[name].assign(value)

    def _get_keras_model(self):
        """
        Return the underlying Keras model for weight access.

        Returns:
            tf.keras.Model: The underlying model referenced for weights.

        Raises:
            AttributeError: If the subclass did not expose a model and did
                not override this helper.

        """
        if hasattr(self, "model") and self.model is not None:
            return self.model
        msg = (
            "No `model` attribute found. Subclasses of TensorflowBaseModel "
            "must either store a Keras model as `self.model` or override "
            "`_get_keras_model()`."
        )
        raise AttributeError(msg)
