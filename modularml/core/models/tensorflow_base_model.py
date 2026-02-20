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
    Base class for all ModularML-native TensorFlow/Keras models.

    This class is intended for framework-owned models
    (e.g., SequentialMLP, SequentialCNN built on Keras).

    User-defined tf.keras.Model objects can subclass this,
    but it is easier to use the TensorflowModelWrapper.

    Note:
        Unlike TorchBaseModel which uses multiple inheritance with
        torch.nn.Module, this class does NOT inherit from tf.keras.Model
        directly to avoid MRO conflicts with BaseModel's metaclass.
        Subclasses should create their Keras layers/model in `build()`
        and store them as instance attributes.

    """

    def __init__(self, **init_args: Any):
        _ = ensure_tensorflow()

        # BaseModel handles backend + built flag
        _ = init_args.pop("backend", None)
        super().__init__(backend=Backend.TENSORFLOW, **init_args)

    # ================================================
    # Model Weights (Stateful)
    # ================================================
    def get_weights(self) -> dict[str, np.ndarray]:
        """Weights are returned as a mapping of variable names to np arrays."""
        if not self.is_built:
            return {}
        # Subclasses must expose a `model` attribute holding the Keras model
        model = self._get_keras_model()
        return {var.name: var.numpy() for var in model.variables}

    def set_weights(self, weights: dict[str, np.ndarray]) -> None:
        """Restore weights retrieved from `get_weights`."""
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

        Subclasses should override this if the Keras model is not stored
        as `self.model`. By default, looks for `self.model`.

        Returns:
            tf.keras.Model: The underlying Keras model.

        Raises:
            AttributeError: If no model attribute is found.

        """
        if hasattr(self, "model") and self.model is not None:
            return self.model
        msg = (
            "No `model` attribute found. Subclasses of TensorflowBaseModel "
            "must either store a Keras model as `self.model` or override "
            "`_get_keras_model()`."
        )
        raise AttributeError(msg)
