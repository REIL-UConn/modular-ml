"""Base classes for native PyTorch models in ModularML."""

from abc import ABC
from typing import Any

import numpy as np

from modularml.core.models.base_model import BaseModel
from modularml.utils.environment.optional_imports import check_torch
from modularml.utils.nn.backend import Backend

torch = check_torch()

if torch is not None:
    TorchModuleBase = torch.nn.Module
else:

    class TorchModuleBase:
        def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: ARG002
            raise ImportError(
                "PyTorch is required to use TorchBaseModel. "
                "Install it with `pip install torch`.",
            )


class TorchBaseModel(BaseModel, TorchModuleBase, ABC):
    """
    Base class for ModularML-native PyTorch models.

    Description:
        Intended for framework-owned PyTorch architectures such as
        :class:`modularml.models.torch.SequentialMLP`. User-defined modules
        can subclass this base, although :class:`TorchModelWrapper` is
        usually simpler for existing :class:`torch.nn.Module` graphs.
    """

    def __init__(self, **init_args: Any):
        """Initialize the PyTorch + :class:`BaseModel` inheritance chain."""
        if torch is None:
            raise ImportError(
                "PyTorch is required to instantiate TorchBaseModel. "
                "Install it with `pip install torch`.",
            )

        # torch.nn.Module must be initialized first
        torch.nn.Module.__init__(self)

        # BaseModel handles backend + built flag
        _ = init_args.pop("backend", None)
        super().__init__(backend=Backend.TORCH, **init_args)

    # ================================================
    # Model Weights (Stateful)
    # ================================================
    def get_weights(self) -> dict[str, np.ndarray]:
        """Return PyTorch tensors as numpy arrays via :meth:`state_dict`."""
        if not self.is_built:
            return {}
        return {k: v.detach().cpu().numpy() for k, v in self.state_dict().items()}

    def set_weights(self, weights: dict[str, np.ndarray]) -> None:
        """Restore numpy-based weights produced by :meth:`get_weights`."""
        if not weights:
            return
        torch_state = {k: torch.as_tensor(v) for k, v in weights.items()}
        self.load_state_dict(torch_state, strict=True)

    def reset_weights(self) -> None:
        """Re-initialize all model weights using each layer's default initializer."""

        def _reset(m: TorchModuleBase) -> None:
            if hasattr(m, "reset_parameters") and callable(m.reset_parameters):
                m.reset_parameters()

        self.apply(_reset)
