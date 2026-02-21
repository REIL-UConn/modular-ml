"""Built-in model registry helpers for ModularML."""

from modularml.utils.registries import CaseInsensitiveRegistry
from modularml.utils.nn.backend import Backend

# Import all pre-built models
from .torch.sequential_mlp import SequentialMLP as Torch_SequentialMLP
from .torch.sequential_cnn import SequentialCNN as Torch_SequentialCNN


# Create registry
model_registry = CaseInsensitiveRegistry()


def model_naming_fn(x: type):
    """
    Return the canonical registry name for a model class.

    Args:
        x (type): Model class to register.

    Returns:
        str: Fully-qualified registry key in ``backend::ClassName`` form.

    """
    backend: str = x.__module__.split(".")[-2]  # folder name (eg, "torch")
    return f"{backend}::{x.__qualname__}"  # eg "torch::SequentialMLP"


torch_models = [
    Torch_SequentialMLP,
    Torch_SequentialCNN,
]
for t in torch_models:
    model_registry.register(model_naming_fn(t), t)
