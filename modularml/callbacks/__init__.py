from modularml.utils.registries import CaseInsensitiveRegistry

# Import all sampler modules
from .evaluation import Evaluation

__all__ = [
    "Evaluation",
]

# Create registry
callback_registry = CaseInsensitiveRegistry()


def callback_naming_fn(x):
    return x.__qualname__


# Register modularml callbacks
mml_callbacks: list[type] = [
    Evaluation,
]
for t in mml_callbacks:
    callback_registry.register(callback_naming_fn(t), t)
