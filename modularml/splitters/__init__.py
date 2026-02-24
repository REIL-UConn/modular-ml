"""Built-in FeatureSet splitter registrations."""

from modularml.utils.registries import CaseInsensitiveRegistry

# Import all splitter
from .random_splitter import RandomSplitter
from .condition_splitter import ConditionSplitter


__all__ = ["ConditionSplitter", "RandomSplitter"]

# Create registry
splitter_registry = CaseInsensitiveRegistry()


def splitter_naming_fn(x):
    """
    Return the registry key for a splitter class.

    Args:
        x (type): Splitter class being registered.

    Returns:
        str: Qualname used to index the splitter in :data:`splitter_registry`.

    """
    return x.__qualname__


# Register modularml splitters
mml_scalers: list[type] = [RandomSplitter, ConditionSplitter]
for t in mml_scalers:
    splitter_registry.register(splitter_naming_fn(t), t)
