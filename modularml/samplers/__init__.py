"""Built-in sampler registration utilities."""

from modularml.utils.registries import CaseInsensitiveRegistry

# Import all sampler modules
from .simple_sampler import SimpleSampler
from .n_sampler import NSampler
from .paired_sampler import PairedSampler
from .triplet_sampler import TripletSampler

__all__ = [
    "NSampler",
    "PairedSampler",
    "SimpleSampler",
    "TripletSampler",
]

# Create registry
sampler_registry = CaseInsensitiveRegistry()


def sampler_naming_fn(x):
    """
    Return the registry key for a sampler class.

    Args:
        x (type): Sampler class being registered.

    Returns:
        str: Qualname used to index the sampler in :data:`sampler_registry`.

    """
    return x.__qualname__


# Register modularml samplers
mml_samplers: list[type] = [
    NSampler,
    PairedSampler,
    SimpleSampler,
    TripletSampler,
]
for t in mml_samplers:
    sampler_registry.register(sampler_naming_fn(t), t)
