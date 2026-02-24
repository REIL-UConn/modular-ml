"""Registration helpers for sampling classes and serialization kinds."""

from modularml.core.io.symbol_registry import symbol_registry
from modularml.core.io.conventions import SerializationKind, kind_registry
from .similiarity_condition import SimilarityCondition
from .base_sampler import BaseSampler

from modularml.samplers import sampler_naming_fn, sampler_registry


def register_builtin():
    """Register built-in sampler classes and registries."""
    symbol_registry.register_builtin_class(
        key="SimilarityCondition",
        cls=SimilarityCondition,
    )

    symbol_registry.register_builtin_class(
        key="BaseSampler",
        cls=BaseSampler,
    )

    symbol_registry.register_builtin_registry(
        import_path="modularml.samplers.sampler_registry",
        registry=sampler_registry,
        naming_fn=sampler_naming_fn,
    )


def register_kinds():
    """Register serialization kinds for sampling-related classes."""
    kind_registry.register(
        cls=SimilarityCondition,
        kind=SerializationKind(name="SimilarityCondition", kind="smc"),
    )

    kind_registry.register(
        cls=BaseSampler,
        kind=SerializationKind(name="BaseSampler", kind="sm"),
    )
