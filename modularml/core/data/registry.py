"""Registration helpers for :class:`FeatureSet` serialization and symbol lookup."""

from modularml.core.io.symbol_registry import symbol_registry
from modularml.core.io.conventions import SerializationKind, kind_registry

from .featureset import FeatureSet


def register_builtin():
    """Register :class:`FeatureSet` in the global symbol registry."""
    symbol_registry.register_builtin_class(
        key="FeatureSet",
        cls=FeatureSet,
    )


def register_kinds():
    """Register the :class:`FeatureSet` serialization kind."""
    kind_registry.register(
        cls=FeatureSet,
        kind=SerializationKind(name="FeatureSet", kind="fs"),
    )
