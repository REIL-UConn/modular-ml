"""Registration helpers for built-in splitter classes and serialization kinds."""

from modularml.core.io.symbol_registry import symbol_registry
from modularml.core.io.conventions import SerializationKind, kind_registry
from .base_splitter import BaseSplitter

from modularml.splitters import splitter_registry, splitter_naming_fn


def register_builtin():
    """Register built-in splitter classes and registries with the symbol registry."""
    symbol_registry.register_builtin_class(
        key="BaseSplitter",
        cls=BaseSplitter,
    )

    symbol_registry.register_builtin_registry(
        import_path="modularml.splitters.splitter_registry",
        registry=splitter_registry,
        naming_fn=splitter_naming_fn,
    )


def register_kinds():
    """Register serialization kinds for splitter classes."""
    kind_registry.register(
        cls=BaseSplitter,
        kind=SerializationKind(name="BaseSplitter", kind="sp"),
    )
