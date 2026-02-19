"""Registration of built-in IO symbols and serialization kinds."""

from modularml.core.io.symbol_registry import symbol_registry
from modularml.core.io.conventions import SerializationKind, kind_registry

from .checkpoint import Checkpoint


def register_builtin():
    """Register built-in IO classes with the :class:`SymbolRegistry`."""
    # Register base classes
    symbol_registry.register_builtin_class(
        key="Checkpoint",
        cls=Checkpoint,
    )


def register_kinds():
    """Register serialization kinds for IO classes via :class:`KindRegistry`."""
    kind_registry.register(
        cls=Checkpoint,
        kind=SerializationKind(name="Checkpoint", kind="ckpt"),
    )
