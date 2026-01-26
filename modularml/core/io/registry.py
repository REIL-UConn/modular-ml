from modularml.core.io.symbol_registry import symbol_registry
from modularml.core.io.conventions import SerializationKind, kind_registry

from .checkpoint import Checkpoint


def register_builtin():
    # Register base classes
    symbol_registry.register_builtin_class(
        key="Checkpoint",
        cls=Checkpoint,
    )


def register_kinds():
    kind_registry.register(
        cls=Checkpoint,
        kind=SerializationKind(name="Checkpoint", kind="ckpt"),
    )
