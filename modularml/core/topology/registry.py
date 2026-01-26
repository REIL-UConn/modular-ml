from modularml.core.io.symbol_registry import symbol_registry
from modularml.core.io.conventions import SerializationKind, kind_registry
from modularml.core.topology.model_node import ModelNode

from .model_graph import ModelGraph


def register_builtin():
    # Register base classes
    symbol_registry.register_builtin_class(
        key="ModelGraph",
        cls=ModelGraph,
    )
    symbol_registry.register_builtin_class(
        key="ModelNode",
        cls=ModelNode,
    )


def register_kinds():
    kind_registry.register(
        cls=ModelGraph,
        kind=SerializationKind(name="ModelGraph", kind="mg"),
    )
