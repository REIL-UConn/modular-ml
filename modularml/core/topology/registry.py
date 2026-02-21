"""Registry helpers for serializing topology components."""

from modularml.core.io.symbol_registry import symbol_registry
from modularml.core.io.conventions import SerializationKind, kind_registry
from modularml.core.topology.model_node import ModelNode
from modularml.core.topology.merge_nodes.merge_node import MergeNode

from .model_graph import ModelGraph


def register_builtin():
    """Register built-in topology classes with the global registry."""
    # Register base classes
    symbol_registry.register_builtin_class(
        key="ModelGraph",
        cls=ModelGraph,
    )
    symbol_registry.register_builtin_class(
        key="ModelNode",
        cls=ModelNode,
    )
    symbol_registry.register_builtin_class(
        key="MergeNode",
        cls=MergeNode,
    )


def register_kinds():
    """Register serialization kinds for topology classes."""
    kind_registry.register(
        cls=ModelGraph,
        kind=SerializationKind(name="ModelGraph", kind="mg"),
    )
