"""Registration helpers for built-in reference classes."""

from modularml.core.io.symbol_registry import symbol_registry
# from modularml.core.io.conventions import SerializationKind, kind_registry

from .execution_reference import ExecutionReference
from .experiment_reference import (
    ExperimentReference,
    ExperimentNodeReference,
    GraphNodeReference,
)
from .featureset_reference import (
    FeatureSetReference,
    FeatureSetSplitReference,
    FeatureSetColumnReference,
)
from .model_io_reference import ModelOutputReference


def register_builtin():
    """
    Register built-in reference classes with the symbol registry.

    Returns:
        None: This function mutates the global registry in place.

    """
    # Register reference types
    symbol_registry.register_builtin_class(
        key="ExecutionReference",
        cls=ExecutionReference,
    )

    symbol_registry.register_builtin_class(
        key="ExperimentReference",
        cls=ExperimentReference,
    )
    symbol_registry.register_builtin_class(
        key="ExperimentNodeReference",
        cls=ExperimentNodeReference,
    )
    symbol_registry.register_builtin_class(
        key="GraphNodeReference",
        cls=GraphNodeReference,
    )

    symbol_registry.register_builtin_class(
        key="FeatureSetReference",
        cls=FeatureSetReference,
    )
    symbol_registry.register_builtin_class(
        key="FeatureSetSplitReference",
        cls=FeatureSetSplitReference,
    )
    symbol_registry.register_builtin_class(
        key="FeatureSetColumnReference",
        cls=FeatureSetColumnReference,
    )
    symbol_registry.register_builtin_class(
        key="ModelOutputReference",
        cls=ModelOutputReference,
    )


def register_kinds():
    """
    Register reference serialization kinds.

    Returns:
        None: Placeholder that should register :class:`SerializationKind` instances.

    """
    return
