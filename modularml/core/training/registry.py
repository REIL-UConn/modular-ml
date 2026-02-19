"""Registry helpers for Loss and Optimizer serialization."""

from modularml.core.io.symbol_registry import symbol_registry
from modularml.core.io.conventions import SerializationKind, kind_registry
from modularml.core.training.loss import Loss

from .optimizer import Optimizer


def register_builtin():
    """Register :class:`Optimizer` and :class:`Loss` with to builtin symbol registry."""
    symbol_registry.register_builtin_class(
        key="Optimizer",
        cls=Optimizer,
    )
    symbol_registry.register_builtin_class(
        key="Loss",
        cls=Loss,
    )


def register_kinds():
    """Register kinds for :class:`Optimizer` and :class:`Loss` artifacts."""
    kind_registry.register(
        cls=Optimizer,
        kind=SerializationKind(name="Optimizer", kind="op"),
    )
    kind_registry.register(
        cls=Loss,
        kind=SerializationKind(name="Loss", kind="ls"),
    )
