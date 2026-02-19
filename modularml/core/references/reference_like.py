"""Structural protocol for reference-like objects."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from modularml.core.data.execution_context import ExecutionContext
    from modularml.core.experiment.experiment_context import ExperimentContext


@runtime_checkable
class ReferenceLike(Protocol):
    """
    Structural interface for reference objects.

    Description:
        A :class:`ReferenceLike` resolves into a concrete object when provided with
        either an :class:`ExperimentContext` or an :class:`ExecutionContext`.
    """

    def resolve(
        self,
        ctx: ExperimentContext | ExecutionContext | None = None,
    ) -> Any:
        """
        Resolve the reference within an experiment or execution context.

        Args:
            ctx (ExperimentContext | ExecutionContext | None): Context to resolve against.

        Returns:
            Any: Concrete value resolved by the reference.

        """

    # ================================================
    # Configurable
    # ================================================
    def get_config(self) -> dict[str, Any]:
        """
        Return a JSON-serializable configuration for this reference.

        Returns:
            dict[str, Any]: Serialized configuration values.

        """

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> ReferenceLike:
        """
        Construct a reference from a configuration dictionary.

        Args:
            config (dict[str, Any]): Serialized configuration values.

        Returns:
            ReferenceLike: Concrete reference defined by the configuration.

        """
