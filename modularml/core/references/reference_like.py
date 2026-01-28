from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from modularml.context.execution_context import ExecutionContext
    from modularml.context.experiment_context import ExperimentContext


@runtime_checkable
class ReferenceLike(Protocol):
    """
    Structural interface for reference objects.

    A ReferenceLike resolves into a concrete object given
    a runtime context.
    """

    def resolve(
        self,
        ctx: ExperimentContext | ExecutionContext | None = None,
    ) -> Any: ...

    # ================================================
    # Configurable
    # ================================================
    def get_config(self) -> dict[str, Any]:
        """
        Return configuration required to reconstruct this reference.

        Returns:
            dict[str, Any]: Reference configuration.

        """
        ...

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> ReferenceLike:
        """
        Construct a reference from configuration.

        Args:
            config (dict[str, Any]): Reference configuration.

        Returns:
            ReferenceLike: Reconstructed reference..

        """
        ...
