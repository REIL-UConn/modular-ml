"""Execution strategy interfaces."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modularml.core.experiment.results.group_results import PhaseGroupResults


class ExecutionStrategy:
    """
    Base class for meta-execution strategies.

    Description:
        Execution strategies define how an :class:`Experiment` (or a subset of
        its phases) should be executed under repeated or modified conditions.
    """

    def run(self) -> PhaseGroupResults:
        """
        Execute the strategy and return aggregated phase results.

        Returns:
            :class:`PhaseGroupResults`: Results produced by the execution plan.

        Raises:
            NotImplementedError: Always raised; subclasses must override.

        """
        raise NotImplementedError
