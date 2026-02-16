from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modularml.core.experiment.results.group_results import PhaseGroupResults


class ExecutionStrategy:
    """
    Base class for meta-execution strategies.

    Description:
        Execution strategies define how an Experiment (or a subset of its
        phases) should be executed under repeated or modified conditions.
    """

    def run(self) -> PhaseGroupResults:
        raise NotImplementedError
