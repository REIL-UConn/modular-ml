"""Container describing a single execution of an experiment plan."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from datetime import datetime

    from modularml.core.experiment.results.execution_meta import PhaseGroupExecutionMeta
    from modularml.core.experiment.results.group_results import PhaseGroupResults


@dataclass
class ExperimentRun:
    """
    Represents a single execution of an Experiment execution plan.

    Description:
        Stores run-level timing information, execution status, the
        PhaseGroupResults produced by execution, and a hierarchical
        execution metadata tree describing per-phase timing and status.

        Each call to `Experiment.run()` produces one ExperimentRun
        instance that is appended to Experiment.history.

    Attributes:
        label (str): Human-readable identifier for the run.
        started_at (datetime): Timestamp when execution began.
        ended_at (datetime): Timestamp when execution completed.
        status (Literal["completed", "failed", "stopped"]):
            Terminal run status.
        results (PhaseGroupResults): Execution outputs.
        execution_meta (PhaseGroupExecutionMeta):
            Hierarchical metadata describing per-phase execution.
        metadata (dict[str, Any]): Arbitrary run-level metadata.

    """

    label: str

    started_at: datetime
    ended_at: datetime
    status: Literal["completed", "failed", "stopped"]

    results: PhaseGroupResults
    execution_meta: PhaseGroupExecutionMeta

    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_seconds(self) -> float:
        """
        Total experiment execution time in seconds.

        Returns:
            float: Duration derived from the recorded timestamps.

        """
        return (self.ended_at - self.started_at).total_seconds()

    @property
    def phase_durations(self) -> dict[str, float]:
        """
        Flattened dict of phase labels and durations.

        Returns:
            dict[str, float]: Phase labels mapped to execution durations.

        """
        return {
            phase_lbl: phase_meta.duration_seconds
            for phase_lbl, phase_meta in self.execution_meta.flatten().items()
        }

    def __repr__(self):
        return (
            f"ExperimentRun(label='{self.label}', "
            f"duration={self.duration_seconds}, "
            f"status='{self.status}', "
            f"results={self.results!r})"
        )
