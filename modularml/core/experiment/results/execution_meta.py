"""Metadata structures describing phase execution timing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Iterator
    from datetime import datetime


@dataclass
class PhaseExecutionMeta:
    """
    Execution metadata for a single ExperimentPhase.

    Description:
        Stores timing and other meta information for execution of a single
        phase. Does not contain results; only encodes meta data to be attached
        along with results.

    Attributes:
        label (str): Phase label.
        started_at (datetime): Timestamp when the phase began.
        ended_at (datetime): Timestamp when the phase completed.
        status (Literal["completed", "failed", "stopped"]):
            Terminal execution status.
        metadata (dict): Additional metadata captured during execution.

    """

    label: str
    started_at: datetime
    ended_at: datetime
    status: Literal["completed", "failed", "stopped"]
    metadata: dict = field(default_factory=dict)

    # ================================================
    # Properties
    # ================================================
    @property
    def duration_seconds(self) -> float:
        """
        Total execution time in seconds.

        Returns:
            float: `ended_at - started_at` in seconds.

        """
        return (self.ended_at - self.started_at).total_seconds()


@dataclass
class PhaseGroupExecutionMeta:
    """
    Execution metadata container mirroring a PhaseGroup hierarchy.

    Description:
        Stores timing and execution metadata for a group of phases and/or
        nested phase groups. The structure matches the executed PhaseGroup,
        preserving nesting and execution order.

        Does not store results; only execution-level metadata.

    Attributes:
        label (str): Group label.
        started_at (datetime): When group execution began.
        ended_at (datetime | None): Completion time if finished.
        _children (dict[str, PhaseExecutionMeta | PhaseGroupExecutionMeta]):
            Mapping of child labels to their metadata.

    """

    label: str
    started_at: datetime
    ended_at: datetime | None

    _children: dict[str, PhaseExecutionMeta | PhaseGroupExecutionMeta] = field(
        default_factory=dict,
    )

    # ================================================
    # Properties
    # ================================================
    @property
    def duration_seconds(self) -> float:
        """
        Total execution time in seconds.

        Returns:
            float: Duration derived from timestamps.

        """
        return (self.ended_at - self.started_at).total_seconds()

    # ================================================
    # Runtime Modifiers
    # ================================================
    def add_child(self, meta: PhaseExecutionMeta | PhaseGroupExecutionMeta):
        """
        Add a phase or nested group metadata entry.

        Args:
            meta (PhaseExecutionMeta | PhaseGroupExecutionMeta):
                Execution metadata for a phase or nested group.

        """
        if meta.label in self._children:
            msg = (
                f"Duplicate execution meta label '{meta.label}' in "
                f"group '{self.label}'."
            )
            raise ValueError(msg)
        self._children[meta.label] = meta

    # ================================================
    # Accessors
    # ================================================
    def items(
        self,
    ) -> Iterator[tuple[str, PhaseExecutionMeta | PhaseGroupExecutionMeta]]:
        """
        Iterate over children in execution order.

        Returns:
            Iterator[tuple[str, PhaseExecutionMeta | PhaseGroupExecutionMeta]]:
                Label/metadata pairs.

        """
        yield from self._children.items()

    # ================================================
    # Flattening
    # ================================================
    def _collect_flat(
        self,
        *,
        into: dict[str, PhaseExecutionMeta],
        duplicates: list[str],
    ) -> None:
        """Recursively collect PhaseExecutionMeta into a flat dict."""
        for label, result in self._children.items():
            if isinstance(result, PhaseGroupExecutionMeta):
                result._collect_flat(into=into, duplicates=duplicates)
            else:
                if label in into:
                    duplicates.append(label)
                into[label] = result

    def flatten(self) -> dict[str, PhaseExecutionMeta]:
        """Flatten nested structure into execution-ordered list of phases."""
        flat: dict[str, PhaseExecutionMeta] = []
        duplicates: list[str] = []

        self._collect_flat(into=flat, duplicates=duplicates)
        if duplicates:
            msg = (
                "Cannot flatten PhaseGroupExecutionMeta; duplicate phase labels "
                f"found across the hierarchy: {duplicates}."
            )
            raise ValueError(msg)

        return flat
