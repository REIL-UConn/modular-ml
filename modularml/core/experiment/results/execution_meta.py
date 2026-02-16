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
        """Total execution time in seconds."""
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
        """Total execution time in seconds."""
        return (self.ended_at - self.started_at).total_seconds()

    # ================================================
    # Runtime Modifiers
    # ================================================
    def add_child(self, meta: PhaseExecutionMeta | PhaseGroupExecutionMeta):
        """
        Add a phase or nested group metadata entry.

        Args:
            meta:
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
        """Iterate over children in execution order."""
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
