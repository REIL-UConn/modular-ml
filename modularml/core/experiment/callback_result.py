from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Literal


@dataclass
class CallbackResult:
    """Base class for callback-emitted results."""

    kind: ClassVar[str] = "Callback"

    callback_label: str | None = None
    phase_label: str | None = None
    epoch_idx: int | None = None
    batch_idx: int | None = None
    edge: Literal["start", "end"] | None = None

    # ================================================
    # Scope Properties
    # ================================================
    @property
    def epoch(self) -> int | None:
        """The epoch at which this callback was executed."""
        return self.epoch_idx

    @property
    def batch(self) -> int | None:
        """The batch at which this callback was executed."""
        return self.batch_idx

    @property
    def scope(self) -> dict[str, Any]:
        """
        The point at which this callback was executed.

        The scope consists of:
        - `phase (str)`: the phase name
        - `epoch (int | None)`: the epoch index
        - `batch (int | None)`: the batch index
        - `edge ('start', 'end', None)` indicating on which edge of the
            phase/epoch/batch the callback was executed.

        """
        return {
            "phase": self.phase_label,
            "epoch": self.epoch_idx,
            "batch": self.batch_idx,
            "edge": self.edge,
        }

    def bind_scope(
        self,
        *,
        callback_label: str,
        phase_label: str,
        epoch_idx: int | None,
        batch_idx: int | None,
        edge: Literal["start", "end"] | None,
    ) -> CallbackResult:
        self.callback_label = callback_label
        self.phase_label = phase_label
        self.epoch_idx = epoch_idx
        self.batch_idx = batch_idx
        self.edge = edge
        return self


@dataclass
class PayloadResult(CallbackResult):
    kind: ClassVar[str] = "payload"
    payload: Any = None
