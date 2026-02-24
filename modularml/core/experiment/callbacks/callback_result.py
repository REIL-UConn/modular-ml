"""Result payloads emitted by experiment callbacks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Literal


@dataclass
class CallbackResult:
    """
    Base class for callback-emitted results.

    Attributes:
        kind (ClassVar[str]): Discriminator label for serialization.
        callback_label (str | None): Callback identifier that produced the result.
        phase_label (str | None): Phase label associated with the callback scope.
        epoch_idx (int | None): Epoch index in which the callback ran.
        batch_idx (int | None): Batch index in which the callback ran.
        edge (Literal["start", "end"] | None):
            Lifecycle edge (`start` or `end`) at which the callback fired.

    """

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
        """
        The epoch at which this callback was executed.

        Returns:
            int | None: Epoch index for the callback.

        """
        return self.epoch_idx

    @property
    def batch(self) -> int | None:
        """
        The batch at which this callback was executed.

        Returns:
            int | None: Batch index for the callback.

        """
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

        Returns:
            dict[str, Any]: Dictionary containing `phase`, `epoch`, `batch`, and `edge`.

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
        """
        Attach contextual scope metadata to this result.

        Args:
            callback_label (str): Label of the emitting callback.
            phase_label (str): Label of the associated phase.
            epoch_idx (int | None): Epoch index for the event.
            batch_idx (int | None): Batch index for the event.
            edge (Literal["start", "end"] | None): Lifecycle edge when executed.

        Returns:
            CallbackResult: Self for chaining.

        """
        self.callback_label = callback_label
        self.phase_label = phase_label
        self.epoch_idx = epoch_idx
        self.batch_idx = batch_idx
        self.edge = edge
        return self


@dataclass
class PayloadResult(CallbackResult):
    """
    Generic payload wrapper for callback return values.

    Attributes:
        payload (Any): Arbitrary data emitted by the callback.

    """

    kind: ClassVar[str] = "payload"
    payload: Any = None
