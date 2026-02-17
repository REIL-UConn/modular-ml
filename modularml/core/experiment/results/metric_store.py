from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import ClassVar

from modularml.utils.data.multi_keyed_data import AxisSeries


@dataclass
class MetricEntry:
    """A single recorded metric value with its execution scope."""

    name: str
    value: float
    epoch_idx: int
    batch_idx: int | None = None

    # ================================================
    # Reduction / Aggregation
    # ================================================
    @classmethod
    def sum(cls, *entries: MetricEntry) -> MetricEntry:
        """Sum all entries into a new instance."""
        # Check if passed list instead of separate args
        if (len(entries) == 1) and isinstance(entries[0], list):
            entries = entries[0]

        # Type / value checking
        ref = entries[0]
        for me in entries:
            # Validate type
            if not isinstance(me, MetricEntry):
                msg = (
                    f"Cannot add non-MetricEntry to MetricEntry. Received: {type(me)}."
                )
                raise TypeError(msg)

            # Enforce same name
            if me.name != ref.name:
                msg = (
                    "Cannot combine LossRecords with different names: "
                    f"{ref.name} != {me.name}."
                )
                raise ValueError(msg)

        # Combine
        eps = {e.epoch_idx for e in entries}
        bts = {e.batch_idx for e in entries}
        return MetricEntry(
            name=ref.name,
            value=sum(e.value for e in entries),
            epoch_idx=next(iter(eps)) if len(eps) == 1 else eps,
            batch_idx=next(iter(bts)) if len(bts) == 1 else bts,
        )

    def __add__(self, other: MetricEntry) -> MetricEntry:
        return type(self).sum(self, other)

    def __radd__(self, other: MetricEntry | int) -> MetricEntry:
        if other == 0:
            return self
        return type(self).sum(other, self)

    @classmethod
    def mean(cls, *entries: MetricEntry) -> MetricEntry:
        """Compute the mean of all metric entries."""
        # Check if passed list instead of separate args
        if (len(entries) == 1) and isinstance(entries[0], list):
            entries = entries[0]

        # Get sum of all records
        me_total = MetricEntry.sum(*entries)

        # Average and return
        me_total.value /= len(entries)
        return me_total


@dataclass
class MetricDataSeries(AxisSeries[MetricEntry]):
    """MetricEntry objects keyed by (name, epoch, batch)."""

    supported_reduction_methods: ClassVar[set[str]] = {
        "first",
        "last",
        "sum",
        "mean",
    }

    def __repr__(self):
        return f"MetricDataSeries(keyed_by={self.axes}, len={len(self)})"


class MetricStore:
    """
    A flat namespace of named scalar metric values recorded during phase execution.

    Description:
        MetricStore provides a simple key-value ledger for named scalar metrics
        (e.g. "val_loss", "train_loss", "val_r2"). Any producer (MetricCallbacks,
        the training loop itself) can log values, and any consumer (EarlyStopping,
        progress bars) can read them by name.

        Entries are stored per-name in insertion order. Each entry is tagged with
        its epoch and optional batch index.

    """

    def __init__(self) -> None:
        self._entries: dict[str, list[MetricEntry]] = defaultdict(list)

    # ================================================
    # Writing
    # ================================================
    def log(
        self,
        *,
        name: str,
        value: float,
        epoch_idx: int,
        batch_idx: int | None = None,
    ) -> None:
        """
        Record a metric value.

        Args:
            name (str):
                Metric name (e.g. "val_loss", "train_loss").
            value (float):
                The scalar value to record.
            epoch_idx (int):
                The epoch at which this value was recorded.
            batch_idx (int | None, optional):
                The batch at which this value was recorded. If None, this is
                an epoch-level metric. Defaults to None.

        """
        self._entries[name].append(
            MetricEntry(
                name=name,
                value=value,
                epoch_idx=epoch_idx,
                batch_idx=batch_idx,
            ),
        )

    # ================================================
    # Reading
    # ================================================
    @property
    def names(self) -> list[str]:
        """
        List all recorded metric names.

        Returns:
            list[str]: Metric names that have at least one entry.

        """
        return [name for name, entries in self._entries.items() if entries]

    def entries(self) -> MetricDataSeries:
        """
        Get all metric entries.

        Returns:
            MetricDataSeries:
                Entries keyed by `(name, epoch, batch)`.
                Epoch-level entries have `batch=None`.

        """
        axes = ("name", "epoch", "batch")
        data: dict[tuple, MetricEntry] = {}
        for entries in self._entries.values():
            for entry in entries:
                key = (entry.name, entry.epoch_idx, entry.batch_idx)
                data[key] = entry
        return MetricDataSeries(axes=axes, _data=data)

    # ================================================
    # Representation
    # ================================================
    def __repr__(self) -> str:
        return f"MetricStore(metrics={self.names})"
