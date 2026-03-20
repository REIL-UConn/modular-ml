"""Storage helpers for metrics recorded during phase execution."""

from __future__ import annotations

import pickle
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from modularml.utils.data.multi_keyed_data import AxisSeries


@dataclass
class MetricEntry:
    """
    A single recorded metric value with its execution scope.

    Attributes:
        name (str): Metric identifier.
        value (float): Recorded scalar value.
        epoch_idx (int): Epoch index at which the metric was logged.
        batch_idx (int | None): Batch index if the metric is per-batch.

    """

    name: str
    value: float
    epoch_idx: int
    batch_idx: int | None = None

    # ================================================
    # Reduction / Aggregation
    # ================================================
    @classmethod
    def sum(cls, *entries: MetricEntry) -> MetricEntry:
        """
        Sum all entries into a new instance.

        Args:
            *entries (MetricEntry | list[MetricEntry]):
                Metric entries to aggregate. Accepts splatted entries or a list.

        Returns:
            MetricEntry: New entry with summed value.

        Raises:
            TypeError: If any entry is not a :class:`MetricEntry`.
            ValueError: If metric names differ across entries.

        """
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
        """
        Compute the mean of all metric entries.

        Args:
            *entries (MetricEntry | list[MetricEntry]):
                Metric entries to average. Accepts splatted entries or a list.

        Returns:
            MetricEntry: New entry representing the mean value.

        """
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
    """
    MetricEntry objects keyed by (name, epoch, batch).

    Attributes:
        axes (tuple[str, ...]): Axis labels describing the series dimensions.
        _data (dict[tuple, MetricEntry]): Underlying mapping of axis keys to entries.
        supported_reduction_methods (ClassVar[set[str]]):
            Allowed reducers for :meth:`AxisSeries.collapse`.

    """

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

    Entries are stored per-name in insertion order, each tagged with an epoch
    and optional batch index. When a ``location`` directory is provided,
    each entry is also pickled to disk as it is recorded so that large
    result sets do not exhaust RAM.
    """

    def __init__(self, location: Path | None = None) -> None:
        self._location: Path | None = location
        self._entries: dict[str, list[MetricEntry]] = defaultdict(list)
        self._count: int = 0

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
        entry = MetricEntry(
            name=name,
            value=value,
            epoch_idx=epoch_idx,
            batch_idx=batch_idx,
        )
        self._entries[name].append(entry)
        if self._location is not None:
            self._save_to_disk(entry)

    def _save_to_disk(self, entry: MetricEntry) -> None:
        """Pickle a single :class:`MetricEntry` to ``_location``."""
        filepath = Path(self._location) / f"{self._count}.pkl"
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with filepath.open("wb") as f:
            pickle.dump(entry, f, protocol=pickle.HIGHEST_PROTOCOL)
        self._count += 1

    # ================================================
    # Reading
    # ================================================
    @property
    def names(self) -> list[str]:
        """List all recorded metric names."""
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
    # Snapshots / Reconstruction
    # ================================================
    def to_memory(self) -> MetricStore:
        """
        Return a new in-memory :class:`MetricStore` with all entries loaded.

        Returns:
            MetricStore: In-memory copy of this store.

        """
        new_store = MetricStore(location=None)
        new_store._entries = defaultdict(
            list,
            {name: list(entries) for name, entries in self._entries.items()},
        )
        return new_store

    @classmethod
    def from_directory(cls, location: Path) -> MetricStore:
        """
        Reconstruct a :class:`MetricStore` from pickle files in ``location``.

        Args:
            location (Path): Directory containing ``*.pkl`` metric files.

        Returns:
            MetricStore: Store with all entries loaded from disk.

        """
        location = Path(location)
        store = cls(location=location)
        pkl_files = sorted(location.glob("*.pkl"), key=lambda p: int(p.stem))
        for pkl_file in pkl_files:
            with pkl_file.open("rb") as f:
                entry: MetricEntry = pickle.load(f)
            store._entries[entry.name].append(entry)
        store._count = len(pkl_files)
        return store

    # ================================================
    # Representation
    # ================================================
    def __repr__(self) -> str:
        return f"MetricStore(metrics={self.names})"
