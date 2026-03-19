"""Storage helpers for rich non-scalar artifacts recorded during phase execution."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from modularml.utils.data.multi_keyed_data import AxisSeries


class ArtifactEntry:
    """
    A single recorded artifact with its execution scope.

    Attributes:
        name (str): Artifact identifier.
        artifact (Any):
            The stored artifact. Transparently deserializes from disk on access
            when the parent ``ArtifactStore`` was configured with a location.
        epoch_idx (int): Epoch index at which the artifact was logged.
        batch_idx (int | None): Batch index, or ``None`` for epoch-level artifacts.

    """

    def __init__(
        self,
        name: str,
        artifact: Any,
        epoch_idx: int,
        batch_idx: int | None = None,
    ) -> None:
        self.name = name
        self._artifact = artifact
        self.epoch_idx = epoch_idx
        self.batch_idx = batch_idx

    @property
    def artifact(self) -> Any:
        """The artifact object. Transparently loads from disk if serialized as a Path."""
        if isinstance(self._artifact, Path):
            import pickle

            with self._artifact.open("rb") as f:
                return pickle.load(f)
        return self._artifact

    def __repr__(self) -> str:
        return (
            f"ArtifactEntry(name={self.name!r}, epoch_idx={self.epoch_idx}, "
            f"batch_idx={self.batch_idx})"
        )


@dataclass
class ArtifactDataSeries(AxisSeries[ArtifactEntry]):
    """
    ArtifactEntry objects keyed by (name, epoch, batch).

    Attributes:
        axes (tuple[str, ...]): Axis labels describing the series dimensions.
        _data (dict[tuple, ArtifactEntry]): Underlying mapping of axis keys to entries.
        supported_reduction_methods (ClassVar[set[str]]):
            Allowed reducers for :meth:`AxisSeries.collapse`.

    """

    supported_reduction_methods: ClassVar[set[str]] = {"first", "last"}

    def __repr__(self) -> str:
        return f"ArtifactDataSeries(keyed_by={self.axes}, len={len(self)})"


class ArtifactStore:
    """
    A flat namespace of named artifact objects recorded during phase execution.

    Artifacts can be any Python object (figures, DataFrames, arrays, text, etc.)
    keyed by name and execution scope. Pass a ``location`` directory to serialize
    artifacts to disk as they are logged, freeing them from RAM. All access
    transparently loads from disk as needed.
    """

    def __init__(self, location: Path | None = None) -> None:
        self._location = location
        self._entries: dict[str, list[ArtifactEntry]] = {}

    # ================================================
    # Writing
    # ================================================
    def log(
        self,
        *,
        name: str,
        artifact: Any,
        epoch_idx: int,
        batch_idx: int | None = None,
    ) -> None:
        """
        Record an artifact.

        Args:
            name (str): Artifact name (e.g. ``"val_scatter"``, ``"confusion_matrix"``).
            artifact (Any):
                The object to store. Serialized to disk when this store has a location.
            epoch_idx (int): The epoch at which this artifact was produced.
            batch_idx (int | None, optional):
                The batch index. Defaults to None (epoch-level artifact).

        """
        stored = artifact
        if self._location is not None:
            stored = self._save_to_disk(
                name=name,
                artifact=artifact,
                epoch_idx=epoch_idx,
                batch_idx=batch_idx,
            )
        entry = ArtifactEntry(
            name=name,
            artifact=stored,
            epoch_idx=epoch_idx,
            batch_idx=batch_idx,
        )
        if name not in self._entries:
            self._entries[name] = []
        self._entries[name].append(entry)

    def _save_to_disk(
        self,
        *,
        name: str,
        artifact: Any,
        epoch_idx: int,
        batch_idx: int | None,
    ) -> Path:
        """Serialize ``artifact`` to disk and return the file path."""
        import pickle

        batch_str = f"_b{batch_idx}" if batch_idx is not None else ""
        filename = f"{name}_e{epoch_idx}{batch_str}.pkl"
        filepath = Path(self._location) / filename  # type: ignore[arg-type]
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with filepath.open("wb") as f:
            pickle.dump(artifact, f)
        return filepath

    # ================================================
    # Reading
    # ================================================
    @property
    def names(self) -> list[str]:
        """List all recorded artifact names."""
        return [name for name, entries in self._entries.items() if entries]

    def entries(self) -> ArtifactDataSeries:
        """
        Get all artifact entries.

        Returns:
            ArtifactDataSeries: Entries keyed by ``(name, epoch, batch)``.

        """
        axes = ("name", "epoch", "batch")
        data: dict[tuple, ArtifactEntry] = {}
        for entries in self._entries.values():
            for entry in entries:
                key = (entry.name, entry.epoch_idx, entry.batch_idx)
                data[key] = entry
        return ArtifactDataSeries(axes=axes, _data=data)

    # ================================================
    # Snapshots / Reconstruction
    # ================================================
    def to_memory(self) -> ArtifactStore:
        """
        Return a new in-memory store with all artifacts loaded.

        Useful when you want a fully self-contained copy, for example before
        serializing the experiment or passing results to another process.

        Returns:
            ArtifactStore: New store with ``location=None`` and all artifacts in memory.

        """
        store = ArtifactStore(location=None)
        for name, entries in self._entries.items():
            store._entries[name] = [
                ArtifactEntry(
                    name=e.name,
                    artifact=e.artifact,  # triggers disk load if stored as Path
                    epoch_idx=e.epoch_idx,
                    batch_idx=e.batch_idx,
                )
                for e in entries
            ]
        return store

    @classmethod
    def from_directory(cls, location: Path) -> ArtifactStore:
        """
        Reconstruct a store index from existing pickle files in ``location``.

        Globs for ``*.pkl`` files and parses ``{name}_e{epoch}`` or
        ``{name}_e{epoch}_b{batch}`` from filenames.

        Args:
            location (Path): Directory containing serialized artifact files.

        Returns:
            ArtifactStore: Store with ``_entries`` populated as ``Path`` references.

        """
        store = cls(location=location)
        # Matches: {name}_e{epoch}.pkl  or  {name}_e{epoch}_b{batch}.pkl
        pattern = re.compile(r"^(.+)_e(\d+)(?:_b(\d+))?\.pkl$")
        for filepath in sorted(Path(location).glob("*.pkl")):
            m = pattern.match(filepath.name)
            if m is None:
                continue
            name = m.group(1)
            epoch_idx = int(m.group(2))
            batch_idx = int(m.group(3)) if m.group(3) is not None else None
            entry = ArtifactEntry(
                name=name,
                artifact=filepath,
                epoch_idx=epoch_idx,
                batch_idx=batch_idx,
            )
            if name not in store._entries:
                store._entries[name] = []
            store._entries[name].append(entry)
        return store

    # ================================================
    # Representation
    # ================================================
    def __repr__(self) -> str:
        return f"ArtifactStore(artifacts={self.names})"
