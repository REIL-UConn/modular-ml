"""Storage helpers for rich non-scalar artifacts recorded during phase execution."""

from __future__ import annotations

import pickle
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
            with self._artifact.open("rb") as f:
                payload = pickle.load(f)

            # New format: dict payload containing the artifact under "artifact" key
            if isinstance(payload, dict) and "artifact" in payload:
                return payload["artifact"]

            # Legacy format: file contained just the artifact object
            return payload
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
        self._count: int = 0

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
        if self._location is not None:
            artifact_value = self._save_to_disk(
                name=name,
                artifact=artifact,
                epoch_idx=epoch_idx,
                batch_idx=batch_idx,
            )
        else:
            artifact_value = artifact
        entry = ArtifactEntry(
            name=name,
            artifact=artifact_value,
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
        """Serialize artifact and metadata to disk, returning the file path."""
        import pickle

        filepath = Path(self._location) / f"{self._count}.pkl"  # type: ignore[arg-type]
        filepath.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "name": name,
            "epoch_idx": epoch_idx,
            "batch_idx": batch_idx,
            "artifact": artifact,
        }
        with filepath.open("wb") as f:
            pickle.dump(payload, f)
        self._count += 1
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

        Args:
            location (Path): Directory containing serialized artifact files.

        Returns:
            ArtifactStore: Store with ``_entries`` populated as ``Path`` references.

        """
        import pickle

        store = cls(location=location)
        pkl_files = sorted(Path(location).glob("*.pkl"), key=lambda p: int(p.stem))
        for filepath in pkl_files:
            with filepath.open("rb") as f:
                payload = pickle.load(f)
            name = payload["name"]
            entry = ArtifactEntry(
                name=name,
                artifact=filepath,  # lazy-load
                epoch_idx=payload["epoch_idx"],
                batch_idx=payload["batch_idx"],
            )
            if name not in store._entries:
                store._entries[name] = []
            store._entries[name].append(entry)
        store._count = len(pkl_files)
        return store

    # ================================================
    # Representation
    # ================================================
    def __repr__(self) -> str:
        return f"ArtifactStore(artifacts={self.names})"
