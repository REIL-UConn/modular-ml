"""Checkpoint data structures for persisting stateful objects."""

from dataclasses import dataclass, field
from typing import Any

from modularml.core.io.protocols import Stateful


@dataclass
class CheckpointEntry:
    """
    Snapshot of a single :class:`Stateful` object.

    Attributes:
        version (str): Schema version for the checkpoint entry.
        entry_cls (type): Class of the serialized object.
        entry_state (dict[str, Any] | None): Serialized state dictionary.
        entry_obj (:class:`Stateful` | None): Live object reference, if available.

    """

    version = "1.0"

    entry_cls: type
    entry_state: dict[str, Any] | None = None
    entry_obj: Stateful | None = None


@dataclass
class Checkpoint:
    """
    Collection of :class:`CheckpointEntry` snapshots plus metadata.

    Attributes:
        version (str): Schema version for the checkpoint.
        entries (dict[str, CheckpointEntry]): Mapping of checkpoint keys to entries.
        meta (dict[str, Any]): Arbitrary metadata associated with the checkpoint.

    """

    version = "1.0"

    # Mapping of CheckpointEntry to a string-base key
    # E.g., {"node:encoder": CheckpointEntry}
    entries: dict[str, CheckpointEntry] = field(default_factory=dict)

    # Additional meta data to assign to this checkpoint
    meta: dict[str, Any] = field(default_factory=dict)

    def add_entry(self, key: str, obj: Stateful):
        """
        Add a checkpoint entry for a given object.

        Args:
            key (str): Unique label for the checkpoint entry.
            obj (Stateful): Object implementing :meth:`Stateful.get_state`.

        Raises:
            ValueError: If the key already exists.
            TypeError: If `obj` does not implement :class:`Stateful`.

        """
        if key in self.entries:
            msg = f"Key '{key}' already exists in this Checkpoint."
            raise ValueError(msg)

        if not isinstance(obj, Stateful):
            msg = "Checkpoints can only be created for Stateful objects."
            raise TypeError(obj)

        self.entries[key] = CheckpointEntry(
            entry_cls=type(obj),
            entry_obj=obj,
            entry_state=obj.get_state(),
        )

    def add_meta(self, key: str, value: Any):
        """
        Attach a metadata entry to this checkpoint.

        Args:
            key (str): Unique label for the metadata value.
            value (Any): Metadata value; must be pickle-able.

        Raises:
            ValueError: If the key already exists.

        """
        if key in self.meta:
            msg = f"Meta data key '{key}' already exists."
            raise ValueError(msg)
        self.meta[key] = value
