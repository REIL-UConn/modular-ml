from dataclasses import dataclass, field
from typing import Any

from modularml.core.io.protocols import Stateful


@dataclass
class CheckpointEntry:
    version = "1.0"

    entry_cls: type
    entry_state: dict[str, Any] | None = None
    entry_obj: Stateful | None = None


@dataclass
class Checkpoint:
    version = "1.0"

    # Mapping of CheckpointEntry to a string-base key
    # E.g., {"node:encoder": CheckpointEntry}
    entries: dict[str, CheckpointEntry] = field(default_factory=dict)

    def add_entry(self, key: str, obj: Stateful):
        """
        Adds a checkpoint entry for a given object.

        Args:
            key (str):
                A unique label to assign to this object's checkpoint entry.

            obj (Stateful):
                The object to add a checkpoint entry for.
                The object must implement `get_state` and `set_state`.

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
