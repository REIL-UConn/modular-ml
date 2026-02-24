"""Checkpointing utilities shared by experiments and train phases."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal

from modularml.utils.data.formatting import ensure_list
from modularml.utils.logging.logger import get_logger
from modularml.utils.logging.warnings import warn

logger = get_logger(name="Checkpointing")

# Valid hook names by context
TRAINING_HOOKS = frozenset(
    {
        "phase_start",
        "phase_end",
        "epoch_start",
        "epoch_end",
        "batch_start",
        "batch_end",
    },
)

EXPERIMENT_HOOKS = frozenset(
    {
        "phase_start",
        "phase_end",
        "group_start",
        "group_end",
        "experiment_start",
        "experiment_end",
    },
)

# Valid placeholders by context
TRAINING_PLACEHOLDERS = frozenset({"phase", "epoch", "batch"})
EXPERIMENT_PLACEHOLDERS = frozenset({"label"})

# Default name templates per context
TRAINING_NAME_TEMPLATE = "{phase}_epoch{epoch}"
EXPERIMENT_NAME_TEMPLATE = "{label}"


class Checkpointing:
    """
    Automatic state checkpointing configuration and tracker.

    Description:
        Manages when/where to save and tracks saved checkpoints.
        The actual save/restore operations are performed by the parent object
        (:class:`TrainPhase` for model state, :class:`Experiment` for full
        experiment state).

        When attached to a :class:`TrainPhase`, model graph state is saved at
        training lifecycle hooks (epoch boundaries, etc.). When attached
        to an :class:`Experiment`, the full experiment (including results) is
        saved at execution lifecycle hooks (phase/group boundaries).

    Attributes:
        mode (str): Storage mode (`memory` or `disk`).
        save_on (list[str]): Lifecycle hooks that trigger saving.
        every_n (int): Frequency multiplier applied to hook counts.
        directory (Path | None): Destination directory for disk checkpoints.
        name_template (str | None): Template used to format checkpoint names.
        max_to_keep (int | None): Limit on retained checkpoints (None keeps all).
        overwrite (bool): Whether disk checkpoints may overwrite existing files.

    """

    def __init__(
        self,
        *,
        mode: Literal["memory", "disk"] = "memory",
        save_on: str | list[str] = "epoch_end",
        every_n: int = 1,
        # Disk mode settings
        directory: Path | str | None = None,
        name_template: str | None = None,
        max_to_keep: int | None = None,
        overwrite: bool = True,
    ) -> None:
        """
        Initialize a Checkpointing configuration.

        Args:
            mode (Literal["memory", "disk"], optional):
                Storage mode for checkpoints. `memory` uses in-memory
                state snapshots (fast, no I/O). `disk` writes checkpoint
                files to the filesystem (persistent). Defaults to `memory`.

            save_on (str | list[str], optional):
                Lifecycle hook(s) at which to save checkpoints. Valid values
                depend on where the Checkpointing is attached:

                - :class:`TrainPhase`:
                  `phase_start`, `phase_end`, `epoch_start`,
                  `epoch_end`, `batch_start`, `batch_end`
                - :class:`Experiment`: `phase_start`, `phase_end`,
                  `group_start`, `group_end`

                Defaults to `epoch_end`.

            every_n (int, optional):
                Save a checkpoint every `every_n` occurrences of the hook.
                For `epoch_end`, this means every N epochs.
                Defaults to 1.

            directory (Path | str | None, optional):
                Directory for disk-mode checkpoints. Ignored in memory mode.
                When attached to a :class:`TrainPhase` and left as None, the directory
                is inherited from the parent :class:`Experiment` at execution time
                (saved under a `{phase.label}/` subdirectory).

            name_template (str | None, optional):
                Template for checkpoint file names. Available placeholders
                depend on context. For TrainPhase: `{phase}`, `{epoch}`,
                `{batch}`. For Experiment: `{label}`.
                If None, a context-appropriate default is assigned by the
                parent's `set_checkpointing()` method. Defaults to None.

            max_to_keep (int | None, optional):
                Maximum number of checkpoints to retain. When exceeded, the
                oldest checkpoint is removed. If None, all checkpoints are
                kept. Defaults to None.

            overwrite (bool, optional):
                Whether to overwrite existing checkpoint files in disk mode.
                Defaults to True.

        """
        # Validate mode
        if mode not in ("memory", "disk"):
            msg = f"Invalid mode: '{mode}'. Expected 'memory' or 'disk'."
            raise ValueError(msg)
        self._mode = mode

        # Normalize save_on (validation is deferred to the parent)
        self._save_on: list[str] = ensure_list(save_on)

        # Validate every_n
        if every_n < 1:
            msg = "every_n must be >= 1."
            raise ValueError(msg)
        self._every_n = every_n

        # Disk mode settings
        self._directory: Path | None = (
            Path(directory) if directory is not None else None
        )
        self._name_template = name_template
        self._max_to_keep = max_to_keep
        self._overwrite = overwrite

        # Warn if the directory already contains checkpoint files
        if self._directory is not None and self._directory.is_dir():
            existing = list(self._directory.glob("*.ckpt.mml"))
            if existing:
                warn(
                    f"Checkpoint directory '{self._directory}' already "
                    f"contains {len(existing)} checkpoint file(s). Existing "
                    f"files will only be overwritten if an exact name match "
                    f"occurs and overwrite=True.",
                    stacklevel=2,
                )

        # Internal state
        self._memory_states: dict[int | str, dict[str, Any]] = {}
        self._disk_paths: dict[int | str, Path] = {}

        # Tracks insertion order for max_to_keep eviction
        self._key_order: list[int | str] = []

        # Counter for hook occurrences (used with every_n)
        self._hook_counts: dict[str, int] = {}

    # ================================================
    # Properties
    # ================================================
    @property
    def mode(self) -> str:
        """
        Get the storage mode.

        Returns:
            str: Either `memory` or `disk`.

        """
        return self._mode

    @property
    def save_on(self) -> list[str]:
        """
        Get the hook names at which checkpoints are saved.

        Returns:
            list[str]: Lifecycle hooks configured for saving.

        """
        return list(self._save_on)

    @property
    def every_n(self) -> int:
        """
        Get the save frequency.

        Returns:
            int: Number of hook invocations between saves.

        """
        return self._every_n

    @property
    def directory(self) -> Path | None:
        """
        Get the checkpoint directory.

        Returns:
            Path | None: Disk directory used when `mode` is `disk`.

        """
        return self._directory

    @property
    def overwrite(self) -> bool:
        """
        Determine if disk checkpoints overwrite existing files.

        Returns:
            bool: True if overwrites are allowed.

        """
        return self._overwrite

    @property
    def name_template(self) -> str | None:
        """
        Get the template for checkpoint names.

        Returns:
            str | None: Template string with placeholders such as `{phase}`.

        """
        return self._name_template

    @name_template.setter
    def name_template(self, value: str) -> None:
        """
        Update the template used when formatting checkpoint names.

        Args:
            value (str): New template containing allowed placeholders.

        """
        self._name_template = value

    @property
    def saved_keys(self) -> list[int | str]:
        """
        List cache keys for which checkpoints exist.

        Returns:
            list[int | str]: Keys in sorted order.

        """
        if self._mode == "memory":
            return sorted(self._memory_states.keys(), key=str)
        return sorted(self._disk_paths.keys(), key=str)

    # ================================================
    # State Recording
    # ================================================
    def should_save(self, hook: str) -> bool:
        """
        Check if a save should occur for this hook.

        Checks whether `hook` is in `save_on` and whether the
        `every_n` condition is satisfied.

        Args:
            hook (str): The lifecycle hook name.

        Returns:
            bool: True if a checkpoint should be saved.

        """
        if hook not in self._save_on:
            return False
        count = self._increment_hook(hook)
        return count % self._every_n == 0

    def record_memory(self, key: int | str, state: dict[str, Any]) -> None:
        """
        Record an in-memory state snapshot.

        Args:
            key (int | str):
                Identifier for this checkpoint (e.g. epoch index or phase label).
            state (dict[str, Any]):
                The state dictionary to store.

        """
        self._memory_states[key] = state
        self._track_key(key)

    def record_disk(self, key: int | str, path: Path) -> None:
        """
        Record a disk checkpoint path.

        Args:
            key (int | str):
                Identifier for this checkpoint.
            path (Path):
                The saved checkpoint file path.

        """
        self._disk_paths[key] = path
        self._track_key(key)

    # ================================================
    # Query
    # ================================================
    def has_key(self, key: int | str) -> bool:
        """
        Check whether a checkpoint exists for the given key.

        Args:
            key (int | str): Identifier to query.

        Returns:
            bool: True if the checkpoint is available.

        """
        if self._mode == "memory":
            return key in self._memory_states
        return key in self._disk_paths

    def get_state(self, key: int | str) -> dict[str, Any] | None:
        """
        Return the in-memory state dict for a given key.

        Only available in memory mode. Returns None if no state exists
        or if in disk mode.

        Args:
            key (int | str): Identifier previously supplied to :meth:`record_memory`.

        Returns:
            dict[str, Any] | None: Stored state if available.

        """
        if self._mode != "memory":
            return None
        return self._memory_states.get(key)

    def get_path(self, key: int | str) -> Path | None:
        """
        Return the disk checkpoint path for a given key.

        Only available in disk mode. Returns None if no path exists
        or if in memory mode.

        Args:
            key (int | str): Identifier previously supplied to :meth:`record_disk`.

        Returns:
            Path | None: Checkpoint path if it exists on disk.

        """
        if self._mode != "disk":
            return None
        return self._disk_paths.get(key)

    # ================================================
    # Lifecycle
    # ================================================
    def reset(self) -> None:
        """
        Clear all tracked states and counters.

        Called by the parent at the start of a new phase or experiment run
        to ensure fresh checkpointing state.

        """
        self._memory_states.clear()
        self._disk_paths.clear()
        self._key_order.clear()
        self._hook_counts.clear()

    # ================================================
    # Utilities
    # ================================================
    @staticmethod
    def validate_placeholders(
        template: str,
        allowed: frozenset[str],
        context_name: str,
    ) -> None:
        """
        Validate that a name template only uses allowed placeholders.

        Args:
            template (str): The name template string to validate.
            allowed (frozenset[str]): Set of allowed placeholder names.
            context_name (str): Name of the context (for error messages).

        Raises:
            ValueError: If the template contains disallowed placeholders.

        """
        used = set(re.findall(r"\{(\w+)\}", template))
        invalid = used - allowed
        if invalid:
            msg = (
                f"Invalid placeholder(s) in name_template for {context_name}: "
                f"{sorted(invalid)}. Allowed: {sorted(allowed)}."
            )
            raise ValueError(msg)

    def format_name(self, **kwargs: Any) -> str:
        """
        Format the name template with provided key-value pairs.

        Args:
            **kwargs (Any): Values to substitute into the name template.
                Common keys: `label`, `key`, `phase`, `epoch`, `batch`.

        Returns:
            str: The formatted checkpoint name.

        """
        return self._name_template.format(**kwargs)

    # ================================================
    # Internal
    # ================================================
    def _increment_hook(self, hook_name: str) -> int:
        """Increment and return the 1-based count for a given hook."""
        self._hook_counts[hook_name] = self._hook_counts.get(hook_name, 0) + 1
        return self._hook_counts[hook_name]

    def _track_key(self, key: int | str) -> None:
        """Track key for ordering and enforce max_to_keep."""
        if key not in self._key_order:
            self._key_order.append(key)

        # Enforce max_to_keep
        if self._max_to_keep is not None:
            while len(self._key_order) > self._max_to_keep:
                oldest = self._key_order.pop(0)
                self._memory_states.pop(oldest, None)
                self._disk_paths.pop(oldest, None)

    # ================================================
    # Configurable
    # ================================================
    def get_config(self) -> dict[str, Any]:
        """Return configuration details required to reconstruct this object."""
        return {
            "mode": self._mode,
            "save_on": self._save_on,
            "every_n": self._every_n,
            "directory": (
                str(self._directory) if self._directory is not None else None
            ),
            "name_template": self._name_template,
            "max_to_keep": self._max_to_keep,
            "overwrite": self._overwrite,
        }

    @classmethod
    def from_config(cls, config: dict) -> Checkpointing:
        """
        Construct from config data.

        Args:
            config (dict): Serialized configuration produced by :meth:`get_config`.

        Returns:
            Checkpointing: Reconstructed checkpointing helper.

        """
        directory = config.get("directory")
        return cls(
            mode=config.get("mode", "memory"),
            save_on=config.get("save_on", "epoch_end"),
            every_n=config.get("every_n", 1),
            directory=Path(directory) if directory is not None else None,
            name_template=config.get("name_template"),
            max_to_keep=config.get("max_to_keep"),
            overwrite=config.get("overwrite", True),
        )
