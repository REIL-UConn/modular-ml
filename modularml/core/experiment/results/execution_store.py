"""Storage for forward-pass execution contexts recorded during phase execution."""

from __future__ import annotations

import copy
import pickle
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

    from modularml.core.data.execution_context import ExecutionContext


class ExecutionStore:
    """
    An ordered store of ``ExecutionContext`` objects recorded during phase execution.

    Contexts are kept in memory by default. Pass a ``location`` directory to
    serialize each context to disk as it is appended, freeing output tensors
    from RAM. All access (iteration, snapshots) transparently loads from disk
    as needed.
    """

    def __init__(self, location: Path | None = None) -> None:
        self._location: Path | None = location
        self._items: list[ExecutionContext | Path] = []

    # ================================================
    # Writing
    # ================================================
    def append(self, ctx: ExecutionContext) -> None:
        """
        Record an execution context.

        Args:
            ctx (ExecutionContext): The context to store.

        """
        if self._location is not None:
            stored: ExecutionContext | Path = self._save_to_disk(ctx)
        else:
            stored = ctx
        self._items.append(stored)

    def _save_to_disk(self, ctx: ExecutionContext) -> Path:
        # input_views holds lazy BatchView references that don't survive pickle;
        # strip them from the copy before serializing
        filename = f"{len(self._items)}.pkl"
        filepath = Path(self._location) / filename  # type: ignore[arg-type]
        filepath.parent.mkdir(parents=True, exist_ok=True)

        ctx_copy = copy.copy(ctx)
        ctx_copy.input_views = {}

        with filepath.open("wb") as f:
            pickle.dump(ctx_copy, f, protocol=pickle.HIGHEST_PROTOCOL)

        return filepath

    def clear(self) -> None:
        """
        Remove all entries from this store.

        On-disk files are not deleted; only the in-memory index is cleared.
        """
        self._items.clear()

    # ================================================
    # Snapshots / Reconstruction
    # ================================================
    def snapshot(self) -> list[ExecutionContext]:
        """
        Load all contexts into memory and return as a plain list.

        The returned list is independent of this store; a subsequent
        ``clear()`` has no effect on the snapshot.

        Returns:
            list[ExecutionContext]: All recorded contexts in insertion order.

        """
        return [self._load(item) for item in self._items]

    def to_memory(self) -> ExecutionStore:
        """
        Return a new in-memory store with all contexts loaded.

        Useful when you want a fully self-contained copy, for example before
        serializing the experiment or passing results to another process.

        Returns:
            ExecutionStore: New store with ``location=None`` and all contexts in memory.

        """
        store = ExecutionStore(location=None)
        store._items = [self._load(item) for item in self._items]
        return store

    @classmethod
    def from_list(
        cls,
        contexts: list[ExecutionContext],
        *,
        location: Path | None = None,
    ) -> ExecutionStore:
        """
        Build a store from an existing list of contexts.

        Args:
            contexts (list[ExecutionContext]):
                Contexts to populate the store with.
            location (Path | None, optional):
                Directory for on-disk storage. ``None`` keeps contexts in memory.
                Defaults to ``None``.

        Returns:
            ExecutionStore: A new store containing all provided contexts.

        """
        store = cls(location=location)
        for ctx in contexts:
            store.append(ctx)
        return store

    @classmethod
    def from_directory(cls, location: Path) -> ExecutionStore:
        """
        Reconstruct a store index from existing pickle files in ``location``.

        Args:
            location (Path): Directory containing serialized context files.

        Returns:
            ExecutionStore: Store with entries populated as disk references.

        """
        store = cls(location=location)
        files = sorted(
            Path(location).glob("*.pkl"),
            key=lambda p: int(p.stem),
        )
        store._items = files
        return store

    # ================================================
    # Reading
    # ================================================
    @staticmethod
    def _load(item: ExecutionContext | Path) -> ExecutionContext:
        if isinstance(item, Path):
            with item.open("rb") as f:
                return pickle.load(f)
        return item

    # ================================================
    # Iteration / Sizing
    # ================================================
    def __iter__(self) -> Iterator[ExecutionContext]:
        """Yield live ``ExecutionContext`` objects in insertion order."""
        for item in self._items:
            yield self._load(item)

    def __len__(self) -> int:
        return len(self._items)

    def __bool__(self) -> bool:
        return len(self._items) > 0

    # ================================================
    # Representation
    # ================================================
    def __repr__(self) -> str:
        loc = str(self._location) if self._location is not None else "in-memory"
        return f"ExecutionStore(len={len(self)}, location={loc!r})"
