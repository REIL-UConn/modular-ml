"""Storage for callback results recorded during phase execution."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

    from modularml.core.experiment.callbacks.callback import CallbackResult


class CallbackStore:
    """
    An ordered store of ``CallbackResult`` objects recorded during phase execution.

    Results are kept in memory by default. Pass a ``location`` directory to
    serialize each result to disk as it is appended. Each result is stored as
    ``{location}/{callback_label}/{global_seq}.pkl`` where ``global_seq`` is a
    monotonically increasing integer across all callback labels, preserving
    insertion order for reconstruction.

    All access (iteration) transparently loads from disk as needed.
    """

    def __init__(self, location: Path | None = None) -> None:
        self._location: Path | None = location
        self._items: list[CallbackResult | Path] = []

    # ================================================
    # Writing
    # ================================================
    def append(self, cb_res: CallbackResult) -> None:
        """
        Record a callback result.

        Args:
            cb_res (CallbackResult): The result to store.

        """
        if self._location is not None:
            stored: CallbackResult | Path = self._save_to_disk(cb_res)
        else:
            stored = cb_res
        self._items.append(stored)

    def _save_to_disk(self, cb_res: CallbackResult) -> Path:
        # Global insertion seq ensures order is recoverable on reload
        seq = len(self._items)
        label_dir = Path(self._location) / cb_res.callback_label
        label_dir.mkdir(parents=True, exist_ok=True)
        filepath = label_dir / f"{seq}.pkl"
        with filepath.open("wb") as f:
            pickle.dump(cb_res, f, protocol=pickle.HIGHEST_PROTOCOL)
        return filepath

    def clear(self) -> None:
        """
        Remove all entries from this store.

        On-disk files are not deleted; only the in-memory index is cleared.
        """
        self._items.clear()

    # ================================================
    # Reading
    # ================================================
    @staticmethod
    def _load(item: CallbackResult | Path) -> CallbackResult:
        if isinstance(item, Path):
            with item.open("rb") as f:
                return pickle.load(f)
        return item

    # ================================================
    # Reconstruction
    # ================================================
    @classmethod
    def from_directory(cls, location: Path) -> CallbackStore:
        """
        Reconstruct a store index from existing pickle files under ``location``.

        Globs for ``{location}/*/*.pkl`` and sorts by integer stem to recover
        insertion order across all callback labels.

        Args:
            location (Path): Root callbacks directory containing label subdirectories.

        Returns:
            CallbackStore: Store with entries populated as disk references.

        """
        store = cls(location=location)
        files = sorted(
            Path(location).glob("*/*.pkl"),
            key=lambda p: int(p.stem),
        )
        store._items = files
        return store

    # ================================================
    # Iteration / Sizing
    # ================================================
    def __iter__(self) -> Iterator[CallbackResult]:
        """Yield live ``CallbackResult`` objects in insertion order."""
        for item in self._items:
            yield self._load(item)

    def __len__(self) -> int:
        return len(self._items)

    def __bool__(self) -> bool:
        return bool(self._items)

    # ================================================
    # Representation
    # ================================================
    def __repr__(self) -> str:
        loc = str(self._location) if self._location is not None else "in-memory"
        return f"CallbackStore(len={len(self)}, location={loc!r})"
