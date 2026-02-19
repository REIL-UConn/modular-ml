"""Abstract base utilities for dataset splitting strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

from modularml.core.io.protocols import Configurable, Stateful

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from modularml.core.data.featureset_view import FeatureSetView


class BaseSplitter(Configurable, Stateful, ABC):
    """
    Abstract base class for algorithms that derive :class:`FeatureSetView` objects.

    Description:
        Defines the shared API for all splitters used to partition FeatureSets
        or :class:`FeatureSetView` instances into multiple subsets (for example,
        train/validation/test). Splitters may be invoked directly via
        :meth:`split` or through FeatureSet convenience wrappers such as
        :meth:`SplitMixin.split_random`.

        Subclasses must implement :meth:`split`, :meth:`get_config`, and
        :meth:`from_config`. The base class provides helpers for returning
        consistent types, managing serialization, and tracking random state.

    """

    # ================================================
    # Core abstract methods
    # ================================================
    @abstractmethod
    def split(
        self,
        view: FeatureSetView,
        *,
        return_views: bool = True,
    ) -> Mapping[str, FeatureSetView] | Mapping[str, Sequence[int]]:
        """
        Split a :class:`FeatureSetView` into multiple subsets.

        Description:
            Generates one or more subsets (as either index arrays or FeatureSetViews)
            derived from the provided FeatureSetView. Subclasses implement the
            internal logic that determines which sample indices belong to each subset.

            All indices returned by subclasses must be **relative** to the input
            `FeatureSetView` (i.e., ranging from 0 to len(view)-1).

        Args:
            view (FeatureSetView):
                The input view to partition.
            return_views (bool, optional):
                If True, return a mapping of labels to FeatureSetViews.
                If False, return a mapping of labels to relative index sequences.
                Defaults to True.

        Returns:
            Mapping[str, FeatureSetView] | Mapping[str, Sequence[int]]:
                A mapping from subset label (e.g., "train", "val", "test") to
                either a FeatureSetView or an array/list of integer indices.

        """

    # ================================================
    # Convenience methods for subclasses
    # ================================================
    def _return_splits(
        self,
        view: FeatureSetView,
        split_indices: Mapping[str, Sequence[int]],
        *,
        return_views: bool,
    ) -> Mapping[str, FeatureSetView] | Mapping[str, Sequence[int]]:
        """
        Convert relative index arrays into the desired output format.

        Description:
            Converts a dictionary of relative index arrays into either
            :class:`FeatureSetView` objects or raw index arrays depending on
            `return_views`.

            Subclasses should always produce **relative indices** for the
            provided view. This method ensures they are safely converted
            into new FeatureSetViews via `view.select_rows()`.

        Args:
            view (FeatureSetView):
                The source view from which new views will be derived.
            split_indices (Mapping[str, Sequence[int]]):
                Mapping from label to relative row indices (0 to len(view)-1).
            return_views (bool):
                Whether to return FeatureSetViews or index arrays.

        Returns:
            Mapping[str, FeatureSetView] | Mapping[str, Sequence[int]]:
                Either derived views or the raw relative index mapping.

        Raises:
            IndexError: If any relative index falls outside `[0, len(view) - 1]`.

        """
        # Validate indices are within the current view
        n = len(view)
        for lbl, idx in split_indices.items():
            if any(i < 0 or i >= n for i in idx):
                msg = (
                    f"Splitter `{type(self).__name__}` produced out-of-range indices "
                    f"for subset '{lbl}'. Expected values in [0, {n - 1}]."
                )
                raise IndexError(msg)

        # Return indices directly
        if not return_views:
            return split_indices

        # FeatureSetView.select_rows constructs new views from itself
        return {
            label: view.take(rel_indices=idxs, label=label)
            for label, idxs in split_indices.items()
        }

    # ================================================
    # Configurable
    # ================================================
    def get_config(self) -> dict[str, Any]:
        """
        Return configuration required to reconstruct this splitter.

        Returns:
            dict[str, Any]: Serializable splitter configuration.

        Raises:
            NotImplementedError: Must be implemented by subclasses.

        """
        raise NotImplementedError

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> BaseSplitter:
        """
        Construct a splitter from configuration.

        Args:
            config (dict[str, Any]): Serialized splitter configuration.

        Returns:
            BaseSplitter: Unfitted splitter instance.

        Raises:
            KeyError: If `splitter_name` is missing from `config`.

        """
        from modularml.splitters import splitter_registry

        if "splitter_name" not in config:
            msg = (
                "Splitter config must store 'splitter_name' if using "
                "BaseSplitter to instantiate."
            )
            raise KeyError(msg)
        splitter_cls: BaseSplitter = splitter_registry.get(config["splitter_name"])
        return splitter_cls.from_config(config)

    # ================================================
    # Stateful
    # ================================================
    def get_state(self) -> dict[str, Any]:
        """
        Return runtime state (for example, RNG) of the splitter.

        Returns:
            dict[str, Any]: Splitter state dictionary.

        Raises:
            NotImplementedError: Must be implemented by subclasses.

        """
        raise NotImplementedError

    def set_state(self, state: dict[str, Any]) -> None:
        """
        Restore runtime state previously produced by :meth:`get_state`.

        Args:
            state (dict[str, Any]): State dictionary produced by :meth:`get_state`.

        Raises:
            NotImplementedError: Must be implemented by subclasses.

        """
        raise NotImplementedError

    # ================================================
    # Serialization
    # ================================================
    def save(self, filepath: Path, *, overwrite: bool = False) -> Path:
        """
        Serialize this splitter to the specified filepath.

        Args:
            filepath (Path):
                File location to save to (suffix may be adjusted to match the
                ModularML schema).
            overwrite (bool):
                Whether to overwrite an existing file.

        Returns:
            Path: Actual filepath written by the serializer.

        """
        from modularml.core.io.serialization_policy import SerializationPolicy
        from modularml.core.io.serializer import serializer

        return serializer.save(
            self,
            filepath,
            policy=SerializationPolicy.REGISTERED,
            overwrite=overwrite,
        )

    @classmethod
    def load(cls, filepath: Path, *, allow_packaged_code: bool = False) -> BaseSplitter:
        """
        Load a splitter from disk.

        Args:
            filepath (Path): Location of a previously saved splitter artifact.
            allow_packaged_code (bool): Whether packaged code execution is allowed.

        Returns:
            BaseSplitter: Reloaded splitter instance.

        """
        from modularml.core.io.serializer import _enforce_file_suffix, serializer

        # Append proper sufficx only if no suffix is given
        if Path(filepath).suffix == "":
            filepath = _enforce_file_suffix(path=filepath, cls=cls)

        return serializer.load(filepath, allow_packaged_code=allow_packaged_code)
