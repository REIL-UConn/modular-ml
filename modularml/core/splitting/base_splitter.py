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
    Abstract base class for algorithms that derive FeatureSetViews from a FeatureSet.

    Description:
        Defines the core API for all splitters used to partition FeatureSets
        or FeatureSetViews into multiple subsets (e.g., train/val/test).
        Splitters may be called directly on FeatureSetViews, or indirectly
        through a FeatureSet convenience method (e.g., `fs.split_random()`).

        All subclasses must implement `split()`, `get_config()`, and
        `from_config()` methods. The base class handles consistent typing,
        return conventions, and metadata tracking for reproducibility.

    Usage:
        ```python
        splitter = RandomSplitter(ratios={"train": 0.8, "val": 0.2})
        splits = splitter.split(fs.as_view(), return_views=True)
        # or equivalently
        fs.split_random(ratios={"train": 0.8, "val": 0.2})
        ```

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
        Split a FeatureSetView into multiple subsets.

        Description:
            Generates one or more subsets (as either index arrays or FeatureSetViews) \
            derived from the provided FeatureSetView. Subclasses implement the \
            internal logic that determines which sample indices belong to each subset.

            All indices returned by subclasses must be **relative** to the input \
            `FeatureSetView` (i.e., ranging from 0 to len(view)-1).

        Args:
            view (FeatureSetView):
                The input view to partition.
            return_views (bool, optional):
                If True, return a mapping of labels to FeatureSetViews. \
                If False, return a mapping of labels to relative index sequences. \
                Defaults to True.

        Returns:
            Mapping[str, FeatureSetView] | Mapping[str, Sequence[int]]:
                A mapping from subset label (e.g., "train", "val", "test") to \
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
        Standardized return helper for splitter subclasses.

        Description:
            Converts a dictionary of relative index arrays into either
            FeatureSetViews or raw index arrays depending on `return_views`.

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
                Either derived FeatureSetViews or the raw relative index mapping.

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
            dict[str, Any]: Splitter configuration.

        """
        raise NotImplementedError

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> BaseSplitter:
        """
        Construct a Splitter from configuration.

        Args:
            config (dict[str, Any]): Splitter configuration.

        Returns:
            BaseSplitter: Unfitted splitter instance.

        """
        from modularml.splitters import splitter_registry

        if "splitter_name" not in config:
            msg = "Splitter config must store 'splitter_name' if using BaseSplitter to instantiate."
            raise KeyError(msg)
        splitter_cls: BaseSplitter = splitter_registry.get(config["splitter_name"])
        return splitter_cls.from_config(config)

    # ================================================
    # Stateful
    # ================================================
    def get_state(self) -> dict[str, Any]:
        """
        Return runtime (i.e. rng) state of the splitter.

        Returns:
            dict[str, Any]: Splitter state.

        """
        raise NotImplementedError

    def set_state(self, state: dict[str, Any]) -> None:
        """
        Restore runtime state of the splitter.

        Args:
            state (dict[str, Any]):
                State produced by get_state().

        """
        raise NotImplementedError

    # ================================================
    # Serialization
    # ================================================
    def save(self, filepath: Path, *, overwrite: bool = False) -> Path:
        """
        Serializes this Splitter to the specified filepath.

        Args:
            filepath (Path):
                File location to save to. Note that the suffix may be overwritten
                to enforce the ModularML file extension schema.
            overwrite (bool, optional):
                Whether to overwrite any existing file at the save location.
                Defaults to False.

        Returns:
            Path: The actual filepath to write the Splitter is saved.

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
        Load a Splitter from file.

        Args:
            filepath (Path):
                File location of a previously saved Splitter.
            allow_packaged_code : bool
                Whether bundled code execution is allowed.

        Returns:
            BaseSplitter: The reloaded sampler.

        """
        from modularml.core.io.serializer import _enforce_file_suffix, serializer

        # Append proper sufficx only if no suffix is given
        if Path(filepath).suffix == "":
            filepath = _enforce_file_suffix(path=filepath, cls=cls)

        return serializer.load(filepath, allow_packaged_code=allow_packaged_code)
