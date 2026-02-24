"""Binding definitions for cross-validation FeatureSets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from modularml.core.data.featureset import FeatureSet
from modularml.core.experiment.experiment_context import ExperimentContext
from modularml.utils.data.formatting import ensure_list
from modularml.utils.logging.warnings import warn

if TYPE_CHECKING:
    from modularml.core.data.featureset_view import FeatureSetView


class CVBinding:
    """
    Configuration for cross-validation of a single FeatureSet.

    Description:
        Defines how a specified :class:`FeatureSet` should participate in
        cross-validation. The head nodes of a :class:`ModelGraph` are typically
        bound to a split or :class:`FeatureSet`. This configuration maps
        fold-specific outputs back onto those bindings (for example, mapping
        `train_split_name='my_training'`). Each fold then updates the expected
        split names with data belonging only to the current fold.

    Attributes:
        featureset (FeatureSet): The :class:`FeatureSet` to create folds from.
        source_splits (list[str]): Existing split names combined to form the pool.
        group_by (str | list[str] | None): Columns used for group-based splitting.
        stratify_by (str | list[str] | None): Columns used for stratified splitting.

    """

    def __init__(
        self,
        fs: str | FeatureSet,
        source_splits: list[str],
        *,
        group_by: str | list[str] | None = None,
        stratify_by: str | list[str] | None = None,
        train_split_name: str = "train",
        val_split_name: str = "val",
        val_size: float | None = None,
    ):
        """
        Configure cross-validation for a single FeatureSet.

        Args:
            fs (str | FeatureSet):
                :class:`FeatureSet` (or its node ID/label) to apply cross-validation to.
            source_splits (list[str]):
                Existing splits of `fs` to pool before folding. For example,
                `source_splits=['train', 'val']` merges both splits prior to sampling.
            group_by (str | list[str] | None, optional):
                Optional tag keys used for group-based splitting. Mutually exclusive
                with `stratify_by`. Defaults to None.
            stratify_by (str | list[str] | None, optional):
                Optional tag keys used for stratified splitting. Mutually exclusive
                with `group_by`. Defaults to None.
            train_split_name (str, optional):
                Split label that should receive each fold's training partition.
                Defaults to `train`.
            val_split_name (str, optional):
                Split label that should receive each fold's validation partition.
                Defaults to `val`.
            val_size (float | None, optional):
                Explicit validation proportion for each fold. If None, computed as
                `1 / n_folds`. Defaults to None.

        Raises:
            ValueError: If configuration references missing splits or invalid sizes.

        """
        # Store FeatureSet node ID
        exp_ctx = ExperimentContext.get_active()
        if not isinstance(fs, FeatureSet):
            fs = exp_ctx.get_node(val=fs, enforce_type="FeatureSet")
        self._fs_id = fs.node_id

        # Existing splits to draw CV samples from
        self.source_splits: list[str] = ensure_list(source_splits)
        missing_splits = [
            spl for spl in self.source_splits if spl not in fs.available_splits
        ]
        if missing_splits:
            msg = f"FeatureSet '{fs.label}' does not contain splits: {missing_splits}."
            raise ValueError(msg)

        # Splitting config
        if (group_by is not None) and (stratify_by is not None):
            msg = "Only one of `group_by` and `stratify_by` can be defined, not both."
            raise ValueError(msg)
        self.group_by = group_by
        self.stratify_by = stratify_by
        if (val_size is not None) and ((val_size >= 1) or (val_size <= 0)):
            raise ValueError("`val_size` must be between 0 and 1, exclusive.")
        self.val_size = val_size

        # Fold split naming
        self.train_split_name = train_split_name
        self.val_split_name = val_split_name
        if self.train_split_name not in fs.available_splits:
            msg = (
                f"`train_split_name` must correspond to an existing split name in "
                f"FeatureSet '{fs.label}'. Available: {fs.available_splits}."
            )
            raise ValueError(msg)
        if self.val_split_name not in fs.available_splits:
            msg = (
                f"`val_split_name` of '{self.val_split_name}' does not match an "
                f"existing split in FeatureSet '{fs.label}'. The validation split "
                "produced by each fold will not be used. "
            )
            warn(message=msg, stacklevel=2)


@dataclass
class _FoldViews:
    fold_idx: int
    train: FeatureSetView
    val: FeatureSetView
