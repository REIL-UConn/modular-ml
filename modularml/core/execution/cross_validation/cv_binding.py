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
    Configuration for cross validation of a single FeatureSet.

    Description:
        Defines how a specified FeatureSet should participate in cross-validation.
        The head nodes of a ModelGraph are typically bound to a split or FeatureSet.
        This configuration defines how cross-validation folds should be mapped
        to those existing head node input bindings. For example, say we have a FeatureSet
        with existing splits: "my_training" and "my_validation". Since each fold of
        cross-validation produces two splits, we need to map the returned names to
        our existing split names. To do this, we would set `train_split_name='my_training'`
        and `val_split_name='my_validation'`. And that's it. Each fold of cross validation
        will update the expected split names with only data belonging to the current fold.

    Attributes:
        featureset: The FeatureSet (or its label/node_id) to create folds from.
        source_splits: List of existing split names to combine as the CV pool.
        group_by: Column name for group-based splitting (keeps groups together).
        stratify_by: Column name for stratified splitting (maintains distribution).

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
        Defines cross validation for a single FeatureSet.

        Args:
            fs (str | FeatureSet):
                FeatureSet to apply cross validation to. Can be the node ID, label, or
                the FeatureSet instance.

            source_splits (list[str]):
                Existing splits of `fs` to use for fold creation. For example, if
                `source_splits=['train', 'val']`, folds will be created by drawing
                samples for the merged collection of `'train'` and `'val'`.

            group_by (str | list[str] | None, optional):
                Optional grouping to be applied during fold generation. Can be one
                or more tag keys to group samples by before splitting. Mutually
                exclusive with `stratify_by`. Defaults to None.

            stratify_by (str | list[str] | None, optional):
                Optional stratification to be applied during fold generation. Can
                be one or more tag keys to define strata for splitting. Mutually
                exclusive with `group_by`. Defaults to None.

            train_split_name (str, optional):
                Each fold of cross validation produces two splits: "train" and "val".
                These can be mapped to the split label used in the cross validation
                template training phase. Defaults to "train".

            val_split_name (str, optional):
                Name to use for "val" split produced for each fold. Defaults to "val".

            val_size (float | None, optional):
                Optional size of "val" split of each fold. If None, the size will be
                set via the number of folds (e.g., 5 folds -> val_size=0.2).
                Defaults to None.

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
