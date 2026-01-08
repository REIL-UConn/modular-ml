from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from modularml.core.splitting.base_splitter import BaseSplitter
from modularml.utils.errors.exceptions import SplitOverlapWarning
from modularml.utils.logging.warnings import warn

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from modularml.core.data.featureset_view import FeatureSetView


class ConditionSplitter(BaseSplitter):
    """
    Splits a FeatureSetView into subsets based on user-defined logical conditions.

    Description:
        Each subset is defined by one or more filtering rules expressed as
        a mapping `{field_name: condition}`, where `condition` can be:
          - A literal value (exact match)
          - A sequence of allowed values
          - A callable (predicate) that returns True/False

        Fields may reference feature, target, or tag keys from the FeatureSet schema.

    Example:
        ```python
        splitter = ConditionSplitter(
            low_temp={"temperature": lambda x: x < 20},
            high_temp={"temperature": lambda x: x >= 20},
            cell_5={"cell_id": 5},
        )
        splits = splitter.split(fs_view, return_views=True)
        ```

        If a sample satisfies multiple subset conditions, a warning is raised and
        the sample will appear in multiple subsets.

    """

    def __init__(self, **conditions: Mapping[str, Mapping[str, Any | Sequence | Callable]]):
        """
        Initialize a ConditionSplitter.

        Args:
            **conditions:
                Mapping of subset labels to condition dictionaries.
                Each condition dictionary maps a key (feature/target/tag name)
                to either:
                  - A literal value (equality)
                  - A list/tuple/set/array of allowed values
                  - A callable predicate `f(x) -> bool`

        """
        self.conditions = conditions

    def split(
        self,
        view: FeatureSetView,
        *,
        return_views: bool = True,
    ) -> Mapping[str, FeatureSetView] | Mapping[str, Sequence[int]]:
        """
        Split a FeatureSetView into subsets based on user-defined conditions.

        Args:
            view (FeatureSetView):
                The input view to partition.
            return_views (bool, optional):
                If True, returns a mapping of subset labels to FeatureSetViews. \
                If False, returns relative index arrays. Defaults to True.

        Returns:
            Mapping[str, FeatureSetView] | Mapping[str, Sequence[int]]:
                A mapping of subset label to either FeatureSetViews or index arrays.

        """
        split_indices: dict[str, np.ndarray] = {}
        sample_to_subsets: dict[int, list[str]] = {}
        for sub_label, sub_conds in self.conditions.items():
            # Filter the source FeatureSet directly
            filt_view = view.filter(label=sub_label, conditions=sub_conds)

            # Map absolute indices of filtered view to relative indices of given view
            abs_selected = set(filt_view.indices.tolist())
            rel_selected = np.array(
                [i for i, abs_i in enumerate(view.indices) if abs_i in abs_selected],
                dtype=int,
            )

            # Record split indices for this subset
            split_indices[sub_label] = rel_selected

            # Record which subset each sample index appears in (to check overlap)
            for idx in rel_selected:
                sample_to_subsets.setdefault(idx, []).append(sub_label)

        # Warn if any sample appears in more than one subset
        overlapping = {i: subsets for i, subsets in sample_to_subsets.items() if len(subsets) > 1}
        if overlapping:
            examples_str = ", ".join(
                f"Sample {k} is in {v}" for k, v in list(overlapping.items())[: min(2, len(overlapping))]
            )
            msg = (
                f"\n{len(overlapping)} samples were assigned to multiple subsets. "
                f"Overlap may affect downstream assumptions.\n"
                f"Examples: {examples_str} ..."
            )
            warn(msg, category=SplitOverlapWarning, stacklevel=2)

        return self._return_splits(
            view=view,
            split_indices=split_indices,
            return_views=return_views,
        )

    # ================================================
    # Configurable
    # ================================================
    def get_config(self) -> dict[str, Any]:
        """
        Return configuration required to reconstruct this splitter.

        Returns:
            dict[str, Any]: Splitter configuration.

        """
        return {
            "splitter_name": "ConditionSplitter",
            "conditions": self.conditions,
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> ConditionSplitter:
        """
        Construct a Splitter from configuration.

        Args:
            config (dict[str, Any]): Splitter configuration.

        Returns:
            BaseSplitter: Unfitted splitter instance.

        """
        return cls(conditions=config["conditions"])
