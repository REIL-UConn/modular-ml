"""Condition-based sample partitioning utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from modularml.core.splitting.base_splitter import BaseSplitter
from modularml.predicates import Lambda, Predicate, _extract_lambda_source

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from modularml.core.data.featureset_view import FeatureSetView


# ================================================
# Condition serialization helpers
# ================================================
def _serialize_condition(cond: Any) -> Any:
    """Convert a single condition value to a JSON-serializable form."""
    if isinstance(cond, Predicate):
        return {"__predicate__": cond.to_dict()}
    if callable(cond):
        # Raw lambda or named function — extract source automatically
        source = _extract_lambda_source(cond)
        return {"__predicate__": Lambda(source).to_dict()}
    # Literal value or list — already JSON-safe
    return cond


def _deserialize_condition(val: Any) -> Any:
    """Restore a condition from its serialized form."""
    if isinstance(val, dict) and "__predicate__" in val:
        return Predicate.from_dict(val["__predicate__"])
    return val


class ConditionSplitter(BaseSplitter):
    """
    Split a :class:`FeatureSetView` using user-defined logical conditions.

    Description:
        Each subset is defined by a mapping `{field_name: condition}`, where each
        condition may be a literal value (exact match), a sequence of allowed values,
        or a predicate callable returning a boolean. Field names can refer to features,
        targets, or tags.

        If a sample satisfies multiple subset conditions, a warning is emitted and the
        sample will appear in every matching subset.

    Attributes:
        conditions (dict[str, Mapping[str, Any | Sequence | Callable]]):
            Subset labels to condition dictionaries.

    """

    def __init__(
        self,
        **conditions: Mapping[str, Mapping[str, Any | Sequence | Callable]],
    ):
        """
        Initialize the condition-based splitter.

        Args:
            **conditions:
                Keyword arguments mapping subset labels to conditiondictionaries.
                Each dictionary entry maps a FeatureSet key to a literal value, an
                iterable of allowed values, or a predicate callable (`f(x) -> bool`).

        """
        self.conditions = conditions

    def split(
        self,
        view: FeatureSetView,
        *,
        return_views: bool = True,
    ) -> Mapping[str, FeatureSetView] | Mapping[str, Sequence[int]]:
        """
        Split a :class:`FeatureSetView` using the configured conditions.

        Args:
            view (FeatureSetView):
                Input view to partition.
            return_views (bool):
                If True, return :class:`FeatureSetView` objects; otherwise
                return relative index arrays.

        Returns:
            Mapping[str, FeatureSetView] | Mapping[str, Sequence[int]]:
                Subset label mapped to either views or relative indices.

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

        Description:
            Callable conditions (lambdas, :class:`Predicate` instances) are
            converted to JSON-serializable dicts so that the containing
            :class:`FeatureSet` can be saved without errors.

        Returns:
            dict[str, Any]: Serializable splitter configuration.

        """
        serializable_conditions = {
            subset_label: {
                field: _serialize_condition(cond) for field, cond in sub_conds.items()
            }
            for subset_label, sub_conds in self.conditions.items()
        }
        return {
            "splitter_name": "ConditionSplitter",
            "conditions": serializable_conditions,
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> ConditionSplitter:
        """
        Construct a splitter from configuration.

        Args:
            config (dict[str, Any]): Serialized splitter configuration.

        Returns:
            ConditionSplitter: Unbound splitter instance.

        """
        raw_conditions: dict[str, Any] = config["conditions"]
        conditions = {
            subset_label: {
                field: _deserialize_condition(val) for field, val in sub_conds.items()
            }
            for subset_label, sub_conds in raw_conditions.items()
        }
        return cls(**conditions)
