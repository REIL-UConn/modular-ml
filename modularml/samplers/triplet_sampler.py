"""Convenience wrappers for triplet sampling."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from modularml.core.data.schema_constants import (
    ROLE_ANCHOR,
    ROLE_NEGATIVE,
    ROLE_POSITIVE,
)
from modularml.samplers.n_sampler import NSampler

if TYPE_CHECKING:
    from modularml.core.data.featureset import FeatureSet
    from modularml.core.data.featureset_view import FeatureSetView
    from modularml.core.sampling.similiarity_condition import SimilarityCondition


class TripletSampler(NSampler):
    """
    Sampler for generating anchor-positive-negative triplets.

    Description:
        A specialized :class:`NSampler` wrapper that declares `positive` and `negative`
        roles so that each batch contains aligned anchor, positive, and negative indices
        together with similarity scores.

    """

    def __init__(
        self,
        pos_conditions: dict[str, SimilarityCondition],
        neg_conditions: dict[str, SimilarityCondition],
        *,
        batch_size: int = 1,
        shuffle: bool = False,
        max_samples_per_anchor: int | None = 3,
        choose_best_only: bool = False,
        group_by: list[str] | None = None,
        group_by_role: str = ROLE_ANCHOR,
        stratify_by: list[str] | None = None,
        stratify_by_role: str = ROLE_ANCHOR,
        strict_stratification: bool = True,
        drop_last: bool = False,
        seed: int | None = None,
        show_progress: bool = True,
        source: FeatureSet | FeatureSetView | None = None,
    ):
        """
        Initialize a similarity-based sampler for triplets.

        Description:
            :class:`TripletSampler` resolves the column identifiers in `pos_conditions` and
            `neg_conditions` via :meth:`FeatureSetColumnReference.from_string`, then
            configures :class:`NSampler` with `positive` and `negative` roles. Only
            anchors that have valid matches for both roles are emitted.

        Args:
            pos_conditions (dict[str, SimilarityCondition]):
                Similarity rules for selecting positive samples.

            neg_conditions (dict[str, SimilarityCondition]):
                Similarity rules for selecting negative samples.

            batch_size (int):
                Number of triplets per batch.

            shuffle (bool):
                Whether to shuffle aligned triplets before batching.

            max_samples_per_anchor (int | None):
                Maximum positive/negative matches per anchor; `None` keeps all matches.

            choose_best_only (bool):
                Select only top-scoring matches per role.

            group_by (list[str] | None):
                Optional FeatureSet keys for grouping (mutually exclusive with `stratify_by`).

            group_by_role (str):
                Role used for grouping; defaults to :attr:`ROLE_ANCHOR`.

            stratify_by (list[str] | None):
                Optional keys for stratified sampling.

            stratify_by_role (str):
                Role used for stratification; defaults to :attr:`ROLE_ANCHOR`.

            strict_stratification (bool):
                Whether batching stops when any stratum is exhausted.

            drop_last (bool):
                Drop the final incomplete batch.

            seed (int | None):
                Random seed for reproducible shuffling.

            show_progress (bool):
                Whether to show progress updates.

            source (FeatureSet | FeatureSetView | None):
                Optional :class:`FeatureSet` or :class:`FeatureSetView` to bind
                immediately.

        """
        super().__init__(
            condition_mapping={
                ROLE_POSITIVE: pos_conditions,
                ROLE_NEGATIVE: neg_conditions,
            },
            batch_size=batch_size,
            shuffle=shuffle,
            max_samples_per_anchor=max_samples_per_anchor,
            choose_best_only=choose_best_only,
            group_by=group_by,
            group_by_role=group_by_role,
            stratify_by=stratify_by,
            stratify_by_role=stratify_by_role,
            strict_stratification=strict_stratification,
            drop_last=drop_last,
            seed=seed,
            show_progress=show_progress,
            source=source,
        )

    def __repr__(self):
        """Return a concise string describing sampler state."""
        if self.is_bound:
            return f"TripletSampler(n_batches={self.num_batches}, batch_size={self.batcher.batch_size})"
        return f"TripletSampler(batch_size={self.batcher.batch_size})"

    # ================================================
    # Configurable
    # ================================================
    def get_config(self) -> dict[str, Any]:
        """
        Return configuration required to reconstruct this sampler.

        Returns:
            dict[str, Any]: Serializable sampler configuration (sources excluded).

        """
        cfg = super().get_config()
        cfg.update({"sampler_name": "TripletSampler"})
        return cfg

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> TripletSampler:
        """
        Construct a sampler from configuration.

        Args:
            config (dict[str, Any]): Serialized sampler configuration.

        Returns:
            TripletSampler: Unbound sampler instance.

        Raises:
            ValueError: If the configuration was not produced by :meth:`get_config`.

        """
        if ("sampler_name" not in config) or (
            config["sampler_name"] != "TripletSampler"
        ):
            raise ValueError("Invalid config for TripletSampler.")

        return cls(
            pos_conditions=config["condition_mapping"][ROLE_POSITIVE],
            neg_conditions=config["condition_mapping"][ROLE_NEGATIVE],
            batch_size=config["batch_size"],
            shuffle=config["shuffle"],
            max_samples_per_anchor=config["max_samples_per_anchor"],
            choose_best_only=config["choose_best_only"],
            group_by=config["group_by"],
            group_by_role=config["group_by_role"],
            stratify_by=config["stratify_by"],
            stratify_by_role=config["stratify_by_role"],
            strict_stratification=config["strict_stratification"],
            drop_last=config["drop_last"],
            seed=config["seed"],
            show_progress=config["show_progress"],
        )
