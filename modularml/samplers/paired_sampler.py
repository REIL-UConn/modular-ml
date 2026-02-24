"""Convenience wrappers for similarity-based pair samplers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from modularml.core.data.schema_constants import ROLE_ANCHOR, ROLE_PAIR
from modularml.samplers.n_sampler import NSampler

if TYPE_CHECKING:
    from modularml.core.data.featureset import FeatureSet
    from modularml.core.data.featureset_view import FeatureSetView
    from modularml.core.sampling.similiarity_condition import SimilarityCondition


class PairedSampler(NSampler):
    """
    Sampler for generating anchor-pair matches using similarity conditions.

    Description:
        A thin wrapper around :class:`NSampler` configured with a single role named `pair`.
        Each batch therefore contains aligned anchor indices plus candidate indices stored under
        the `pair` role together with similarity weights.

    """

    def __init__(
        self,
        conditions: dict[str, SimilarityCondition],
        *,
        batch_size: int = 1,
        shuffle: bool = False,
        max_pairs_per_anchor: int | None = 3,
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
        Initialize a similarity-based anchor-pair sampler.

        Description:
            :class:`PairedSampler` resolves every column specifier in `conditions` using
            :meth:`FeatureSetColumnReference.from_string`, evaluates matches via the associated
            :class:`SimilarityCondition`, and delegates batching to :class:`NSampler` with a
            single role named `pair`.

        Args:
            conditions (dict[str, SimilarityCondition]):
                Mapping from column identifiers to similarity rules used for the `pair` role.

            batch_size (int):
                Number of (anchor, pair) tuples per batch.

            shuffle (bool):
                Whether to shuffle aligned pairs prior to batching.

            max_pairs_per_anchor (int | None):
                Maximum number of partners per anchor; `None` keeps all matches.

            choose_best_only (bool):
                Select only the top-scoring partner(s) per anchor.

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
                Optional :class:`FeatureSet` or :class:`FeatureSetView` to bind immediately.

        """
        super().__init__(
            condition_mapping={ROLE_PAIR: conditions},
            batch_size=batch_size,
            shuffle=shuffle,
            max_samples_per_anchor=max_pairs_per_anchor,
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
            return f"PairedSampler(n_batches={self.num_batches}, batch_size={self.batcher.batch_size})"
        return f"PairedSampler(batch_size={self.batcher.batch_size})"

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
        cfg.update({"sampler_name": "PairedSampler"})
        return cfg

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> PairedSampler:
        """
        Construct a sampler from configuration.

        Args:
            config (dict[str, Any]): Serialized sampler configuration.

        Returns:
            PairedSampler: Unbound sampler instance.

        Raises:
            ValueError: If the configuration was not produced by :meth:`get_config`.

        """
        if ("sampler_name" not in config) or (
            config["sampler_name"] != "PairedSampler"
        ):
            raise ValueError("Invalid config for PairedSampler.")

        return cls(
            conditions=config["condition_mapping"][ROLE_PAIR],
            batch_size=config["batch_size"],
            shuffle=config["shuffle"],
            max_pairs_per_anchor=config["max_samples_per_anchor"],
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
