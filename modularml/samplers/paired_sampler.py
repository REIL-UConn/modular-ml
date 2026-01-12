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

    This sampler is a convenience wrapper around `NSampler` for the
    common two-way pairing case. It accepts a single mapping of
    column keys to SimilarityCondition objects and internally delegates
    to `NSampler` using a single role named `"pair"`. The resulting
    batches contain aligned indices under the roles `"anchor"` & `"pair"`
    where each anchor is matched with one or more candidate samples
    that satisfy the provided similarity conditions.
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
            `PairedSampler` selects, for each anchor sample, a set of
            candidate pair samples that satisfy the provided similarity
            conditions. All keys in `conditions` are resolved to concrete
            (domain, key, variant) columns using `DataReference` and
            evaluated using their associated SimilarityCondition objects.

            The sampler internally invokes `NSampler` with a single role
            named `"pair"`, meaning each batch will contain:
                - aligned anchor indices under `"anchor"`
                - aligned partner indices under `"pair"`
                - per-pair similarity weights

            This sampler is intended for contrastive, metric-learning, or
            supervised-pairing workflows where only two roles are needed.


        Args:
            conditions (dict[str, SimilarityCondition]):
                Mapping from user-defined column specifiers to similarity
                conditions. Each key is resolved using
                :meth:`DataReference.from_string` with the bound FeatureSet label,
                and must ultimately refer to a concrete (domain, key, variant).
                Example keys:
                    - "SOH_PCT"
                    - "features.voltage.raw"
                    - "targets.health_label.transformed"

            batch_size (int, optional):
                Number of (anchor, pair) pairs per batch. Defaults to 1.

            shuffle (bool, optional):
                If True, the final list of (anchor, pair) pairs is shuffled before
                batching. Stochastic selection within match categories is also
                controlled by this flag. Defaults to False.

            max_pairs_per_anchor (int | None, optional):
                Maximum number of pairs to generate per anchor:
                - If None: all valid pairs are used.
                - If int: at most `max_pairs_per_anchor` pairs per anchor are
                    created, prioritizing full → partial → non-matches.
                Defaults to 3.

            choose_best_only (bool):
                Instead of generating multiple pairs, select only the \
                highest-score pair per anchor.

            group_by (list[str], optional):
                FeatureSet key(s) defining grouping behavior.
                Only one grouping strategy can be active at a time.

            group_by_role (str, optional):
                If `group_by=True`, the role on which to draw data for grouping
                must be specified. Defaults to `"anchor"`.

            stratify_by (list[str], optional):
                FeatureSet key(s) defining strata for stratified sampling.
                Conflicts with `group_by`.

            stratify_by_role (str, optional):
                If `stratify_by=True`, the role on which to draw data for stratification
                must be specified. Defaults to `"anchor"`.

            strict_stratification (bool, optional):
                See description above.

            drop_last (bool, optional):
                If True, the final incomplete batch (with fewer than `batch_size`
                pairs) is dropped. Defaults to False.

            seed (int | None, optional):
                Random seed for reproducible shuffling and stochastic selection.

            show_progress (bool, optional):
                Whether to show a progress bar during the batch building process.
                Defaults to True.

            source (FeatureSet | FeatureSetView):
                The source data from samples are drawn. Note that batches are not
                constructed until `materialize_batches` is called.

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
        if self.is_bound:
            return f"PairedSampler(n_batches={self.num_batches}, batch_size={self.batcher.batch_size})"
        return f"PairedSampler(batch_size={self.batcher.batch_size})"

    # ================================================
    # Configurable
    # ================================================
    def get_config(self) -> dict[str, Any]:
        """
        Return configuration required to reconstruct this sampler.

        Description:
            This *does not* restore the source, only the sampler configurtion.

        Returns:
            dict[str, Any]: Sampler configuration.

        """
        cfg = super().get_config()
        cfg.update({"sampler_name": "PairedSampler"})
        return cfg

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> PairedSampler:
        """
        Construct a sampler from configuration.

        Args:
            config (dict[str, Any]): Sampler configuration.

        Returns:
            PairedSampler: Unfitted sampler instance.

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
