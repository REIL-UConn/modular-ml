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

    This sampler is a convenience wrapper around ``NSampler`` for the
    three-way sampling pattern commonly used in metric learning and
    contrastive representation learning. It accepts two sets of
    SimilarityCondition mappings—one for selecting positive samples
    (similar to the anchor) and one for selecting negative samples
    (dissimilar to the anchor). Internally, it defines two roles,
    `"positive"` & `"negative"`, and returns aligned triplets under
    the roles `"anchor"`, `"positive"`, and `"negative"`.
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
        Initialize a similarity-based sampler for anchor-positive-negative triplets.

        Description:
            `TripletSampler` constructs training triplets by selecting, for
            each anchor sample, a positive sample that satisfies the provided
            positive similarity conditions and a negative sample that satisfies
            the negative conditions. Column identifiers in
            `pos_conditions` and `neg_conditions` are resolved using
            `DataReference.from_string` and can refer to FeatureSet features,
            targets, or tags (e.g., "SOH_PCT", "features.voltage.raw").

            Under the hood, the sampler invokes `NSampler` with two roles:
                - "positive": candidates satisfying `pos_conditions`
                - "negative": candidates satisfying `neg_conditions`

            Only anchors that produce valid matches for *both* roles are kept.
            All triplet components are then sorted and aligned into consistent
            batches, each containing:
                - anchor indices
                - matched positive indices
                - matched negative indices
                - per-role similarity scores

        Args:
            pos_conditions (dict[str, SimilarityCondition]):
                Mapping from column specifiers to similarity rules for selecting
                positive samples. All keys must resolve to concrete
                (domain, key, variant) columns.

            neg_conditions (dict[str, SimilarityCondition]):
                Mapping from column specifiers to similarity rules for selecting
                negative samples. These often mirror the positive rules but with
                different tolerances or match modes.

            batch_size (int, optional):
                Number of triplets to return per batch. Defaults to 1.

            shuffle (bool, optional):
                Whether to shuffle triplets after alignment but before batching.
                Defaults to False.

            max_samples_per_anchor (int | None, optional):
                Maximum number of positive or negative samples selected for each
                anchor. If None, all valid matches are included. Defaults to 3.

            choose_best_only (bool, optional):
                If True, only the highest-scoring matches are selected for each
                role instead of random sampling within match categories.
                Defaults to False.

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
                Whether to drop the final batch if it contains fewer than
                ``batch_size`` triplets. Defaults to False.

            seed (int | None, optional):
                Random seed for deterministic random behavior. Defaults to None.

            show_progress (bool, optional):
                Whether to show a progress bar during the batch building process.
                Defaults to True.

            source (FeatureSet | FeatureSetView):
                The source data from samples are drawn. Note that batches are not
                constructed until `materialize_batches` is called.

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
        if self.is_bound:
            return f"TripletSampler(n_batches={self.num_batches}, batch_size={self.batcher.batch_size})"
        return f"TripletSampler(batch_size={self.batcher.batch_size})"

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
        cfg.update({"sampler_name": "TripletSampler"})
        return cfg

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> TripletSampler:
        """
        Construct a sampler from configuration.

        Args:
            config (dict[str, Any]): Sampler configuration.

        Returns:
            TripletSampler: Unfitted sampler instance.

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
