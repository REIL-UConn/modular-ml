"""Simple batching sampler implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from modularml.core.data.featureset_view import FeatureSetView
from modularml.core.data.schema_constants import ROLE_DEFAULT, STREAM_DEFAULT
from modularml.core.sampling.base_sampler import BaseSampler, SamplerStreamSpec, Samples

if TYPE_CHECKING:
    from modularml.core.data.featureset import FeatureSet


class SimpleSampler(BaseSampler):
    """Single-stream sampler that chunks a :class:`FeatureSet` or view."""

    __SPECS__ = SamplerStreamSpec(
        stream_names=(STREAM_DEFAULT,),
        roles=(ROLE_DEFAULT,),
    )

    def __init__(
        self,
        *,
        batch_size: int = 1,
        shuffle: bool = False,
        group_by: list[str] | None = None,
        stratify_by: list[str] | None = None,
        strict_stratification: bool = True,
        drop_last: bool = False,
        seed: int | None = None,
        show_progress: bool = True,
        source: FeatureSet | FeatureSetView | None = None,
    ):
        """
        Initialize batching logic for a :class:`FeatureSet` or view.

        Description:
            :class:`SimpleSampler` can build batches using one of three strategies:

            1. Grouping (`group_by`): samples with identical column values share a bucket.
            2. Stratification (`stratify_by`): strata interleave samples to balance roles.
            3. Sequential slicing: rows are taken in order (optionally shuffled).

            Each batch is returned as a zero-copy :class:`~modularml.core.data.batch_view.BatchView`
            produced by :meth:`BaseSampler.materialize_batches`.

        Args:
            batch_size (int):
                Number of samples in each batch.

            shuffle (bool):
                Whether to shuffle samples and completed batches.

            group_by (list[str] | None):
                FeatureSet keys that define grouping buckets. Mutually exclusive
                with `stratify_by`.

            stratify_by (list[str] | None):
                Keys used for stratified sampling. Cannot be combined with `group_by`.

            strict_stratification (bool):
                Whether to end batching when any stratum is exhausted.

            drop_last (bool):
                Drop the final incomplete batch if True.

            seed (int | None):
                Random seed for reproducible shuffling.

            show_progress (bool):
                Whether to emit progress updates while materializing batches.

            source (FeatureSet | FeatureSetView | None):
                Optional :class:`FeatureSet` or :class:`FeatureSetView` to bind
                immediately.

        Raises:
            ValueError: If both `group_by` and `stratify_by` are provided.

        """
        super().__init__(
            sources=source,
            batch_size=batch_size,
            shuffle=shuffle,
            group_by=group_by,
            group_by_role=ROLE_DEFAULT,
            stratify_by=stratify_by,
            stratify_by_role=ROLE_DEFAULT,
            strict_stratification=strict_stratification,
            drop_last=drop_last,
            seed=seed,
            show_progress=show_progress,
        )

    def build_samples(self) -> dict[tuple[str, str], Samples]:
        """
        Construct single-stream batches using grouping or stratification.

        Returns:
            dict[tuple[str, str], Samples]:
                Mapping from `(stream_label, source_label)` to :class:`Samples`
                objects describing batch indices and weights.

        Raises:
            RuntimeError: If :meth:`BaseSampler.bind_source` has not been called.
            TypeError: If the bound source is not a :class:`FeatureSetView`.

        """
        if self.sources is None:
            raise RuntimeError(
                "`bind_source` must be called before sampling can occur.",
            )
        src_lbl = next(iter(self.sources.keys()))
        src = self.sources[src_lbl]
        if not isinstance(src, FeatureSetView):
            msg = f"`source` must be of type FeatureSetView. Received: {type(src)}"
            raise TypeError(msg)

        # dict key is 2-tuple of stream_label, source_label
        # For single-stream samplers like this one, we use a default label
        return {
            (STREAM_DEFAULT, src_lbl): Samples(
                role_indices={ROLE_DEFAULT: src.indices},
                role_weights=None,
            ),
        }

    def __repr__(self):
        """Return a developer-friendly string summarizing sampler state."""
        if self.is_bound:
            return f"SimpleSampler(n_batches={self.num_batches}, batch_size={self.batcher.batch_size})"
        return f"SimpleSampler(batch_size={self.batcher.batch_size})"

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
        cfg.update({"sampler_name": "SimpleSampler"})
        return cfg

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> SimpleSampler:
        """
        Construct a sampler from configuration.

        Args:
            config (dict[str, Any]): Serialized sampler configuration.

        Returns:
            SimpleSampler: Unbound sampler instance.

        Raises:
            ValueError: If the configuration was not produced by :meth:`get_config`.

        """
        if ("sampler_name" not in config) or (
            config["sampler_name"] != "SimpleSampler"
        ):
            raise ValueError("Invalid config for SimpleSampler.")

        return cls(
            batch_size=config["batch_size"],
            shuffle=config["shuffle"],
            group_by=config["group_by"],
            stratify_by=config["stratify_by"],
            strict_stratification=config["strict_stratification"],
            drop_last=config["drop_last"],
            seed=config["seed"],
            show_progress=config["show_progress"],
        )
