"""Randomized partitioning utilities for FeatureSet views."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from modularml.core.experiment.experiment_context import ExperimentContext
from modularml.core.references.featureset_reference import FeatureSetColumnReference
from modularml.core.splitting.base_splitter import BaseSplitter
from modularml.utils.data.data_format import DataFormat
from modularml.utils.data.formatting import ensure_list, to_hashable

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from modularml.core.data.featureset_view import FeatureSetView


class RandomSplitter(BaseSplitter):
    """
    Randomly split a :class:`FeatureSetView` according to labeled ratios.

    Description:
        Partitions samples into subsets (for example, `train`, `val`, `test`) based on
        specified ratios. Optional grouping ensures all samples sharing the same tag
        values stay together, while stratification preserves per-stratum proportions.

    Attributes:
        ratios (dict[str, float]): Mapping of subset labels to ratios that sum to 1.0.
        group_by (list[str] | None): Tag keys that enforce group-wise assignment.
        stratify_by (list[str] | None): Tag keys that enforce stratified sampling.
        seed (int): Seed used to initialize the NumPy random generator.
        rng (np.random.Generator): Random generator used for shuffling/group assignment.

    """

    def __init__(
        self,
        ratios: Mapping[str, float],
        *,
        group_by: str | Sequence[str] | None = None,
        stratify_by: str | Sequence[str] | None = None,
        strict_stratification: bool = False,
        seed: int = 13,
    ):
        """
        Initialize the randomized splitter.

        Args:
            ratios (Mapping[str, float]):
                Subset ratios that sum to 1.0
                (e.g., `{"train": 0.7, "val": 0.2, "test": 0.1}`).

            group_by (list[str] | None):
                Column selectors used to keep groups together.

            stratify_by (list[str] | None):
                Column selectors used to balance strata.
                Mutually exclusive with `group_by`.

            strict_stratification (bool):
                Whether the returned splits should be perfectly stratified,
                or use all samples. Defaults to False.

            seed (int): Random seed used to initialize the generator.

        Raises:
            ValueError:
                If ratios fall outside [0, 1], do not sum to 1.0, or
                grouping/stratification are both requested.

        """
        if not all((v >= 0) and (v <= 1) for v in ratios.values()):
            msg = "Ratios values must be between 0 and 1."
            raise ValueError(msg)
        total = float(sum(ratios.values()))
        if not np.isclose(total, 1.0):
            msg = f"ratios must sum to 1.0. Received: {total})."
            raise ValueError(msg)

        self.ratios = dict(ratios)
        self.group_by: list[str] | None = (
            None if group_by is None else ensure_list(group_by)
        )
        self.stratify_by: list[str] | None = (
            None if stratify_by is None else ensure_list(stratify_by)
        )
        self.strict_stratification = strict_stratification
        if self.group_by and self.stratify_by:
            raise ValueError("`group_by` and `stratify_by` are mutually exclusive.")

        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

    # =====================================================
    # Core splitting logic
    # =====================================================
    def split(
        self,
        view: FeatureSetView,
        *,
        return_views: bool = True,
    ) -> Mapping[str, FeatureSetView] | Mapping[str, Sequence[int]]:
        """
        Split a :class:`FeatureSetView` into labeled subsets.

        Description:
            Samples (or sample groups/strata) are shuffled using the configured random
            generator before allocating according to `ratios`. Grouping guarantees
            all samples with the same tag values move together, while stratification
            preserves per-stratum proportions.

        Args:
            view (FeatureSetView):
                Input view to partition.
            return_views (bool):
                If True, return :class:`FeatureSetView` objects; otherwise return
                relative index arrays.

        Returns:
            Mapping[str, FeatureSetView] | Mapping[str, Sequence[int]]:
                Subset label mapped to either views or relative indices.

        Raises:
            ValueError: If grouping/stratification yields empty subsets.

        """
        # Case 1: Stratify
        if self.stratify_by is not None:
            split_idxs = self._stratify(view=view)
            return self._return_splits(
                view=view,
                split_indices=split_idxs,
                return_views=return_views,
            )

        # Case 2: Groupby
        if self.group_by is not None:
            split_idxs = self._groupby(view=view)
            return self._return_splits(
                view=view,
                split_indices=split_idxs,
                return_views=return_views,
            )

        # Case 3: No grouping or stratification
        n = len(view)
        rel_indices = np.arange(n)
        self.rng.shuffle(rel_indices)
        boundaries = self._compute_split_boundaries(n)
        split_indices: dict[str, np.ndarray] = {}
        for label, (start, end) in boundaries.items():
            split_indices[label] = rel_indices[start:end]
        return self._return_splits(
            view=view,
            split_indices=split_indices,
            return_views=return_views,
        )

    # ================================================
    # Utilities
    # ================================================
    def _compute_split_boundaries(self, n: int) -> dict[str, tuple[int, int]]:
        """
        Compute start/end indices for each split using the largest remainder method.

        Args:
            n (int): Total number of elements to distribute.

        Returns:
            dict[str, tuple[int, int]]:
                Mapping from subset label to `[start, end)` index boundaries.

        Raises:
            ValueError:
                If no samples are provided or any subset would receive zero samples.

        """
        if n <= 0:
            raise ValueError("Cannot split zero samples.")

        labels = list(self.ratios.keys())
        ratios = list(self.ratios.values())

        # 1. Compute raw allocations (can be fractional)
        raw = [r * n for r in ratios]

        # 2. Allocate floor
        base = [int(x) for x in raw]
        allocated = sum(base)

        # 3. Distribute remaining
        remainders = [raw[i] - base[i] for i in range(len(raw))]
        n_remaining = int(n - allocated)
        if n_remaining > 0:
            order = sorted(
                np.arange(len(remainders)),
                key=lambda i: remainders[i],
                reverse=True,
            )
            for i in order[:n_remaining]:
                base[i] += 1

        # 4. Enforce all groups have at least one sample
        for i, cnt in enumerate(base):
            if cnt < 1:
                closure = "."
                if self.group_by is not None:
                    closure = ", or adjust your grouping conditions."
                elif self.stratify_by is not None:
                    closure = ", or adjust your stratification conditions."
                msg = (
                    f"Split '{labels[i]}' would receive zero samples. "
                    f"Increase split ratio{closure}"
                )
                raise ValueError(msg)

        # 5. Build boundaries
        boundaries = {}
        current = 0
        for lbl, cnt in zip(labels, base, strict=True):
            boundaries[lbl] = (current, current + cnt)
            current += cnt
        return boundaries

    def _stratify(self, view: FeatureSetView) -> dict[str, list[int]]:
        # self.stratify_by contains a list of strings, each string can be any column in FeatureSet
        # FeatureSetColumnReference infers the node, domain, key, and variant given a user-defined strings
        # E.g, "voltages" -> ("MyFS", "features", "voltages", "raw")
        strata_refs: list[FeatureSetColumnReference] = [
            FeatureSetColumnReference.from_string(
                val=x,
                known_attrs={
                    "node_label": view.source.label,
                    "node_id": view.source.node_id,
                },
                experiment=ExperimentContext.get_active(),
            )
            for x in self.stratify_by
        ]

        # Collect data for each defined stratify_by key
        # The np data for each key uses relative indices (role_indices defines absolute indices)
        strata_data: dict[str, np.ndarray] = {}
        for ref in strata_refs:
            # Get source data
            k = ref.to_string()
            if k in strata_data:
                msg = (
                    f"ColumnReference.to_string() already exists in `strata_data`: {k}"
                )
                raise ValueError(msg)
            ref_data: np.ndarray = view.get_data(
                columns=f"{ref.domain}.{ref.key}.{ref.rep}",
                fmt=DataFormat.NUMPY,
            )
            strata_data[k] = ref_data

        # Construct strata buckets
        # Each bucket defines a unique strata class
        # Each bucket holds relative indices of the rows of `view` belonging to each strata
        buckets: dict[tuple, list[int]] = {}
        for i in range(len(view)):
            row_vals = tuple(to_hashable(strata_data[k][i]) for k in strata_data)
            buckets.setdefault(row_vals, []).append(i)

        # Shuffle relative indices in each bucket
        for k, rel_idxs in buckets.items():
            buckets[k] = np.asarray(rel_idxs)
            self.rng.shuffle(buckets[k])

        # If strict, trim all buckets to the size of the smallest stratum
        if self.strict_stratification:
            min_size = min(len(v) for v in buckets.values())
            buckets = {k: v[:min_size] for k, v in buckets.items()}

        # Split each stratum independently by ratios, then merge per split label.
        # This guarantees every split contains samples from every stratum.
        split_labels = list(self.ratios.keys())
        split_indices: dict[str, list[int]] = {lbl: [] for lbl in split_labels}

        for stratum_idxs in buckets.values():
            boundaries = self._compute_split_boundaries(len(stratum_idxs))
            for label, (start, end) in boundaries.items():
                split_indices[label].extend(stratum_idxs[start:end])

        return split_indices

    def _groupby(self, view: FeatureSetView) -> dict[str, list[int]]:
        # self.group_by contains a list of strings, each string can be any column in FeatureSet
        # FeatureSetColumnReference infers the node, domain, key, and variant given a user-defined strings
        # E.g, "voltages" -> ("MyFS", "features", "voltages", "raw")
        group_refs: list[FeatureSetColumnReference] = [
            FeatureSetColumnReference.from_string(
                val=x,
                known_attrs={
                    "node_label": view.source.label,
                    "node_id": view.source.node_id,
                },
                experiment=ExperimentContext.get_active(),
            )
            for x in self.group_by
        ]

        # Collect data for each defined group_by key
        group_data: dict[str, np.ndarray] = {}
        for ref in group_refs:
            # Get source data
            k = ref.to_string()
            if k in group_data:
                msg = f"ColumnReference.to_string() already exists in `group_data`: {k}"
                raise ValueError(msg)
            ref_data: np.ndarray = view.get_data(
                columns=f"{ref.domain}.{ref.key}.{ref.rep}",
                fmt=DataFormat.NUMPY,
            )
            group_data[k] = ref_data

        # Convert `group_data` to rows of tuples, each tuple becomes the unique grouping key
        # These row_tuples are used to construct unique group buckets
        # Each bucket holds relative indices of the rows of `view` belonging to each bucket
        buckets: dict[tuple, list[int]] = {}
        for i in range(len(view)):
            row_vals = tuple(to_hashable(group_data[k][i]) for k in group_data)
            buckets.setdefault(row_vals, []).append(i)

        # Split groups on ratios
        group_keys = list(buckets.keys())
        boundaries = self._compute_split_boundaries(len(group_keys))
        split_indices: dict[str, list[int]] = {}
        for split_lbl, (start, end) in boundaries.items():
            split_groups = group_keys[start:end]
            split_indices[split_lbl] = []
            for g in split_groups:
                split_indices[split_lbl].extend(buckets[g])

        return split_indices

    # ================================================
    # Configurable
    # ================================================
    def get_config(self) -> dict[str, Any]:
        """
        Return configuration required to reconstruct this splitter.

        Returns:
            dict[str, Any]: Serializable splitter configuration (sources excluded).

        """
        return {
            "splitter_name": "RandomSplitter",
            "ratios": self.ratios,
            "group_by": self.group_by,
            "stratify_by": self.stratify_by,
            "seed": self.seed,
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> BaseSplitter:
        """
        Construct a splitter from configuration.

        Args:
            config (dict[str, Any]): Serialized splitter configuration.

        Returns:
            BaseSplitter: Unbound splitter instance.

        """
        return cls(
            ratios=config["ratios"],
            group_by=config.get("group_by"),
            stratify_by=config.get("stratify_by"),
            seed=config["seed"],
        )

    # ================================================
    # Stateful
    # ================================================
    def get_state(self) -> dict[str, Any]:
        """
        Return the runtime RNG state of the splitter.

        Returns:
            dict[str, Any]: Splitter state dictionary suitable for :meth:`set_state`.

        """
        return {"rng_state": self.rng.bit_generator.state}

    def set_state(self, state: dict[str, Any]) -> None:
        """
        Restore the runtime RNG state of the splitter.

        Args:
            state (dict[str, Any]): State previously produced by :meth:`get_state`.

        """
        self.rng.bit_generator.state = state["rng_state"]
