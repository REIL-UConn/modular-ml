from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from modularml.core.data.schema_constants import DOMAIN_TAGS, REP_RAW
from modularml.core.splitting.base_splitter import BaseSplitter
from modularml.utils.data.data_format import DataFormat
from modularml.utils.data.formatting import ensure_list, to_hashable

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from modularml.core.data.featureset_view import FeatureSetView


class RandomSplitter(BaseSplitter):
    """
    Randomly splits a FeatureSetView into subsets according to user-specified ratios.

    Description:
        This splitter partitions samples randomly into subsets (e.g., "train", "val", "test")
        based on the given ratios. Optionally, samples can be grouped by one or more tag keys
        before splitting, ensuring that all samples from the same group fall into the same subs etc.

        The split operates on **relative indices** within the provided FeatureSetView,
        preserving full traceability via SAMPLE_ID in the source FeatureSet.

    Example:
        ```python
        splitter = RandomSplitter(
            ratios={"train": 0.8, "val": 0.2}, group_by="cell_id", seed=42
        )
        splits = splitter.split(fs_view, return_views=True)
        ```

    """

    def __init__(
        self,
        ratios: Mapping[str, float],
        group_by: str | Sequence[str] | None = None,
        stratify_by: str | Sequence[str] | None = None,
        seed: int = 13,
    ):
        """
        Initialize the RandomSplitter.

        Args:
            ratios (Mapping[str, float]):
                Dictionary mapping subset labels to relative ratios. Must sum to 1.0.
                Example: {"train": 0.7, "val": 0.2, "test": 0.1}.
            group_by (str | Sequence[str] | None, optional):
                One or more tag keys to group samples by before splitting.
                If None, samples are split individually. Mutually exclusive with `stratify_by`.
            stratify_by (str | Sequence[str] | None, optional):
                One or more tag keys to stratify samples by during splitting.
                Ensures each split maintains the same proportion of each stratum as the original data.
                Mutually exclusive with `group_by`.
            seed (int, optional):
                Random seed for reproducibility. Default is 13.

        """
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
        Randomly split a FeatureSetView into multiple subsets.

        Description:
            Splits are based on sample order, shuffled using a fixed random seed. \
            If `group_by` is provided, all samples sharing the same tag values \
            are assigned to the same subset.

        Args:
            view (FeatureSetView):
                The input FeatureSetView to partition.
            return_views (bool, optional):
                If True, returns a mapping of labels to FeatureSetViews. \
                If False, returns relative index arrays. Defaults to True.

        Returns:
            Mapping[str, FeatureSetView] | Mapping[str, Sequence[int]]:
                A mapping of subset label to either FeatureSetViews or index arrays.

        """
        n = len(view)

        # ================================================
        # Case 1: No grouping or stratification
        # ================================================
        if self.group_by is None and self.stratify_by is None:
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
        # Case 2: Stratify by tag(s)
        # ================================================
        if self.stratify_by is not None:
            coll = view.source.collection

            # Extract raw tag arrays as numpy, aligned with the FULL FeatureSet
            tag_data: dict[str, np.ndarray] = coll._get_domain_data(
                domain=DOMAIN_TAGS,
                keys=self.stratify_by,
                fmt=DataFormat.DICT_NUMPY,
                rep=REP_RAW,
                include_rep_suffix=False,
                include_domain_prefix=False,
            )
            # Restrict to the samples inside this view
            view_abs_indices = view.indices  # absolute sample indices
            tag_cols_view: list[np.ndarray] = [
                tag_data[k][view_abs_indices] for k in self.stratify_by
            ]

            # Build stratum keys
            stratum_keys = np.array(
                [tuple(to_hashable(col[i]) for col in tag_cols_view) for i in range(n)],
                dtype=object,
            )

            # Map unique tuple to stratum ID
            unique_strata, inv = np.unique(stratum_keys, return_inverse=True)

            # Build mapping: stratum_id to list of relative indices
            stratum_to_rel_idxs: dict[int, list[int]] = {}
            for rel_idx, s_id in enumerate(inv):
                stratum_to_rel_idxs.setdefault(s_id, []).append(rel_idx)

            # For each stratum, shuffle and split according to ratios
            # Then combine into final split_indices
            split_indices: dict[str, list[int]] = {label: [] for label in self.ratios}

            for s_id in range(len(unique_strata)):
                stratum_rel_idxs = np.array(stratum_to_rel_idxs[s_id])
                self.rng.shuffle(stratum_rel_idxs)

                # Apply split boundaries within this stratum
                stratum_n = len(stratum_rel_idxs)
                boundaries = self._compute_split_boundaries(stratum_n)
                for label, (start, end) in boundaries.items():
                    split_indices[label].extend(stratum_rel_idxs[start:end])

            # Convert to sorted numpy arrays
            split_indices = {
                label: np.sort(np.array(idxs, dtype=int))
                for label, idxs in split_indices.items()
            }

            return self._return_splits(
                view=view,
                split_indices=split_indices,
                return_views=return_views,
            )

        # ================================================
        # Case 3: Group by tag(s)
        # ================================================
        # Extract raw tag arrays as numpy for the grouping keys
        tag_data: dict[str, np.ndarray] = view.get_tags(
            fmt=DataFormat.DICT_NUMPY,
            tags=self.group_by,
            rep=REP_RAW,
            include_rep_suffix=False,
            include_domain_prefix=False,
        )

        # Build group keys
        # Example:
        #   if grouping by ["cell_id", "cycle_number"]
        #   then group_keys[i] = ("A1", 45)
        group_keys = np.array(
            [tuple(col[i] for col in tag_data.values()) for i in range(n)],
            dtype=object,
        )
        unique_groups, inv = np.unique(group_keys, return_inverse=True)

        # Build mapping: group_id to list of relative indices
        group_to_rel_idxs: dict[int, list[int]] = {}
        for rel_idx, g_id in enumerate(inv):
            group_to_rel_idxs.setdefault(g_id, []).append(rel_idx)

        # Shuffle group IDs, not individual samples
        group_ids = np.arange(len(unique_groups))
        self.rng.shuffle(group_ids)

        # Apply split boundaries at the group level
        boundaries = self._compute_split_boundaries(len(group_ids))
        split_indices: dict[str, list[int]] = {label: [] for label in self.ratios}

        for label, (start, end) in boundaries.items():
            selected = set(group_ids[start:end])
            for g in selected:
                split_indices[label].extend(group_to_rel_idxs[g])

        # Convert & sort
        split_indices = {
            label: np.sort(np.array(idxs, dtype=int))
            for label, idxs in split_indices.items()
        }

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
        Compute index boundaries for each split given n total elements.

        Uses the Largest Remainder Method to ensure:
        - Total count equals `n`
        - Allocation is proportional to `self.ratios`
        - No systematic bias toward the last group

        Args:
            n (int):
                The total number of samples.

        Returns:
            dict[str, tuple[int, int]]:
                A dictionary where keys are subset labels and values are tuples representing
                the start and end indices for each subset.

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
            "splitter_name": "RandomSplitter",
            "ratios": self.ratios,
            "group_by": self.group_by,
            "stratify_by": self.stratify_by,
            "seed": self.seed,
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> BaseSplitter:
        """
        Construct a Splitter from configuration.

        Args:
            config (dict[str, Any]): Splitter configuration.

        Returns:
            BaseSplitter: Unfitted splitter instance.

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
        Return runtime (i.e. rng) state of the splitter.

        Returns:
            dict[str, Any]: Splitter state.

        """
        return {"rng_state": self.rng.bit_generator.state}

    def set_state(self, state: dict[str, Any]) -> None:
        """
        Restore runtime state of the splitter.

        Args:
            state (dict[str, Any]):
                State produced by get_state().

        """
        self.rng.bit_generator.state = state["rng_state"]
