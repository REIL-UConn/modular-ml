from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from modularml.context.experiment_context import ExperimentContext
from modularml.core.data.batch_view import BatchView
from modularml.core.references.featureset_reference import FeatureSetColumnReference
from modularml.utils.data.data_format import DataFormat
from modularml.utils.data.formatting import ensure_list, to_hashable

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from modularml.core.data.featureset_view import FeatureSetView


class Batcher:
    """
    Encapsulates batching logic for samplers.

    Description:
        Handles grouping, stratification, shuffling, and slicing of absolute
        sample indices into zero-copy BatchView objects. This class does not
        perform sampling; it only batches already-selected indices.
    """

    def __init__(
        self,
        *,
        batch_size: int = 1,
        shuffle: bool = False,
        drop_last: bool = False,
        group_by: list[str] | None = None,
        stratify_by: list[str] | None = None,
        strict_stratification: bool = True,
        group_by_role: str | None = None,
        stratify_by_role: str | None = None,
        seed: int | None = None,
    ):
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)

        self.group_by = ensure_list(group_by)
        self.group_by_role = group_by_role
        self.stratify_by = ensure_list(stratify_by)
        self.stratify_by_role = stratify_by_role
        self.strict_stratification = strict_stratification

        if self.group_by and self.stratify_by:
            raise ValueError("`group_by` and `stratify_by` are mutually exclusive.")

        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def batch(
        self,
        *,
        view: FeatureSetView,
        role_indices: dict[str, NDArray[np.int_]],
        role_weights: dict[str, NDArray[np.float32]] | None = None,
    ) -> list[BatchView]:
        """
        Slice aligned role indices into BatchView objects.

        Args:
            view:
                FeatureSetView used for BatchView construction.
            role_indices (dict[str, np.ndarray]):
                Mapping of role name to aligned absolute indices.
            role_weights (dict[str, np.ndarray] | None):
                Optional mapping of role name to aligned weights.

        Returns:
            list[BatchView]:
                List of zero-copy BatchView objects.

        """
        # Get sample indices (each role must have same number of samples)
        role_indices = {k: np.asarray(v).reshape(-1) for k, v in role_indices.items()}
        role_weights = (
            None
            if role_weights is None
            else {k: np.asarray(v).reshape(-1) for k, v in role_weights.items()}
        )
        n = len(next(iter(role_indices.values())))
        for v in role_indices.values():
            if len(v) != n:
                msg = f"All roles must have the same number of samples: {len(v)} != {n}"
                raise ValueError(msg)

        # Ensure proper arguments provided for multi-role stratification/grouping
        is_multi_role = len(role_indices) > 1
        if is_multi_role:
            if self.group_by and self.group_by_role is None:
                raise ValueError(
                    "Multi-role group_by requires defining the role in which to determine common groups. Set `group_by_role`.",
                )
            if self.stratify_by and self.stratify_by_role is None:
                raise ValueError(
                    "Multi-role stratify_by requires defining the role in which to determine strata. Set `stratify_by_role`.",
                )
        elif not is_multi_role and self.group_by:
            self.group_by_role = next(iter(role_indices.keys()))
        elif not is_multi_role and self.stratify_by:
            self.stratify_by_role = next(iter(role_indices.keys()))

        # Construct batches
        if self.stratify_by:
            return self._stratify(
                view=view,
                role_indices=role_indices,
                role_weights=role_weights,
            )
        if self.group_by:
            return self._group(
                view=view,
                role_indices=role_indices,
                role_weights=role_weights,
            )

        # If no groupy_by or stratify_by, shuffle before batching
        if self.shuffle:
            perm = self.rng.permutation(n)
            for k in role_indices:
                role_indices[k] = role_indices[k][perm]
                if role_weights is not None:
                    role_weights[k] = role_weights[k][perm]

        # Create sequential batches
        batches: list[BatchView] = []
        for start in range(0, n, self.batch_size):
            stop = start + self.batch_size
            if stop > n and self.drop_last:
                break

            batches.append(
                BatchView(
                    source=view.source,
                    role_indices={k: v[start:stop] for k, v in role_indices.items()},
                    role_indice_weights=None
                    if role_weights is None
                    else {k: v[start:stop] for k, v in role_weights.items()},
                ),
            )

        return batches

    def _stratify(
        self,
        *,
        view: FeatureSetView,
        role_indices: dict[str, NDArray[np.int_]],
        role_weights: dict[str, NDArray[np.float32]] | None = None,
    ) -> list[BatchView]:
        """
        Build batches ensuring balanced representation across strata.

        Description:
            Samples are partitioned into strata defined by `stratify_by` columns. \
            Batches are created by interleaving samples from each stratum.

            - If `strict_stratification=True`, interleaving stops when the first \
              stratum is exhausted (perfectly balanced).
            - Otherwise, interleaving continues until all strata are empty.

        Returns:
            list[BatchView]:
                BatchViews where each batch contains samples from all strata.

        Raises:
            ValueError:
                If the number of strata exceeds `batch_size`.

        """
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
        # Each bucket holds absolute indices of the rows of `view` belonging to each strata
        # For multi-role batching, use only the role data in `stratify_by_role`
        abs_indices = role_indices[self.stratify_by_role]
        abs_to_viewpos = {abs_i: pos for pos, abs_i in enumerate(view.indices)}
        buckets: dict[tuple, list[int]] = {}
        for i, abs_idx in enumerate(abs_indices):
            view_pos = abs_to_viewpos[abs_idx]
            row_vals = tuple(to_hashable(strata_data[k][view_pos]) for k in strata_data)
            buckets.setdefault(row_vals, []).append(i)

        # Number of buckets must be <= batch size
        if len(buckets) > self.batch_size:
            msg = f"The batch size must be larger than the number of strata: {self.batch_size} < {len(buckets)}"
            raise ValueError(msg)

        # Shuffle absolute indices in each bucket
        for k, abs_idxs in buckets.items():
            buckets[k] = np.asarray(abs_idxs)
            if self.shuffle:
                self.rng.shuffle(buckets[k])

        # Interleave (cast each bucket to an iterator)
        iterators = {k: iter(v) for k, v in buckets.items()}
        keys = list(iterators)

        # Holds all abs. indices sorted such each each strata is evenly distributed
        # along the length of the list
        interleaved_idxs = []

        # There are 2 ways to performed stratification:
        #   - interleave until all samples are used up
        #       - after first group runs out, no longer balanced strata
        #   - interleave until first group runs out (strict_stratification)
        #       - all batches balanced, but may not use all samples
        exhausted = set()
        while (
            (len(exhausted) == 0)
            if self.strict_stratification
            else (len(exhausted) < len(keys))
        ):
            # Add next element from each strata into interleaved array
            for k in keys:
                if k in exhausted:
                    continue
                try:
                    interleaved_idxs.append(next(iterators[k]))
                except StopIteration:
                    exhausted.add(k)

        # Construct batches from interleaved sample indices
        interleaved_idxs = np.array(interleaved_idxs)

        batches: list[BatchView] = []
        for start in range(0, len(interleaved_idxs), self.batch_size):
            stop = start + self.batch_size
            batch_idxs = interleaved_idxs[start:stop]
            if len(batch_idxs) < self.batch_size and self.drop_last:
                continue

            batch_role_idxs = {k: role_indices[k][batch_idxs] for k in role_indices}
            batch_role_weights = (
                None
                if role_weights is None
                else {k: role_weights[k][batch_idxs] for k in role_weights}
            )

            batches.append(
                BatchView(
                    source=view.source,
                    role_indices=batch_role_idxs,
                    role_indice_weights=batch_role_weights,
                ),
            )

        return batches

    def _group(
        self,
        *,
        view: FeatureSetView,
        role_indices: dict[str, NDArray[np.int_]],
        role_weights: dict[str, NDArray[np.float32]] | None = None,
    ) -> list[BatchView]:
        """
        Build batches such that each batch contains samples only from a single group.

        Description:
            Samples are assigned to groups based on the `group_by` column values. \
            Each group is independently shuffled (optional) and split into \
            batches of size `batch_size`.

        Returns:
            list[BatchView]:
                One BatchView per constructed batch. Each batch has role "default".

        Notes:
            - No cross-group mixing occurs.
            - Groups with fewer samples than `batch_size` produce no batch \
              when `drop_last=True`.

        """
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
        # Each bucket holds absolute indices of the rows of `view` belonging to each bucket
        # For multi-role batching, use only the role data in `group_by_role`
        abs_indices = role_indices[self.group_by_role]
        abs_to_viewpos = {abs_i: pos for pos, abs_i in enumerate(view.indices)}
        buckets: dict[tuple, list[int]] = {}
        for i, abs_idx in enumerate(abs_indices):
            view_pos = abs_to_viewpos[abs_idx]
            row_vals = tuple(to_hashable(group_data[k][view_pos]) for k in group_data)
            buckets.setdefault(row_vals, []).append(i)

        # Build batches
        batches: list[BatchView] = []
        for abs_idxs in buckets.values():
            # Shuffle if specified
            idxs = np.asarray(abs_idxs)
            if self.shuffle:
                self.rng.shuffle(idxs)

            # Build batches from this group
            for start in range(0, len(idxs), self.batch_size):
                batch_idxs = idxs[start : start + self.batch_size]
                if len(batch_idxs) < self.batch_size and self.drop_last:
                    continue

                batch_role_idxs = {k: role_indices[k][batch_idxs] for k in role_indices}
                batch_role_weights = (
                    None
                    if role_weights is None
                    else {k: role_weights[k][batch_idxs] for k in role_weights}
                )

                batches.append(
                    BatchView(
                        source=view.source,
                        role_indices=batch_role_idxs,
                        role_indice_weights=batch_role_weights,
                    ),
                )

        return batches

    # ============================================
    # Configuration
    # ============================================
    def get_config(self) -> dict[str, Any]:
        return {
            "batch_size": self.batch_size,
            "shuffle": self.shuffle,
            "drop_last": self.drop_last,
            "group_by": self.group_by,
            "group_by_role": self.group_by_role,
            "stratify_by": self.stratify_by,
            "stratify_by_role": self.stratify_by_role,
            "strict_stratification": self.strict_stratification,
            "seed": self.seed,
        }

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> Batcher:
        return cls(**cfg)
