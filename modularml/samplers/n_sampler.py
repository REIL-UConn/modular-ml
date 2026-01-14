from __future__ import annotations

import bisect
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from modularml.context.experiment_context import ExperimentContext
from modularml.core.data.featureset_view import FeatureSetView
from modularml.core.data.schema_constants import ROLE_ANCHOR, STREAM_DEFAULT
from modularml.core.references.featureset_reference import FeatureSetColumnReference
from modularml.core.sampling.base_sampler import BaseSampler, SamplerStreamSpec, Samples
from modularml.utils.data.data_format import DataFormat

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from modularml.core.data.featureset import FeatureSet
    from modularml.core.sampling.similiarity_condition import SimilarityCondition


@dataclass(frozen=True)
class SortedColumn:
    values: np.ndarray  # Sorted values
    original_to_sorted_idxs: (
        np.ndarray
    )  # relative indices used to sort the original array


class NSampler(BaseSampler):
    """
    General N-way sampler for similarity-based multi-role sample selection.

    This sampler generalizes pairwise sampling to support an arbitrary
    number of roles (e.g., "positive", "negative", "pair").
    Each role defines its own set of similarity conditions, and these
    conditions determine which samples may be paired with each anchor.
    The sampler selects valid matches per role, intersects anchors
    across all roles, aligns indexing, and produces N-way batches.
    """

    __SPECS__ = SamplerStreamSpec(
        stream_names=(STREAM_DEFAULT,),
        roles=(ROLE_ANCHOR,),  # minimum guaranteed role
    )

    def __init__(
        self,
        condition_mapping: dict[str, dict[str, SimilarityCondition]],
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
        Initialize an N-way similarity-based sampler.

        Description:
            Each role in `condition_mapping` has its own set of
            SimilarityCondition objects. For each anchor sample, the
            sampler evaluates all candidates according to these conditions,
            selects up to `max_samples_per_anchor` matches per role,
            computes per-role scores, and later intersects anchors across
            all roles to ensure that only anchors with valid matches for
            every role are retained.

        Args:
            condition_mapping (dict[str, dict[str, SimilarityCondition]]):
                Mapping from role name to {column_key to SimilarityCondition}.
                E.g., `{"pair": {...}}` or `{"positive": {...}, "negative": {...}}`

            batch_size (int):
                Number of N-way samples per batch.

            shuffle (bool):
                Shuffle final aligned samples across all roles.

            max_samples_per_anchor (int | None):
                For each role, maximum number of selected samples.
                If None, uses all candidates.

            choose_best_only (bool):
                Whether to select only the highest-scoring sample(s) per role.

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

            drop_last (bool):
                Whether to drop final incomplete batch.

            seed (int | None):
                RNG seed.

            show_progress (bool, optional):
                Whether to show a progress bar during the batch building process.
                Defaults to True.

            source (FeatureSet | FeatureSetView):
                The source data from samples are drawn. Note that batches are not
                constructed until `materialize_batches` is called.

        """
        self.condition_mapping = {
            role: dict(conds) for role, conds in condition_mapping.items()
        }
        if ROLE_ANCHOR in self.condition_mapping:
            msg = f"Condition mapping cannot contain a role named '{ROLE_ANCHOR}'."
            raise ValueError(msg)
        self.max_samples_per_anchor = max_samples_per_anchor
        self.choose_best_only = choose_best_only

        super().__init__(
            batch_size=batch_size,
            shuffle=shuffle,
            group_by=group_by,
            group_by_role=group_by_role,
            stratify_by=stratify_by,
            stratify_by_role=stratify_by_role,
            strict_stratification=strict_stratification,
            drop_last=drop_last,
            seed=seed,
            show_progress=show_progress,
            sources=source,
        )

    def build_samples(self) -> dict[tuple[str, str], Samples]:
        """
        Construct N-way samples using similarity-based matching logic.

        Description:
            For each anchor sample in the bound FeatureSetView:
                1. Resolve role-specific columns for each role.
                2. Generate role-specific candidate matches using
                   similarity and fallback rules.
                3. Intersect anchors across all roles so only anchors
                   with valid matches in every role are retained.
                4. Align role indices according to the canonical anchor
                   ordering.
                5. Shuffle results if requested.
                6. Slice aligned data into batches of size ``batch_size``.

            The resulting BatchView objects contain:
                - "anchor": aligned anchor indices
                - one key per role: aligned role indices
                - role-specific sample weights

        Returns:
            dict[tuple[str, str], Samples]:
                Mapping of stream labels to Samples objects with attributes
                representing an N-way batch of aligned sample indices and weights.
                The dict key must be a 2-tuple of (stream label, source FeatureSet label).

        Raises:
            RuntimeError:
                If no source is bound.
            TypeError:
                If the bound source is not a FeatureSetView.
            ValueError:
                If fewer than two samples are available, or if roles
                fail to produce a valid non-empty intersection of
                anchor samples.

        """
        # Validate sources
        if self.sources is None:
            raise RuntimeError(
                "`bind_source` must be called before sampling can occur.",
            )
        src_lbl = next(iter(self.sources.keys()))
        src = self.sources[src_lbl]
        if not isinstance(src, FeatureSetView):
            msg = f"`source` must be of type FeatureSetView. Received: {type(src)}"
            raise TypeError(msg)
        if len(src) < 2:
            msg = f"NSampler requires at least 2 samples; got {len(src)}."
            raise ValueError(msg)

        # Precompute specs per role
        role_specs: dict[str, list[dict[str, Any]]] = {
            role: self._resolve_columns_for_role(view=src, conds=role_conds)
            for role, role_conds in self.condition_mapping.items()
        }

        # Setup progress bar (only shows if `self.show_progress` is True)
        self._progress_task.set_total(total=len(role_specs) * len(src))

        # Build role-specific matchings
        # Each role key contains a tuple of: anchor indices, role indices, and role scores
        # All returned indices are absolute indicies from the parent source
        role_results: dict[str, tuple[np.ndarray]] = {
            role: self._generate_role_matches(view=src, specs=specs)
            for role, specs in role_specs.items()
        }

        # Each role may return a different array of anchor indices
        # We need to get the intersection and ensure proper alignment
        role_idxs, role_weights = self._standardize_role_idxs(d=role_results)

        # dict key is 2-tuple of stream_label, source_label
        # For single-stream samplers like this one, we use a default label
        return {
            (STREAM_DEFAULT, src_lbl): Samples(
                role_indices=role_idxs,
                role_weights=role_weights,
            ),
        }

    def __repr__(self):
        if self.is_bound:
            return f"NSampler(n_batches={self.num_batches}, batch_size={self.batcher.batch_size})"
        return f"NSampler(batch_size={self.batcher.batch_size})"

    # =====================================================
    # Properties
    # =====================================================
    @property
    def role_names(self) -> list[str]:
        """
        Names of roles produced by this sampler.

        All output streams have the same set of roles.
        """
        return (ROLE_ANCHOR, *self.condition_mapping.keys())

    # =====================================================
    # Helpers
    # =====================================================
    def _standardize_role_idxs(self, d: dict[str, tuple[NDArray, ...]]):
        """
        Standardize and align anchor and role indices across all roles.

        Description:
            Each role independently produces:
                - anchor indices
                - matched sample indices
                - per-pair scores

            Since each role may produce different sets of anchors, this
            method ensures consistent N-way alignment by:

                1. Converting all arrays to NumPy and validating lengths.
                2. Computing the intersection of anchor indices across
                   all roles.
                3. Filtering each role to keep only anchors in the
                   intersection.
                4. Sorting each role's anchors to match a canonical
                   global anchor order.
                5. Building aligned arrays of role indices and scores
                   keyed by:
                        - "anchor"
                        - each role name

        Args:
            d (dict[str, tuple[NDArray,...]]):
                Mapping from role name → (anchor_idxs, role_idxs, role_scores),
                all using absolute sample indices.

        Returns:
            tuple[dict[str, NDArray], dict[str, NDArray]]:
                A pair of dictionaries:
                    - role_idxs:  aligned anchor and role indices
                    - role_scores: aligned score arrays

        Raises:
            ValueError:
                If any role returns mismatched array lengths, if anchors
                fail to intersect across roles, or if anchor order cannot
                be aligned across roles.

        """
        # 1. Convert everything to NumPy arrays
        role_arrays = {}
        for role, arrs in d.items():
            a, r, s = arrs
            a = np.asarray(a).reshape(-1)
            r = np.asarray(r).reshape(-1)
            s = np.asarray(s).reshape(-1)

            if not (len(a) == len(r) == len(s)):
                msg = f"Role '{role}' returned arrays of mismatched length: {len(a)}, {len(r)}, {len(s)}"
                raise ValueError(msg)

            role_arrays[role] = (a, r, s)

        # 2. Compute global valid anchor set (intersection)
        anchor_sets = [set(a) for (a, _, _) in role_arrays.values()]
        valid_anchors = set.intersection(*anchor_sets)
        if not valid_anchors:
            msg = (
                "No common anchors exist across all roles. "
                "At least one anchor must produce valid matches for every role."
            )
            raise ValueError(msg)

        # 3. For each role, keep only anchors in valid_anchors
        all_role_idxs: dict[str, np.ndarray[int]] = {}
        all_role_scores: dict[str, np.ndarray[float]] = {}
        for role, (a, r, s) in role_arrays.items():
            # Keep only valid anchors
            mask = np.isin(a, list(valid_anchors))
            a2 = a[mask]
            r2 = r[mask]
            s2 = s[mask]

            # Sort by anchor idx
            perm = np.argsort(a2)

            # Apply this permutation to all arrays
            if ROLE_ANCHOR not in all_role_idxs:
                all_role_idxs[ROLE_ANCHOR] = a2[perm]
            elif not np.array_equal(all_role_idxs[ROLE_ANCHOR], a2[perm]):
                raise ValueError("Failed to align anchor indices across roles.")
            all_role_idxs[role] = r2[perm]

            # Apply same perm for scores
            if ROLE_ANCHOR not in all_role_scores:
                all_role_scores[ROLE_ANCHOR] = np.ones_like(s2, dtype=float)
            all_role_scores[role] = s2[perm]

        return all_role_idxs, all_role_scores

    def _find_numeric_matches(
        self,
        anchor_val: Any,
        cond: SimilarityCondition,
        spec: dict[str, Any],
    ) -> tuple[set[int], set[int]]:
        """
        Identify numeric matches using tolerance-based binary search.

        Description:
            Uses pre-sorted numeric column values to identify samples
            whose values fall within the similarity window defined by:
                |anchor_val - candidate_val| <= cond.tolerance

            Returns two disjoint sets:
                - match indices
                - non-match indices

        Args:
            anchor_val (Any):
                The value of the anchor sample.
            cond (SimilarityCondition):
                Similarity rule with mode and tolerance.
            spec (dict[str, Any]):
                Column specification containing sorted values,
                index mappings, and the categorical flag.

        Returns:
            tuple[set[int], set[int]]:
                (matched_relative_indices, non_matched_relative_indices)

        Raises:
            ValueError:
                If the column is categorical or if match and non-match
                sets overlap (should never occur).

        """
        if spec["categorical"]:
            raise ValueError("Cannot compute numeric match on categorical data.")

        # orig_idxs:    [0, 1, 2, 3, 4, 5, ...]
        # after_sort:   [4, 2, 1, 5, 3, 0, ...]  <- this is what is stored in `original_to_sorted_idxs`
        sorted_col: SortedColumn = spec["sorted"]
        vals = sorted_col.values
        idxs = sorted_col.original_to_sorted_idxs
        tol = cond.tolerance

        # Similar values will be within the [a - tol, a + tol] window of the sorted array
        # Similarity: |a - b| <= tol -> window [a - tol, a + tol]
        left = anchor_val - tol
        right = anchor_val + tol

        li = bisect.bisect_left(vals, left)
        ri = bisect.bisect_right(vals, right)

        sim_idxs = set(idxs[li:ri])
        dissim_idxs = set(idxs) - sim_idxs

        overlap = set.intersection(sim_idxs, dissim_idxs)
        if len(overlap):
            msg = f"Samples detected in both matching and non-match groups: {overlap}"
            raise ValueError(msg)

        # Return (matches, non-matches)
        if cond.mode == "similar":
            return sim_idxs, dissim_idxs

        # Debugging that values are unique in match vs non-match
        # print(f"anchor_val: {anchor_val}")
        # print("sim_vals:", {spec["col"][i] for i in sim_idxs})
        # print("dissim_vals:", {spec["col"][i] for i in dissim_idxs})
        return dissim_idxs, sim_idxs

    def _find_categorical_matches(
        self,
        anchor_val: Any,
        cond: SimilarityCondition,
        spec: dict[str, Any],
    ) -> tuple[set[int], set[int]]:
        """
        Identify exact-match categorical similarities.

        Description:
            Matches candidate samples whose categorical values equal the
            anchor value. Similarity conditions determine whether the
            match or its complement constitutes the "similar" set.

        Args:
            anchor_val (Any):
                Categorical value for the anchor sample.
            cond (SimilarityCondition):
                Similarity rule describing match mode.
            spec (dict[str, Any]):
                Column specification containing a category mapping.

        Returns:
            tuple[set[int], set[int]]:
                (matched_relative_indices, non_matched_relative_indices)

        Raises:
            ValueError:
                If the column is numeric or if category groups overlap.

        """
        if not spec["categorical"]:
            raise ValueError("Cannot compute categorical match on numeric data.")

        vals = spec["col"]
        cat_map = spec["cat_map"]
        sim_idxs = set(cat_map.get(anchor_val, []))
        dissim_idxs = set(range(len(vals))) - sim_idxs

        overlap = set.intersection(sim_idxs, dissim_idxs)
        if len(overlap):
            msg = f"Samples detected in both matching and non-match groups: {overlap}"
            raise ValueError(msg)

        # Return (matches, non-matches)
        if cond.mode == "similar":
            return sim_idxs, dissim_idxs
        return dissim_idxs, sim_idxs

    def _compute_scores(
        self,
        anchor_idx: int,
        cand_list: list[int],
        specs: list[dict[str, Any]],
    ) -> np.ndarray[float]:
        """
        Compute per-candidate similarity scores across all conditions.

        Description:
            Each candidate sample's score is the mean of the scores
            returned by each SimilarityCondition associated with the
            role. Scores are computed using the anchor and candidate
            values extracted from the resolved column specifications.

        Args:
            anchor_idx (int):
                Relative index of the anchor sample.
            cand_list (list[int]):
                Relative indices for candidate samples.
            specs (list[dict[str, Any]]):
                Column/condition specifications for the role.

        Returns:
            np.ndarray[float]:
                Array of mean similarity scores, one per candidate.

        """
        out = []
        for p_idx in cand_list:
            per_cond_score = []
            for spec in specs:
                cond: SimilarityCondition = spec["cond"]
                a_val = spec["col"][anchor_idx]
                b_val = spec["col"][p_idx]
                per_cond_score.append(cond.score(a_val, b_val))
            out.append(float(np.mean(per_cond_score)))
        return np.array(out)

    def _precompute_numeric_sorted(self, arr: NDArray) -> SortedColumn:
        """
        Precompute sorted order for a numeric array.

        Description:
            Sorts a numeric 1D array and records both the sorted values
            and the permutation mapping from sorted indices back to the
            original array positions. Used for fast numeric range search.

        Args:
            arr (NDArray):
                Numeric 1D array.

        Returns:
            SortedColumn:
                Dataclass holding sorted values and index mappings.

        """
        order = np.argsort(arr)
        return SortedColumn(values=arr[order], original_to_sorted_idxs=order)

    def _resolve_columns_for_role(
        self,
        view: FeatureSetView,
        conds,
    ) -> list[dict[str, Any]]:
        """
        Resolve and prepare column specifications for a single role.

        Description:
            Each role may reference multiple columns using string keys.
            This method resolves those references using DataReference,
            extracts the underlying arrays, validates dimensionality,
            and prepares structures needed for similarity evaluation
            (sorted numeric columns or categorical index maps).

        Args:
            view (FeatureSetView):
                Bound FeatureSetView containing all samples.
            conds (dict[str, SimilarityCondition]):
                Mapping of column identifiers to similarity rules.

        Returns:
            list[dict[str, Any]]:
                A list of column specifications, each including:
                    - resolved DataReference
                    - raw column values
                    - condition object
                    - sorted structure or categorical mapping
                    - datatype flags

        Raises:
            ValueError:
                If column resolution fails, or if data are not 1D.

        """
        # Resolve columns for each condition
        specs: list[dict[str, Any]] = []
        for key_str, cond in conds.items():
            # Infer node/domain/key/rep from string
            ref = FeatureSetColumnReference.from_string(
                val=key_str,
                known_attrs={
                    "node_label": view.source.label,
                    "node_id": view.source.node_id,
                },
                experiment=ExperimentContext.get_active(),
            )
            # Gather actual data in the column defined by `ref`
            col: np.ndarray = view.get_data(
                columns=f"{ref.domain}.{ref.key}.{ref.rep}",
                fmt=DataFormat.NUMPY,
            )

            # Currenlty only support conditions over 1-dimensional data
            col = np.squeeze(col)  # remove singleton dimensions
            if col.ndim != 1:
                msg = (
                    "SimilarityConditions can only be applied to one-dimensional data. "
                    f"Column '{key_str}' has ndim={col.ndim}."
                )
                raise ValueError(msg)

            # We utilize different selection algorithms for numeric and categorical data
            # If numeric, we use a sorted array + binary search
            if np.issubdtype(col.dtype, np.number):
                sorted_col = self._precompute_numeric_sorted(col)
                spec = {
                    "key": key_str,
                    "ref": ref,
                    "cond": cond,
                    "col": col,
                    "sorted": sorted_col,
                    "categorical": False,
                }
            # For categorical, we use binning of unique values
            else:
                # Build reverse lookup: value -> set of relative indices with same value
                cat_map = {}
                for i, v in enumerate(col):
                    cat_map.setdefault(v, []).append(i)
                spec = {
                    "key": key_str,
                    "ref": ref,
                    "cond": cond,
                    "col": col,
                    "cat_map": cat_map,
                    "categorical": True,
                }

            specs.append(spec)

        return specs

    def _generate_role_matches(self, view: FeatureSetView, specs: list[dict[str, Any]]):
        """
        Generate matched samples for a single role.

        Description:
            Iterates over all anchor samples in the view. For each anchor:
                1. Identifies per-condition match/non-match sets.
                2. Applies strict or fallback behavior to prune candidates.
                3. Computes full/partial/non-match groups.
                4. Selects up to ``max_samples_per_anchor`` matches in
                   group priority order.
                5. Computes similarity scores per selected candidate.

            Returns three parallel lists containing:
                - anchor absolute indices
                - matched sample absolute indices
                - per-match scores

        Args:
            view (FeatureSetView):
                Bound FeatureSetView containing samples.
            specs (list[dict[str, Any]]):
                List of column specifications for conditions.

        Returns:
            tuple[list[int], list[int], list[float]]:
                Parallel lists of anchor indices, matched indices,
                and match scores.

        Raises:
            ValueError:
                If a role attempts to pair a sample with itself.

        """
        abs_indices = view.indices

        # Store selected pairings (anchor + role store relative indices, score stores scores of pairing)
        anchor_abs_idxs: list[int] = []
        role_abs_idxs: list[int] = []
        role_scores: list[float] = []

        for rel_idx in range(len(view)):
            # Collect matches per condition
            per_cond_matches = []
            per_cond_fallbacks = []

            # If a condition does not allow fallback, any sample that do not match
            # that condition cannot be used in other conditions. We use `blocked_rel_idxs`
            # to determine which samples cannot be used
            blocked_rel_idxs: set[int] = set()

            for spec in specs:
                cond: SimilarityCondition = spec["cond"]
                anchor_val = spec["col"][rel_idx]

                # Find matches & non-matches bases on data type
                if not spec["categorical"]:
                    matches, non_matches = self._find_numeric_matches(
                        anchor_val=anchor_val,
                        cond=cond,
                        spec=spec,
                    )
                else:
                    matches, non_matches = self._find_categorical_matches(
                        anchor_val=anchor_val,
                        cond=cond,
                        spec=spec,
                    )

                # Remove any blocked idxs and anchor
                matches -= blocked_rel_idxs
                matches.discard(rel_idx)
                per_cond_matches.append(matches)

                # Check if fallback is allowed for this cond
                if cond.allow_fallback:
                    per_cond_fallbacks.append(non_matches)
                else:
                    blocked_rel_idxs |= non_matches

            # Compute the set of indices that are globally allowed after
            # enforcing all strict (no-fallback) conditions.
            allowed_idxs: set[int] = (
                set(range(len(view))) - blocked_rel_idxs - {rel_idx}
            )
            if not allowed_idxs:
                # No valid candidates for this anchor
                continue

            # Combine across all conditions
            full_matches: set[int] = set.intersection(*per_cond_matches) & allowed_idxs
            partial_matches: set[int] = (
                set.union(*per_cond_matches) & allowed_idxs
            ) - full_matches
            non_matches: set[int] = allowed_idxs - full_matches - partial_matches

            # Determine number of pairing to select
            n_select: int = self.max_samples_per_anchor or (len(view) - 1)

            selected_pairs: list[int] = []
            selected_scores: list[float] = []

            # Search in following order: full_matches -> partial -> non
            for group in (full_matches, partial_matches, non_matches):
                if not group:
                    continue

                # Convert set to list for stable processing
                group_list: list[int] = list(group)

                # Perform full scoring only if requested (expensive)
                if self.choose_best_only:
                    scores: np.ndarray[float] = self._compute_scores(
                        anchor_idx=rel_idx,
                        cand_list=group_list,
                        specs=specs,
                    )
                    order = np.flip(np.argsort(scores))  # higher score = better match
                    group_selected = [group_list[j] for j in order[:n_select]]
                    group_scores = [scores[j] for j in order[:n_select]]
                # Otherwise, use random selection
                else:
                    self.rng.shuffle(group_list)
                    group_selected = group_list[:n_select]
                    group_scores = self._compute_scores(
                        anchor_idx=rel_idx,
                        cand_list=group_selected,
                        specs=specs,
                    )

                # Record selection
                selected_pairs.extend(list(group_selected))
                selected_scores.extend(list(group_scores))

                # Check if need to select from next group
                n_select -= len(group_selected)
                if n_select <= 0:
                    break

            # Records pairing for this single anchor
            for p_rel_idx, score in zip(selected_pairs, selected_scores, strict=True):
                if p_rel_idx == rel_idx:
                    msg = f"Pairing between the same sample is not allowed: {rel_idx} == {p_rel_idx}"
                    raise ValueError(msg)

                anchor_abs_idxs.append(abs_indices[rel_idx])
                role_abs_idxs.append(abs_indices[p_rel_idx])
                role_scores.append(score)

            # Update progress bar
            self._progress_task.tick(n=1)

        return anchor_abs_idxs, role_abs_idxs, role_scores

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
        cfg.update(
            {
                "sampler_name": "NSampler",
                "condition_mapping": self.condition_mapping,
                "max_samples_per_anchor": self.max_samples_per_anchor,
                "choose_best_only": self.choose_best_only,
            },
        )
        return cfg

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> NSampler:
        """
        Construct a sampler from configuration.

        Args:
            config (dict[str, Any]): Sampler configuration.

        Returns:
            NSampler: Unfitted sampler instance.

        """
        if ("sampler_name" not in config) or (config["sampler_name"] != "NSampler"):
            raise ValueError("Invalid config for NSampler.")

        return cls(
            condition_mapping=config["condition_mapping"],
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
