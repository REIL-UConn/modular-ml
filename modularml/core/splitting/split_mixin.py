from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from modularml.context.experiment_context import ExperimentContext
from modularml.core.data.schema_constants import (
    DOMAIN_FEATURES,
    DOMAIN_SAMPLE_ID,
    DOMAIN_TAGS,
    DOMAIN_TARGETS,
)
from modularml.core.references.experiment_reference import ResolutionError
from modularml.core.references.featureset_reference import (
    FeatureSetColumnReference,
    FeatureSetSplitReference,
)
from modularml.core.splitting.splitter_record import SplitterRecord
from modularml.utils.data.data_format import DataFormat
from modularml.utils.data.pyarrow_data import resolve_column_selectors
from modularml.utils.io.cloning import clone_via_serialization
from modularml.utils.logging.warnings import warn

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from modularml.core.data.featureset import FeatureSet
    from modularml.core.data.featureset_view import FeatureSetView
    from modularml.core.splitting.base_splitter import BaseSplitter


class SplitMixin:
    """
    Shared mixin providing split-related functionality for both FeatureSet and FeatureSetView.

    Description:
        This mixin unifies split operations across the FeatureSet hierarchy. It allows \
        either a `FeatureSet` or `FeatureSetView` to invoke high-level splitting \
        methods (`split`, `split_random`, `split_by_condition`) without code duplication.

        - When called from a **FeatureSet**, the operation targets the entire \
          SampleCollection.
        - When called from a **FeatureSetView**, the operation applies only to the \
          subset of samples represented by that view.

        Resulting splits are represented as `FeatureSetView` objects and may optionally \
        be registered into the parent FeatureSet's `_splits` and `_split_configs` \
        registries when invoked from a FeatureSet.
    """

    # ================================================
    # Context resolution
    # ================================================
    def _get_split_context(self) -> tuple[FeatureSet, bool]:
        """
        Identify the FeatureSet context for a split operation.

        Description:
            Determines the parent FeatureSet and whether the calling instance is a \
            FeatureSetView or a FeatureSet. This allows the same split logic to \
            adapt to both types of objects.

            - If called on a `FeatureSet`, the method returns the instance itself as \
              the source.
            - If called on a `FeatureSetView`, the method returns the parent FeatureSet \
              (`.source`).
            - If the caller is neither, a `TypeError` is raised.

        Returns:
            tuple[FeatureSet, bool]:
                A 2-tuple containing:
                - `FeatureSet`: Reference to the source FeatureSet instance.
                - `bool`: `True` if the caller is a FeatureSetView, `False` otherwise.

        Raises:
            TypeError:
                If the caller is neither a FeatureSet nor a FeatureSetView.

        """
        from modularml.core.data.featureset import FeatureSet
        from modularml.core.data.featureset_view import FeatureSetView

        # Case 1: called on FeatureSet
        if isinstance(self, FeatureSet):
            source = self
            return source, False

        # Case 2: called on FeatureSetView
        if isinstance(self, FeatureSetView):
            source = self.source
            return source, True

        # Case 3: unknown caller
        msg = f"{self.__class__.__name__} is not a valid split context host. Expected FeatureSet or FeatureSetView."
        raise TypeError(msg)

    # ================================================
    # Row & column filtering
    # ================================================
    def take(
        self,
        rel_indices: Sequence[int],
        label: str | None = None,
    ) -> FeatureSetView:
        """
        Create a new FeatureSetView derived from this one using relative indices.

        Description:
            Produces a new view referencing the same collection and parent \
            FeatureSet but restricted to a subset of rows. The provided indices \
            are interpreted *relative to this view* and mapped to *absolute* \
            indices in the underlying collection.

        Args:
            rel_indices (Sequence[int]):
                Relative indices of the current FeaturSet/View to project into
                a new view.

            label (str, optional):
                The label to assign to the returned FeatureSetView.

        """
        from modularml.core.data.featureset_view import FeatureSetView

        # Get indices of the calling class
        source, is_view = self._get_split_context()
        idxs = self.indices if is_view else np.arange(source.n_samples)

        rel_indices = np.asarray(rel_indices)
        if rel_indices.ndim != 1:
            raise ValueError("rel_indices must be a 1D sequence of integer positions.")
        if np.any(rel_indices >= len(idxs)):
            msg = "Some relative indices exceed the size of the current view."
            raise IndexError(msg)
        abs_indices = np.asarray(idxs)[rel_indices]

        # Preserve columns if already a view
        if is_view:
            return FeatureSetView(
                source=source,
                indices=abs_indices,
                columns=self.columns,
                label=label or f"{self.label}_view",
            )
        return FeatureSetView(
            source=source,
            indices=abs_indices,
            columns=source.get_all_keys(
                include_domain_prefix=True,
                include_rep_suffix=True,
            ),
            label=label or f"{self.label}_view",
        )

    def filter(
        self,
        *,
        conditions: dict[str | FeatureSetColumnReference, Any | list[Any], Callable],
        label: str | None = None,
    ) -> FeatureSetView | None:
        """
        Create a filtered FeatureSetView using reference-aware conditions.

        Description:
            Applies row-level filtering using fully-qualified column references.
            Each condition is evaluated independently using its own representation,
            and all resulting masks are AND-composed.

            Conditions may reference:
                - features.<key>.<rep>
                - targets.<key>.<rep>
                - tags.<key>.<rep>

        Args:
            conditions (dict[str, Any | list[Any], Callable]):
                Mapping of column names (from any domain) to filter criteria.
                Values may be:
                - `scalar`: selects rows where the column equals the value.
                - `sequence`: selects rows where the column is in the given list/set.
                - `callable`: a function that takes a NumPy array and returns a boolean mask.
            label (str, optional):
                The label to assign to the returned FeatureSetView.

        Returns:
            FeatureSetView:
                A view of this FeatureSet containing only rows that satisfy \
                all specified conditions. If no rows match, an empty view is returned.

        Raises:
            KeyError:
                If a specified key does not exist in any of the domains.
            TypeError:
                If a condition value type is unsupported.

        Example:
            For a FeatureSet where its samples have the following attributes:
            - FeatureSet.tag_keys() -> 'cell_id', 'group_id', 'pulse_type'
            - FeatureSet.feature_keys() -> 'voltage', 'current',
            - FeatureSet.target_keys() -> 'soh'

            a filter condition can be applied such that:

            - `cell_id` is in [1, 2, 3]
            - `group_id` is greater than 1, and
            - `pulse_type` equals 'charge'.

        ``` python
        FeatureSet.filter(where={
            "tags.cell_id"=[1,2,3],
            "tags.group_id"=(lambda x: x > 1),
            "tags.pulse_type"="charge",
        })
        ```

        """
        from modularml.core.data.featureset_view import FeatureSetView

        def _evaluate_single_condition(col_data: np.ndarray, cond) -> np.ndarray:
            """
            Evaluate a filter condition on a column of scalar or array values.

            Args:
                col_data (np.ndarray | Sequence[Any]):
                    Column values (scalars or arrays).
                cond (Any | list | tuple | set | callable):
                    The condition to evaluate:
                    - callable(x) -> bool
                    - iterable of allowed values
                    - scalar literal for equality

            Returns:
                np.ndarray[bool]: Boolean mask indicating rows satisfying the condition.

            """
            col_data = np.asarray(col_data, dtype=object)

            # 1. Callable condition
            if callable(cond):
                # Detect scalar vs array-valued column
                first_val = col_data[0]
                if isinstance(first_val, (np.ndarray, list, tuple)):
                    # Apply row-wise for arrays
                    mask = np.fromiter(
                        (bool(cond(x)) for x in col_data),
                        dtype=bool,
                        count=len(col_data),
                    )
                else:
                    # Try applying directly to the vector
                    try:
                        mask = np.asarray(cond(col_data))
                        if np.ndim(mask) == 0:  # single scalar result
                            mask = np.full(len(col_data), bool(mask))
                    except Exception as e:
                        msg = f"Failed to apply callable condition. {e}"
                        raise ValueError(msg) from e
                return mask

            # 2. Iterable of allowed values
            if isinstance(cond, (list, tuple, set, np.ndarray)):
                # For scalar columns -> vectorized np.isin
                first_val = col_data[0]
                if not isinstance(first_val, (np.ndarray, list, tuple)):
                    return np.isin(col_data, cond, assume_unique=False)

                # Array-like row values are not supported yet
                ## This commented out below returns True is any value in the row value is in the provided
                ## condition value list. Not sure this is what is expected. Likely want an exact match?
                # # For array-valued rows -> True if *any element* in the row is in cond
                # allowed = set(cond)
                # mask = np.fromiter(
                #     (any(elem in allowed for elem in x) if x is not None else False for x in col_data),
                #     dtype=bool,
                #     count=len(col_data),
                # )
                msg = "Iterable condition can only be applied to 1-dimensional columns."
                raise TypeError(msg)

            # 3. Scalar equality condition
            # For scalar columns -> vectorized equality
            first_val = col_data[0]
            if not isinstance(first_val, (np.ndarray, list, tuple)):
                return np.asarray(col_data) == cond

            # Array-like row values are not supported yet
            ## This commented out below returns True is any value in the row value is equal to the provided
            ## condition scalar value. Not sure this is what is expected.
            # # For array-valued rows -> True if any element equals the scalar
            # mask = np.fromiter(
            #     (np.any(np.asarray(x) == cond) if x is not None else False for x in col_data),
            #     dtype=bool,
            #     count=len(col_data),
            # )
            msg = (
                "Scalar-valued conditions can only be applied to 1-dimensional columns."
            )
            raise TypeError(msg)

        source, is_view = self._get_split_context()
        collection = source.collection

        # Start with all True mask
        mask = np.ones(collection.n_samples, dtype=bool)

        for arg, cond in conditions.items():
            # Try to cast ref to FeatureSetColumnReference
            if isinstance(arg, str):
                ref = FeatureSetColumnReference.from_string(
                    val=arg,
                    known_attrs={
                        "node_id": source.node_id,
                        "node_label": source.label,
                    },
                    experiment=ExperimentContext.get_active(),
                )
            elif isinstance(arg, FeatureSetColumnReference):
                ref = arg
            else:
                msg = (
                    "Filter conditions must be keyed by a string or FeatureSetColumnReference object. "
                    f"Received key of type {type(ref)}"
                )
                raise TypeError(msg)

            # Validate FeatureSetColumnReference values
            try:
                _ = ref.resolve()
            except ResolutionError as e:
                msg = f"Filter condition ({ref}) could not be resolved. {e}."
                raise ValueError(msg) from e

            # Retrieve column data explicitly
            col_data = collection._get_rep_data(
                domain=ref.domain,
                key=ref.key,
                rep=ref.rep,
                fmt=DataFormat.NUMPY,
            )
            cond_mask = _evaluate_single_condition(col_data=col_data, cond=cond)
            mask &= cond_mask.reshape(collection.n_samples)

        # Restrict mask to current view's indices
        if is_view:
            local_mask = mask[self.indices]
            selected_indices = self.indices[np.where(local_mask)[0]]
        else:
            selected_indices = np.where(mask)[0]

        if len(selected_indices) == 0:
            msg = f"No samples match filter conditions: {list(conditions.keys())}"
            warn(msg, UserWarning, stacklevel=2)

        # Build FeatureSetView using indices
        return FeatureSetView(
            source=source,
            indices=selected_indices,
            columns=source.get_all_keys(
                include_domain_prefix=True,
                include_rep_suffix=True,
            ),
            label=label or f"{self.label}_view",
        )

    def select(
        self,
        columns: str | list[str] | None = None,
        *,
        features: str | list[str] | None = None,
        targets: str | list[str] | None = None,
        tags: str | list[str] | None = None,
        rep: str | None = None,
        label: str | None = None,
    ) -> FeatureSetView:
        """
        Select a subset of columns from this FeatureSet and return a FeatureSetView.

        Description:
            Performs declarative column selection using a shared selector syntax
            across FeatureSet, FeatureSetView, and DataReference creation.

            Selection supports:
                - Explicit fully-qualified columns (e.g. "features.voltage.raw")
                - Domain-based selectors via `features`, `targets`, and `tags`
                - Wildcards (e.g. "*.raw", "voltage.*", "*.*")
                - Automatic domain prefixing ("voltage.raw" → "features.voltage.raw")
                - Optional default representation inference via `rep`

            No data is copied. The returned object is a lightweight view that
            references the same underlying SampleCollection.

        Args:
            columns (str | list[str] | None):
                Fully-qualified column names to include
                (e.g. "features.voltage.raw"). These must exactly match existing
                columns in the FeatureSet.

            features (str | list[str] | None):
                Feature-domain selectors. May be bare keys ("voltage"),
                key/rep pairs ("voltage.raw"), or wildcards.
                The "features." prefix may be omitted.

            targets (str | list[str] | None):
                Target-domain selectors, following the same rules as `features`.

            tags (str | list[str] | None):
                Tag-domain selectors, following the same rules as `features`.

            rep (str | None):
                Default representation suffix to apply when a selector omits a
                representation. Explicit representations are never overridden.

            label (str, optional):
                The label to assign to the returned FeatureSetView.

        Returns:
            FeatureSetView:
                A row-preserving, column-filtered view over this FeatureSet.

        """
        from modularml.core.data.featureset_view import FeatureSetView

        source, is_view = self._get_split_context()

        # Extract real columns from collection
        all_cols: list[str] = source.get_all_keys(
            include_domain_prefix=True,
            include_rep_suffix=True,
        )

        # Build final column selection (organized by domain)
        selected: dict[str, set[str]] = resolve_column_selectors(
            all_columns=all_cols,
            columns=columns,
            features=features,
            targets=targets,
            tags=tags,
            rep=rep,
            include_all_if_empty=False,
        )

        # Order columns: features -> targets -> tags -> sample_id
        sel_cols: list[str] = []
        for d in [DOMAIN_FEATURES, DOMAIN_TARGETS, DOMAIN_TAGS]:
            if d in selected:
                sel_cols.extend(sorted(selected[d]))
        sel_cols.append(DOMAIN_SAMPLE_ID)

        # Maintain same indices of the calling class
        mask = np.ones(source.n_samples, dtype=bool)
        # Restrict mask to current view's indices
        if is_view:
            local_mask = mask[self.indices]
            selected_indices = self.indices[np.where(local_mask)[0]]
        else:
            selected_indices = np.where(mask)[0]

        # Return new view with filtered columns
        return FeatureSetView(
            source=source,
            indices=selected_indices,
            columns=list(sel_cols),
            label=label or f"{self.label}_view",
        )

    def take_intersection(
        self: FeatureSet | FeatureSetView,
        other: FeatureSet | FeatureSetView,
        *,
        order: Literal["self", "other"] = "self",
        label: str | None = None,
    ) -> FeatureSetView:
        """
        Return a view containing rows of `self` that also appear in `other`.

        Description:
            Takes the intersection of this view's indices and anothers.
            The columns of the calling view are preserved.

        Args:
            other (FeatureSet | FeatureSetView):
                Other indice-containing FeatureSet object.
            order (str):
                Whether the intersecting view should use the indice order defined
                in `'self'` or `'other'`. Defaults to `'self'`.
            label (str, optional):
                Label assigned to the returned FeatureSetView.
                Defaults to "<self.label>_intersection".

        """
        if not isinstance(other, SplitMixin):
            msg = f"Intersection is only possible between FeatureSets or FeatureSetViews. Received: {type(other)}."
            raise TypeError(msg)

        # Get indices of caller
        self_src, self_is_view = self._get_split_context()
        self_idxs = self.indices if self_is_view else np.arange(self_src.n_samples)

        # Get indices of other
        other_src, other_is_view = other._get_split_context()
        other_idxs = (
            other.indices if other_is_view else np.arange(other.collection.n_samples)
        )

        # Ensure both are from the same source
        if self_src is not other_src:
            msg = (
                "Cannot perform intersection on two views that reference different "
                f"FeatureSets. {self_src!r} != {other_src!r}."
            )
            raise ValueError(msg)

        # Map self idxs to ordering idx
        abs_to_rel = {abs_idx: rel_idx for rel_idx, abs_idx in enumerate(self_idxs)}

        if order == "self":
            common_abs = np.intersect1d(self_idxs, other_idxs)
        else:
            abs_set = set(self_idxs)
            common_abs = [i for i in other_idxs if i in abs_set]

        rel = [abs_to_rel[i] for i in common_abs]
        return self.take(rel, label=label or f"{self.label}_intersection")

    def take_difference(
        self: FeatureSet | FeatureSetView,
        other: FeatureSet | FeatureSetView,
        *,
        label: str | None = None,
    ) -> FeatureSetView:
        """
        Return a view containing rows of `self` that do not appear in `other`.

        Description:
            Computes the set difference between this object and another
            FeatureSet or FeatureSetView. Any sample whose absolute index
            appears in `other` will be removed from the returned view.

            The resulting view preserves:
                - The column selection of the calling object
                - The ordering of `self`

            This operation is equivalent to a set subtraction:
                `result = self - other`

        Args:
            other (FeatureSet | FeatureSetView):
                The object whose samples should be removed from this one.

            label (str, optional):
                Label assigned to the returned FeatureSetView.
                Defaults to "<self.label>_difference".

        Returns:
            FeatureSetView:
                A new view containing only samples from `self`
                that do not overlap with `other`.

        Raises:
            TypeError:
                If `other` is not a FeatureSet or FeatureSetView.

        """
        if not isinstance(other, SplitMixin):
            msg = (
                "Difference is only possible between FeatureSets or "
                f"FeatureSetViews. Received: {type(other)}."
            )
            raise TypeError(msg)

        # Get indices of caller
        self_src, self_is_view = self._get_split_context()
        self_idxs = self.indices if self_is_view else np.arange(self_src.n_samples)

        # Get indices of other
        other_src, other_is_view = other._get_split_context()
        other_idxs = other.indices if other_is_view else np.arange(other_src.n_samples)

        # Ensure both are from the same source
        if self_src is not other_src:
            msg = (
                "Cannot perform difference on two views that reference different "
                f"FeatureSets. {self_src!r} != {other_src!r}."
            )
            raise ValueError(msg)

        # Build set of indices to remove
        other_set = set(other_idxs)

        # Preserve self ordering
        remaining_abs = [i for i in self_idxs if i not in other_set]

        # Map absolute -> relative indices
        abs_to_rel = {abs_idx: rel_idx for rel_idx, abs_idx in enumerate(self_idxs)}
        rel = [abs_to_rel[i] for i in remaining_abs]

        return self.take(
            rel_indices=rel,
            label=label or f"{self.label}_difference",
        )

    def take_sample_uuids(
        self,
        sample_uuids: Sequence[str],
        *,
        label: str | None = None,
    ) -> FeatureSetView:
        """
        Create a new FeatureSetView over the specified samples.

        Description:
            Produces a new view referencing the same collection and parent
            FeatureSet, but is restricted to a subset of the rows. Only
            rows with matching sample UUIDs to those specified are included
            in the returned view. All sample UUIDs must exist in the caller's
            collection.

        Args:
            sample_uuids (Sequence[str]):
                A list of sample UUIDs to include in the returned view.
                The order is preserved in the returned view.

            label (str, optional):
                The label to assign to the returned FeatureSetView.

        """
        # Get indices of the calling class
        source, is_view = self._get_split_context()
        idxs = self.indices if is_view else np.arange(source.n_samples)

        # Get existing sample UUIDs of caller
        all_sids = source.get_sample_uuids(fmt=DataFormat.NUMPY)[idxs]

        # Get rel indices of subsamples
        take_sids = np.asarray(sample_uuids, dtype=str)
        rel_idxs = np.where(all_sids[:, None] == take_sids)[0]
        return self.take(rel_indices=rel_idxs, label=label)

    # ================================================
    # Split methods
    # ================================================
    def split(
        self,
        splitter: BaseSplitter,
        *,
        return_views: bool = False,
        register: bool = True,
    ) -> list[FeatureSetView] | None:
        """
        Apply a splitter to this FeatureSe.

        Description:
            Runs the provided `BaseSplitter` instance on a view of the caller. \
            The resulting splits (and splitter config) are optionally registered \
            into the source FeatureSet's `_splits` registry.

        Args:
            splitter (BaseSplitter):
                The splitter instance (e.g., RandomSplitter, ConditionSplitter).
            return_views (bool, optional):
                Whether to return FeatureSetViews or not. Defaults to False.
            register (bool, optional):
                Whether to record the resulting views and splitter config. \
                Defaults to True.

        Returns:
            list[FeatureSetView] | None:
                The created splits are returned only if `return_views=True`.

        """
        # Get context of instance calling this mixin
        source, is_view = self._get_split_context()

        # Choose base view (use self if called on FeatureSetView, otherwise convert FeatureSet to a view)
        base_view = self if is_view else source.to_view()
        if ".fold_" in base_view.label:
            msg = "Splitting of a fold (`base_view.label`) is not supported."
            raise NotImplementedError(msg)

        # Perform the split (return splits as FeatureSetView instances)
        results: dict[str, FeatureSetView] = splitter.split(
            base_view,
            return_views=True,
        )

        # Register splits if requested
        if register:
            # Record FeatureSetView as new split
            for split in list(results.values()):
                source.add_split(split)

            # Record this splitter configuration
            # Splitter is cloned to prevent user from modifying state outside of ModularML
            cloned_splitter: BaseSplitter = clone_via_serialization(obj=splitter)
            order = (
                max([rec.order for rec in source._split_recs]) + 1
                if len(source._split_recs) > 0
                else 0
            )
            rec = SplitterRecord(
                order=order,
                splitter=cloned_splitter,
                applied_to=FeatureSetSplitReference(
                    node_label=source.label,
                    node_id=source.node_id,
                    split_name=self.label if is_view else None,
                ),
                produced_splits=list(results.keys()),
            )
            source._split_recs.append(rec)

        # Return views if requested
        if return_views:
            return results
        return None

    def split_random(
        self,
        ratios: Mapping[str, float],
        *,
        group_by: str | Sequence[str] | None = None,
        seed: int = 13,
        return_views: bool = False,
        register: bool = True,
    ) -> list[FeatureSetView] | None:
        """
        Randomly partition this FeatureSet (or FeatureSetView) into subsets.

        Description:
            A convenience wrapper around :class:`RandomSplitter`, which randomly divides \
            the samples of the specified collection into multiple subsets according to \
            user-defined ratios (e.g., `{"train": 0.8, "val": 0.2}`).

            Optionally, one or more tag keys can be provided via `group_by` to ensure \
            that all samples sharing the same tag values (e.g., a common cell ID or batch ID) \
            are placed into the same subset.

        Args:
            ratios (Mapping[str, float]):
                Dictionary mapping subset labels to relative ratios.
                Must sum to 1.0. Example: `{"train": 0.7, "val": 0.2, "test": 0.1}`.
            group_by (str | Sequence[str] | None, optional):
                One or more tag keys to group samples by before splitting.
                If `None`, samples are split individually.
            seed (int, optional):
                Random seed for reproducibility. Defaults to 13.
            return_views (bool, optional):
                Whether to return the resulting FeatureSetViews. Defaults to `False`.
            register (bool, optional):
                Whether to register the resulting splits and splitter configuration in
                `FeatureSet._splits` for future reference. Defaults to `True`.

        Returns:
            list[FeatureSetView] | None:
                The resulting FeatureSetViews (if `return_views=True`).
                Otherwise, returns `None`.

        Example:
            ```python
            fs.split_random(
                ratios={"train": 0.8, "val": 0.2},
                group_by="cell_id",
                seed=42,
            )
            ```

        """
        from modularml.splitters.random_splitter import RandomSplitter

        splitter = RandomSplitter(ratios, group_by=group_by, seed=seed)
        return self.split(
            splitter,
            return_views=return_views,
            register=register,
        )

    def split_by_condition(
        self,
        conditions: dict[str, dict[str, Any]],
        *,
        return_views: bool = False,
        register: bool = True,
    ) -> list[FeatureSetView] | None:
        """
        Split this FeatureSet (or FeatureSetView) based on logical conditions.

        Description:
            A convenience wrapper around :class:`ConditionSplitter`, which partitions \
            samples into subsets based on user-defined filter expressions.

            Each subset is defined by a dictionary mapping feature, target, or tag keys \
            to condition values, which may be:
            - A literal value for equality matching.
            - A list, tuple, or set of allowed values.
            - A callable predicate ``f(x) -> bool`` that returns a boolean mask.

            For example:
            ```python
            fs.split_by_condition(
                {
                    "low_temp": {"temperature": lambda x: x < 20},
                    "high_temp": {"temperature": lambda x: x >= 20},
                    "cell_5": {"cell_id": 5},
                }
            )
            ```

            **Note:** Overlapping subsets are permitted, but a warning will be issued \
            if any sample satisfies multiple conditions.

        Args:
            conditions (Mapping[str, Mapping[str, Any | Sequence | Callable]]):
                Mapping of subset labels → condition dictionaries. \
                Each condition dictionary maps a key (feature/target/tag name) \
                to a condition (scalar, sequence, or callable).
            return_views (bool, optional):
                Whether to return the resulting FeatureSetViews. Defaults to `False`.
            register (bool, optional):
                Whether to register the resulting splits and splitter configuration in \
                `FeatureSet._splits` for future reference. Defaults to `True`.

        Returns:
            list[FeatureSetView] | None:
                The resulting FeatureSetViews (if `return_views=True`). \
                Otherwise, returns `None`.

        Example:
            ```python
            fs.split_by_condition(
                {
                    "train": {"cell_type": "A"},
                    "test": {"cell_type": "B"},
                }
            )
            ```

        """
        from modularml.splitters.condition_splitter import ConditionSplitter

        splitter = ConditionSplitter(**conditions)
        return self.split(
            splitter,
            return_views=return_views,
            register=register,
        )
