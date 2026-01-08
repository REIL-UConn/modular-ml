from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pyarrow as pa

from modularml.core.data.sample_collection import SampleCollection
from modularml.core.data.sample_collection_mixin import SampleCollectionMixin
from modularml.core.data.sample_schema import validate_str_list
from modularml.core.data.schema_constants import (
    DOMAIN_FEATURES,
    DOMAIN_SAMPLE_ID,
    DOMAIN_TAGS,
    DOMAIN_TARGETS,
    REP_RAW,
    REP_TRANSFORMED,
)
from modularml.core.io.protocols import Configurable, Stateful
from modularml.core.references.experiment_reference import ResolutionError
from modularml.core.references.featureset_reference import FeatureSetColumnReference, FeatureSetReference
from modularml.core.splitting.split_mixin import SplitMixin, SplitterRecord
from modularml.core.topology.graph_node import GraphNode
from modularml.core.transforms.scaler import Scaler
from modularml.core.transforms.scaler_record import ScalerRecord
from modularml.utils.data.comparators import deep_equal
from modularml.utils.data.conversion import flatten_to_2d, to_numpy, unflatten_from_2d
from modularml.utils.data.data_format import DataFormat
from modularml.utils.data.formatting import ensure_list
from modularml.utils.data.pyarrow_data import (
    build_sample_schema_table,
    hash_pyarrow_table,
    resolve_column_selectors,
)
from modularml.utils.errors.exceptions import SplitOverlapWarning
from modularml.utils.io.cloning import clone_via_serialization
from modularml.utils.logging.warnings import warn

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pandas as pd

    from modularml.core.data.featureset_view import FeatureSetView
    from modularml.core.data.sample_schema import SampleSchema


class FeatureSet(GraphNode, SplitMixin, SampleCollectionMixin, Configurable, Stateful):
    """
    Unified FeatureSet backed by a single SampleCollection.

    Each representation (e.g., "raw", "transformed") lives within the same \
    SampleCollection rather than as separate sub-collections. \
    Splits store indices into this collection and may specify \
    which representation(s) to use when retrieving data.
    """

    def __init__(
        self,
        label: str,
        table: pa.Table,
        schema: SampleSchema | None = None,
        **kwargs,
    ):
        super().__init__(label=label, **kwargs)
        # Create SampleCollection attribute
        self.collection: SampleCollection = SampleCollection(table=table, schema=schema)

        # Store splits & spliiter configs
        self._splits: dict[str, FeatureSetView] = {}
        self._split_recs: list[SplitterRecord] = []

        # Store scaler logs/configs
        self._scaler_recs: list[ScalerRecord] = []

    @classmethod
    def from_dict(
        cls,
        label: str,
        data: dict[str, Sequence[Any]],
        feature_keys: str | list[str],
        target_keys: str | list[str],
        tag_keys: str | list[str] | None = None,
    ) -> FeatureSet:
        """
        Construct a FeatureSet from a Python dictionary of column data.

        Description:
            Converts a dictionary of lists/arrays into a column-oriented Arrow table \
            following the ModularML `SampleSchema` convention. Each key in the input \
            dictionary corresponds to a column name, and values are list-like sequences \
            of equal length representing sample data.

            Unlike `from_pandas`, this constructor assumes that each list element \
            already represents a complete sample (i.e., no grouping is applied).

        Args:
        label (str):
            Label to assign to this FeatureSet.
        data (dict[str, Sequence[Any]]):
            A mapping from column names to list-like column data. Each list must \
            have the same length, corresponding to the total number of samples.
        feature_keys (str | list[str]):
            Column name(s) in `data` to be used as features.
        target_keys (str | list[str]):
            Column name(s) in `data` to be used as targets.
        tag_keys (str | list[str] | None, optional):
            Column name(s) corresponding to identifying or categorical metadata \
            (e.g., cell ID, protocol, SOC). Defaults to None.

        Returns:
            FeatureSet:
                A new Arrow-backed FeatureSet containing all provided columns, organized
                into standardized domains (`features`, `targets`, and `tags`).

        Raises:
            ValueError:
                If required keys are missing, or column lengths are inconsistent.

        Example:
            ```python
            fs = FeatureSet.from_dict(
                label="CycleData",
                data={
                    "voltage": [[3.1, 3.2, 3.3], [3.2, 3.3, 3.4]],
                    "current": [[1.0, 1.1, 1.0], [0.9, 1.0, 1.1]],
                    "soh": [0.95, 0.93],
                    "cell_id": ["A1", "A2"],
                },
                feature_keys=["voltage", "current"],
                target_keys=["soh"],
                tag_keys=["cell_id"],
            )
            ```

        """
        # 1. Standardize input args
        feature_keys = ensure_list(feature_keys)
        target_keys = ensure_list(target_keys)
        tag_keys = ensure_list(tag_keys)

        # 2. Ensure data is a np.ndarray for table construction
        feature_data = {k: to_numpy(data[k]) for k in feature_keys}
        target_data = {k: to_numpy(data[k]) for k in target_keys}
        tag_data = {k: to_numpy(data[k]) for k in tag_keys}

        # 3. Build table
        table = build_sample_schema_table(
            features=feature_data,
            targets=target_data,
            tags=tag_data,
        )

        # 4. Return new FeatureSet
        return cls(label=str(label), table=table)

    @classmethod
    def from_pandas(
        cls,
        label: str,
        df: pd.DataFrame,
        feature_cols: str | list[str],
        target_cols: str | list[str],
        groupby_cols: str | list[str] | None = None,
        tag_cols: str | list[str] | None = None,
    ) -> FeatureSet:
        """
        Construct a FeatureSet from a pandas DataFrame (column-wise storage).

        Description:
            Converts a DataFrame into a column-oriented Arrow table that matches the \
            ModularML `SampleSchema` convention. Each domain (features, targets, tags) \
            becomes a Struct column in the final Arrow table.

            If `groupby_cols` are provided, all rows sharing the same group key are \
            aggregated into a single sample row, with feature and target columns \
            stored as array-valued sequences (e.g., np.ndarray or list).

        Args:
            label (str):
                Name to assign to this FeatureSet.
            df (pd.DataFrame):
                Input pandas DataFrame containing raw experimental or measurement data.
            feature_cols (str | list[str]):
                Column name(s) in `df` to be used as feature variables.
            target_cols (str | list[str]):
                Column name(s) in `df` to be used as target variables.
            groupby_cols (str | list[str] | None, optional):
                One or more column names defining group boundaries. Each group becomes \
                one sample (row) in the final table, with grouped columns stored as lists. \
                If None, each original DataFrame row is treated as a sample.
            tag_cols (str | list[str] | None, optional):
                Column name(s) corresponding to identifying or categorical metadata \
                (e.g., cell ID, protocol, SOC). Defaults to None.

        Returns:
            FeatureSet:
                A new Arrow-backed FeatureSet whose table follows the ModularML \
                SampleSchema convention.

        Example:
            ```python
            fs = FeatureSet.from_pandas(
                label="PulseData",
                df=raw_df,
                feature_cols=["voltage", "current"],
                target_cols=["soh"],
                groupby_cols=["cell_id", "cycle_index"],
                tag_cols=["temperature", "cell_id"],
            )
            ```

        """
        # 1. Standardize input args
        feature_cols = ensure_list(feature_cols)
        target_cols = ensure_list(target_cols)
        tag_cols = ensure_list(tag_cols)
        groupby_cols = ensure_list(groupby_cols)

        # 2. Apply grouping, if defined
        if groupby_cols:
            grouped = df.groupby(groupby_cols, sort=False)
        else:
            # Each row is a separate sample -> pseudo group by index
            df = df.copy()
            df["_temp_index"] = df.index
            grouped = df.groupby("_temp_index", sort=False)

        # 3. Aggregate each group into one “row”
        feature_data: dict[str, list] = {c: [] for c in feature_cols}
        target_data: dict[str, list] = {c: [] for c in target_cols}
        tag_data: dict[str, list] = {c: [] for c in tag_cols}

        for _, df_gb in grouped:
            # Convert grouped columns into arrays or scalars
            for c in feature_cols:
                vals = df_gb[c].to_numpy()
                feature_data[c].append(vals if len(vals) > 1 else vals[0])
            for c in target_cols:
                vals = df_gb[c].to_numpy()
                target_data[c].append(vals if len(vals) > 1 else vals[0])
            for c in tag_cols:
                unique_vals = df_gb[c].unique()
                tag_data[c].append(unique_vals[0] if len(unique_vals) == 1 else unique_vals.tolist())

        # 4. Ensure data is a np.ndarray for table construction
        feature_data = {k: to_numpy(v) for k, v in feature_data.items()}
        target_data = {k: to_numpy(v) for k, v in target_data.items()}
        tag_data = {k: to_numpy(v) for k, v in tag_data.items()}

        # 5. Build table
        table = build_sample_schema_table(
            features=feature_data,
            targets=target_data,
            tags=tag_data,
        )

        # 6. Return new FeatureSet
        return cls(label=str(label), table=table)

    from_df = from_pandas

    def __eq__(self, other):
        if not isinstance(other, FeatureSet):
            msg = f"Cannot compare equality between FeatureSet and {type(other)}"
            raise TypeError(msg)

        # Compare collection
        if self.collection != other.collection:
            return False

        # Compare splits
        if set(self.available_splits) != set(other.available_splits):
            return False
        for k in self.available_splits:
            s_split = self.get_split(k)
            o_split = other.get_split(k)
            if set(s_split.indices) != set(o_split.indices):
                return False
            if set(s_split.columns) != set(o_split.columns):
                return False

        # Compare splitter configs
        if len(self._split_recs) != len(other._split_recs):
            return False
        for i in range(len(self._split_recs)):
            if self._split_recs[i].get_config() != other._split_recs[i].get_config():
                return False

        # Compare scaler configs
        if len(self._scaler_recs) != len(other._scaler_recs):
            return False
        s_scaler_recs = sorted(self._scaler_recs, key=lambda x: x.order)
        o_scaler_recs = sorted(other._scaler_recs, key=lambda x: x.order)
        for i in range(len(s_scaler_recs)):
            if not deep_equal(s_scaler_recs[i].get_config(), o_scaler_recs[i].get_config()):
                return False

        return True

    __hash__ = None

    # ================================================
    # GraphNode
    # ================================================
    @property
    def allows_upstream_connections(self) -> bool:
        """FeatureSets do not allow input (upstream) connections."""
        return False

    @property
    def allows_downstream_connections(self) -> bool:
        """FeatureSets do allow output (downstream) connections."""
        return True

    # ================================================
    # SampleCollectionMixin
    # ================================================
    def _resolve_caller_attributes(
        self,
    ) -> tuple[SampleCollection, list[str] | None, np.ndarray | None]:
        return (self.collection, None, None)

    # ================================================
    # FeatureSet Properties & Dunders
    # ================================================
    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, key: str) -> FeatureSetView:
        """Alias for get_split(key)."""
        return self.get_split(key)

    def __repr__(self):
        return f"FeatureSet(label='{self.label}', n_samples={len(self)})"

    def __str__(self):
        return self.__repr__()

    def _summary_rows(self) -> list[tuple]:
        return [
            ("label", self.label),
            ("n_samples", self.n_samples),
            (
                "columns",
                [
                    (
                        DOMAIN_FEATURES,
                        str(self.get_feature_keys(include_domain_prefix=False, include_rep_suffix=True)),
                    ),
                    (
                        DOMAIN_TARGETS,
                        str(self.get_target_keys(include_domain_prefix=False, include_rep_suffix=True)),
                    ),
                    (
                        DOMAIN_TAGS,
                        str(self.get_tag_keys(include_domain_prefix=False, include_rep_suffix=True)),
                    ),
                ],
            ),
            (
                "splits",
                [(name, len(self._splits[name])) for name in self.available_splits],
            ),
            # ("transforms", [(f"{rec.domain}", ", ".join(rec.keys)) for rec in self._scaler_recs]),
            # ("node_id", self.node_id),
        ]

    def to_view(self) -> FeatureSetView:
        """
        Create a FeatureSetView over the entire FeatureSet.

        Returns:
            FeatureSetView: A view referencing all rows.

        """
        from modularml.core.data.featureset_view import FeatureSetView

        return FeatureSetView(
            source=self,
            indices=np.arange(self.collection.n_samples),
            columns=self.get_all_keys(include_domain_prefix=True, include_rep_suffix=True),
            label=f"{self.label}_view",
        )

    @property
    def splits(self) -> dict[str, FeatureSetView]:
        return {k: self.get_split(k) for k in self.available_splits}

    @property
    def n_splits(self) -> int:
        return len(self._splits)

    # ================================================
    # Split Utilities
    # ================================================
    # Most splitting logic is handled in the SplitterMixin class
    @property
    def available_splits(self) -> list[str]:
        """
        All available splits.

        Returns:
            list[str]: Available split names.

        """
        return list(self._splits.keys())

    def get_split(self, split_name: str) -> FeatureSetView:
        """
        Gets the specified split, rebuilding the FeatureSetView if necessary.

        Description:
            If new columns (e.g. transformed representations) have been added to
            the FeatureSet after the split was defined, the cached FeatureSetView
            is re-created to include all current columns while preserving row
            indices.

        Returns:
            FeatureSetView:
                A no-copy, row-wise view of the FeatureSet.

        """
        if split_name not in self._splits:
            msg = f"Split '{split_name}' does not exist. Available: {self.available_splits}"
            raise KeyError(msg)

        # If columns differ, rebuild the view
        view = self._splits[split_name]
        current_cols = self.get_all_keys(include_domain_prefix=True, include_rep_suffix=True)
        if set(view.columns) != set(current_cols):
            from modularml.core.data.featureset_view import FeatureSetView

            view = FeatureSetView(
                source=self,
                indices=view.indices,  # preserve row selection
                columns=current_cols,  # refresh columns
                label=view.label,
            )
        self._splits[split_name] = view

        return view

    def add_split(self, split: FeatureSetView):
        """
        Adds a new FeatureSetView.

        Args:
            split (FeatureSetView): The new view to add.

        """
        # Check that split references this instance and collection exists
        if split.source is not self:
            msg = f"New split `{split.label}` is not a view of this FeatureSet instance."
            raise ValueError(msg)

        # Check that split name is unique
        if split.label is None or split.label in self.available_splits:
            msg = f"Split label ('{split.label}') is missing or already exists."
            raise ValueError(msg)

        # Check that new split name follows naming conventions
        try:
            validate_str_list([*self.available_splits, split.label])
        except ValueError as e:
            msg = f"Failed to add new split `{split.label}`. {e}"
            raise RuntimeError(msg) from e

        # Check overlap with existing splits (only within the same collection)
        for existing_split in self._splits.values():
            if not split.is_disjoint_with(existing_split):
                overlap: list[int] = split.get_overlap_with(existing_split)
                msg = (
                    f"Split '{split.label}' has overlapping samples with existing split '{existing_split.label}' "
                    f"(n_overlap = {len(overlap)}). "
                )
                hint = "Consider checking for disjoint splits or revising your conditions."
                warn(msg, category=SplitOverlapWarning, stacklevel=2, hints=hint)

        # Register new split
        self._splits[split.label] = split

    def clear_splits(self) -> None:
        """Removes all previously defined splits."""
        self._splits.clear()
        self._split_recs.clear()

    # ================================================
    # Transform/Scaling
    # ================================================
    def _cleanup_transformed_rep_if_unused(self) -> None:
        """Remove REP_TRANSFORMED from the SampleCollection for any columns with no scalers."""
        # Get columns that are not used in any scalers
        # They should not have a ".transformed" representation
        all_cols = self.get_all_keys(include_domain_prefix=True, include_rep_suffix=False)
        all_cols.remove(DOMAIN_SAMPLE_ID)
        used_cols: list[str] = []
        for rec in self._scaler_recs:
            for k in rec.keys:
                used_cols.append(f"{rec.domain}.{k}")

        unused_cols = set(all_cols).difference(set(used_cols))
        for col in unused_cols:
            d, k = col.split(".")
            if REP_TRANSFORMED in self.collection._get_rep_keys(domain=d, key=k):
                self.collection.delete_rep(domain=d, key=k, rep=REP_TRANSFORMED)

    def _get_flat_data(
        self,
        domain: str,
        keys: list[str],
        rep: str,
        split: str | None,
        merged_axes: int | tuple[int],
    ) -> tuple[np.ndarray, dict]:
        """Returns new 2D array and metadata of flattening process."""
        # Get data specified by domain + keys + representation
        source = self if split is None else self.get_split(split_name=split)
        cols = [f"{domain}.{k}.{rep}" for k in keys]

        data = source.get_data(
            columns=cols,
            fmt=DataFormat.NUMPY,
            rep=rep,
            include_domain_prefix=True,
            include_rep_suffix=True,
        )
        # Scaler requires 2D array
        # If we need to reshape, check `merged_axes`
        #   - If None, throw error that shapes are >2D, print shapes and state valid
        #     indices to use for merged_axes
        #   - If an integer, preserve the specified axis, and flatten the other(s)
        flat_data, flat_meta = data, {}
        if data.ndim > 2:
            if merged_axes is None:
                # Common use case is 1D data with 1 feature key (eg, shape = (n_sample, 1, n_f))
                # We can just reshape to (n_sample, n_f) without throwing error
                if data.ndim == 3 and len(keys) == 1:
                    merged_axes = (1, 2)
                else:
                    msg = (
                        f"Scalers can only be fit to 2-dimensional data, `{domain}.{keys}` "
                        f"results in data with shape: {data.shape}. "
                        f"Use `merged_axes` to combine the specified axes and flatten the rest."
                    )
                    raise ValueError(msg)
            flat_data, flat_meta = flatten_to_2d(arr=data, merged_axes=merged_axes)

        if data.ndim < 2:
            # Add extra dimension to make 2D
            flat_data, flat_meta = flatten_to_2d(arr=data, merged_axes=0)

        return flat_data, flat_meta

    def fit_transform(
        self,
        scaler: Scaler | Any,
        *,
        domain: str,
        keys: str | list[str] | None = None,
        fit_to_split: str | None = None,
        merged_axes: int | tuple[int] | None = None,
    ):
        """
        Fit a scaler to selected columns and apply the transform to the entire FeatureSet.

        Description:
            - Flattens the selected columns (2D), optionally merging axes (`merged_axes`).
            - Fits the provided scaler to samples from `fit_to_split` (or all samples).
            - Transforms all samples.
            - Unflattens results back to original shapes.
            - Stores a new representation (`REP_TRANSFORMED`) for each key.
            - Appends a ScalerRecord to the transform log.

        Args:
            scaler (Scaler | Any):
                A Scaler instance or any sklearn-like object implementing `fit()` and \
                `transform()`. If not a Scaler, it is wrapped as `Scaler(scaler=<obj>)`.
            domain (str):
                One of {"features", "targets", "tags"}.
            keys (str | list[str] | None, optional):
                Column names within the domain to transform. If None, all domain keys \
                are used. Defaults to None.
            fit_to_split (str | None, optional):
                Split name from which to fit the scaler (e.g., "train"). If None, the \
                scaler is fit to all samples. Defaults to None.
            merged_axes (int | tuple[int]):
                Axes whose sizes are merged into a single dimension. If None, no axes are \
                merged. An Error will be thrown if the resulting shape is not 2-dimensional.

        Raises:
            ValueError:
                - If flattening with `merged_axes` does not result in 2D data.
                - If representation or domain keys do not exist.
            TypeError:
                If `scaler` is not a valid Scaler or sklearn-transformer-like object.

        """
        # Ensure scaler is a Scaler instance
        if not isinstance(scaler, Scaler):
            scaler = Scaler(scaler=scaler)

        # Infer representation to fit to:
        #   - if all specified keys already have a "transformed" representation, use that
        #   - otherwise, use the "raw" representation

        # Use specified keys, or all keys
        keys = ensure_list(keys)
        if not keys:
            keys = self.collection._get_domain_keys(
                domain=domain,
                include_domain_prefix=False,
                include_rep_suffix=False,
            )

        # Find common representation
        rep_to_use = REP_RAW
        if self._scaler_recs:
            has_transformed_rep = True
            for k in keys:
                existing_vars = self.collection._get_rep_keys(domain=domain, key=k)
                if REP_TRANSFORMED not in existing_vars:
                    has_transformed_rep = False
                    break
            rep_to_use = REP_TRANSFORMED if has_transformed_rep else REP_RAW

        # Get data to fit to and fit Scaler
        x_fit, _ = self._get_flat_data(
            domain=domain,
            keys=keys,
            rep=rep_to_use,
            split=fit_to_split,
            merged_axes=merged_axes,
        )
        scaler.fit(x_fit)

        # Get data to transform
        x_all, x_all_meta = self._get_flat_data(
            domain=domain,
            keys=keys,
            rep=rep_to_use,
            split=None,
            merged_axes=merged_axes,
        )
        x_trans_flat = scaler.transform(x_all)

        # Unpack transformed data back to original shapes
        x_trans = unflatten_from_2d(flat=x_trans_flat, meta=x_all_meta) if x_all_meta else x_trans_flat

        for k_idx, k in enumerate(keys):
            # x_trans has shape like (n_samples, n_f, f_shape) where n_f is number of keys
            # We need to select data belonging to each key -> shape = (n_samples, f_shape)
            self.collection.add_rep(
                domain=domain,
                key=k,
                rep=REP_TRANSFORMED,
                data=x_trans[:, k_idx],
                overwrite=True,
            )

        # Record this scaler configuration
        # Scaler is cloned to prevent user from modifying state outside of ModularML
        cloned_scaler: Scaler = clone_via_serialization(obj=scaler)
        cloned_scaler.fit(x_fit)
        _ = cloned_scaler.transform(x_all)
        rec = ScalerRecord(
            order=len(self._scaler_recs),
            domain=domain,
            keys=keys,
            rep_in=rep_to_use,
            rep_out=REP_TRANSFORMED,
            fit_split=fit_to_split,
            merged_axes=merged_axes,
            flatten_meta=x_all_meta,
            scaler_obj=cloned_scaler,
        )
        self._scaler_recs.append(rec)

    def undo_last_transform(
        self,
        domain: str,
        keys: str | list[str] | None = None,
    ):
        """
        Undo the most recent transform applied to a specific subset of columns.

        Description:
            Reverts only the last ScalerRecord that:
                - matches exactly the provided `domain` + `keys`, AND
                - is not depended on by a more recent transforms.

            The method:
                - Finds the most recent exact match (same set of keys).
                - Ensures no later transform includes these keys (dependency safety).
                - Flattens data using stored metadata.
                - Applies `inverse_transform()`.
                - Unflattens back to the original shape.
                - Replaces the transformed representation.
                - Removes the ScalerRecord and updates order indices.

        Args:
            domain (str):
                One of {"features", "targets", "tags"}.
            keys (str | list[str] | None):
                Keys to revert. If None, all domain keys are used.

        Raises:
            ValueError:
                - If no transforms were applied on these columns.
                - If the last applicable transform cannot be undone due to \
                dependency on a more recent transform.
                - If flatten metadata does not match stored ScalerRecord metadata.
                - If internal transform history becomes inconsistent.

        """
        # Undoing the last transform applied to some subset of FeatureSet
        #   1. Define the subset of interest
        #   - Transform can only be undone if the specified subset of interest covers
        #       the full subset of the last transform and no other transforms overlap
        #       with this subset of interest
        #   - Example 1: successful
        #       apply X to "features.voltage"
        #       apply Y to "features.voltage"
        #       undo last on "features.voltage" --> inverts Y
        #   - Example 2: invalid
        #       apply X to "features.[voltage, current]"
        #       apply Y to "features.voltage"
        #       undo last on "features.[voltage, current]"
        #           -> the last transform on the specified domain is X
        #           -> but transform Y depends on X so we cannot invert
        #           -> must first undo Y on "features.voltage"

        # Use specified keys, or all keys
        keys = ensure_list(keys)
        if not keys:
            keys = self.collection._get_domain_keys(
                domain=domain,
                include_domain_prefix=False,
                include_rep_suffix=False,
            )

        # Get most recent ScalerRecord for:
        #   a. scalers using only the specified domain + keys (`last_exact`)
        #   b. scalers including the specified domain + keys (`last_incl`)
        # If (a) is not more recent than or equal to (b), the transform cannot be undone
        last_exact, last_incl = None, None
        for record in reversed(self._scaler_recs):
            if record.domain == domain:
                # If record matches specified domain + keys exactly (order doesn't matter)
                if last_exact is None and set(record.keys) == set(keys):
                    last_exact = record
                # If record contains at least the specified domain + keys
                if last_incl is None and all(ki in record.keys for ki in keys):
                    last_incl = record

            if last_exact is not None and last_incl is not None:
                break
        if last_exact is None and last_incl is not None:
            msg = (
                f"There are no Scalers applied to exactly {domain}.{keys}. "
                f"There are Scalers applied that include these keys (along with other). "
                f"To undo those, use the following keys: {last_incl.keys}"
            )
            raise ValueError(msg)
        if last_exact is None and last_incl is None:
            msg = "There are no Scalers to undo."
            raise ValueError(msg)

        # Raise error if cannot undo scaler on specified domain
        if last_exact.order < last_incl.order:
            msg = (
                f"The last scaler applied to {domain}.{keys} cannot be undone as a "
                "more recent scaler depends on it. You need to first undo the scaler "
                f"applied to {last_incl.domain}.{last_incl.keys}."
            )
            raise ValueError(msg)

        # Get flattened data to inverse
        x_flat, x_flat_meta = self._get_flat_data(
            domain=last_exact.domain,
            keys=last_exact.keys,
            rep=last_exact.rep_out,
            split=None,
            merged_axes=last_exact.merged_axes,
        )
        if x_flat_meta != last_exact.flatten_meta:
            msg = (
                "Flattened data does not match expected metadata from ScalerRecord: "
                f"{x_flat_meta} != {last_exact.flatten_meta}"
            )
            raise ValueError(msg)

        # Inverse data and reshape
        x_inv_flat = last_exact.scaler_obj.inverse_transform(x_flat)
        x_inv = unflatten_from_2d(flat=x_inv_flat, meta=x_flat_meta) if x_flat_meta else x_inv_flat

        # Store this data back into collection table
        for k_idx, k in enumerate(last_exact.keys):
            self.collection.add_rep(
                domain=last_exact.domain,
                key=k,
                rep=last_exact.rep_out,
                data=x_inv[:, k_idx],
                overwrite=True,
            )

        # Delete record from logs
        if self._scaler_recs[last_exact.order] != last_exact:
            raise ValueError("The record to be deleted is out of order. Check `_scaler_recs`")

        # Update other records to their new positions
        new_recs = []
        for i in range(len(self._scaler_recs)):
            if i > last_exact.order:
                new_recs.append(replace(self._scaler_recs[i], order=self._scaler_recs[i].order - 1))
            else:
                new_recs.append(self._scaler_recs[i])

        # Delete record
        last_exact = new_recs.pop(last_exact.order)
        self._scaler_recs = new_recs

        # Clean up collection
        self._cleanup_transformed_rep_if_unused()

    def undo_all_transforms(
        self,
        domain: str | None = None,
        keys: str | list[str] | None = None,
    ):
        """
        Undo every transform applied to the specified domain/keys, in reverse order.

        Description:
            Repeatedly calls `undo_last_transform()` until:
                - all matching transforms are undone, or
                - a dependency violation prevents further undoing.

        Note:
            Raw (untransformed) data is always available via \
            `get_features(..., rep="raw")`, even without undoing.

        Args:
            domain (str | None):
                One of {"features", "targets", "tags"}. If None, reverts transforms \
                on all domains.
            keys (str | list[str] | None):
                Keys to fully revert. If None, all domain keys are used.

        Raises:
            ValueError:
                - If a transform cannot be undone because a newer transform depends on it.
                - For any other underlying error encountered by `undo_last_transform()`.

        """
        domains = [domain] if domain is not None else [DOMAIN_FEATURES, DOMAIN_TARGETS, DOMAIN_TAGS]
        for d in domains:
            try:
                while True:
                    self.undo_last_transform(domain=d, keys=keys)
            except ValueError as e:  # noqa: PERF203
                msg = str(e)

                # Case: “There are no Scalers to undo.”
                if "no scalers to undo" in msg.lower():
                    continue  # All transforms successfully undone

                # Case: dependency violation (“cannot be undone… depends on it”)
                if "cannot be undone" in msg.lower():
                    msg = f"Stopped early. Cannot undo all transforms for: {domain}.{keys}. Reason: {msg}"
                    raise ValueError(msg) from e

                # Any other ValueError is unexpected -> re-raise
                raise

    # ================================================
    # Referencing
    # ================================================
    def reference(
        self,
        columns: list[str] | None = None,
        *,
        features: str | list[str] | None = None,
        targets: str | list[str] | None = None,
        tags: str | list[str] | None = None,
        rep: str | None = None,
    ) -> FeatureSetReference:
        """
        Create a FeatureSetReference object pointing to columns in this FeatureSet.

        Description:
            Uses the same column-selection semantics as `FeatureSet.select`, but
            returns symbolic DataReference objects instead of a view or materialized data.

            References preserve:
                - Domain (features / targets / tags)
                - Key name
                - Representation
                - Transform history

            This is the preferred mechanism for wiring FeatureSets into ModelStages
            and other graph components.

        Notes:
            All columns of a domain are included unless specified. I.e., if `tags=None`
            and no tags are specified in `columns`, then all tag columns are included in
            the returned FeatureSetReference.

        Args:
            columns (list[str] | None):
                Fully-qualified column names to reference
                (e.g. "features.voltage.raw").

            features (str | list[str] | None):
                Feature-domain selectors. Accepts exact names or wildcards.
                Domain prefix may be omitted.

            targets (str | list[str] | None):
                Target-domain selectors, following the same rules as `features`.

            tags (str | list[str] | None):
                Tag-domain selectors, following the same rules as `features`.

            rep (str | None):
                Default representation suffix applied when a selector omits a
                representation.

        Returns:
            FeatureSetReference

        """
        # Get all available keys
        all_cols = self.get_all_keys(
            include_domain_prefix=True,
            include_rep_suffix=True,
        )
        all_cols.remove(DOMAIN_SAMPLE_ID)

        # Perform column selection (organized by domain)
        selected: dict[str, set[str]] = resolve_column_selectors(
            all_columns=all_cols,
            columns=columns,
            features=features,
            targets=targets,
            tags=tags,
            rep=rep,
            include_all_if_empty=True,
        )

        return FeatureSetReference(
            node_id=self.node_id,
            node_label=self.label,
            features=tuple(selected[DOMAIN_FEATURES]),
            targets=tuple(selected[DOMAIN_TARGETS]),
            tags=tuple(selected[DOMAIN_TAGS]),
        )

    def column_reference(
        self,
        column: str | None = None,
        *,
        feature: str | None = None,
        target: str | None = None,
        tag: str | None = None,
        rep: str | None = None,
    ) -> FeatureSetColumnReference:
        """
        Create a reference to a single column in this FeatureSet.

        Args:
            column (str | None):
                Fully-qualified column name to reference
                (e.g. "features.voltage.raw").

            feature (str | None):
                Feature-domain selector. Domain prefix may be omitted.

            target (str | None):
                Target-domain selector. Domain prefix may be omitted.

            tag (str | None):
                Tag-domain selector. Domain prefix may be omitted.

            rep (str | None):
                Default representation suffix applied when a selector omits a
                representation.

        Returns:
            FeatureSetColumnReference

        """
        # Early return if column=sample_id
        if column == DOMAIN_SAMPLE_ID:
            return FeatureSetColumnReference(
                node_id=self.node_id,
                node_label=self.label,
                domain=DOMAIN_SAMPLE_ID,
                key=None,
                rep=None,
            )

        # Argument validation
        if column is not None and not isinstance(column, str):
            raise TypeError("`column` must be a string. Received: type(column)")
        if feature is not None and not isinstance(feature, str):
            raise TypeError("`feature` must be a string. Received: type(feature)")
        if target is not None and not isinstance(target, str):
            raise TypeError("`target` must be a string. Received: type(target)")
        if tag is not None and not isinstance(tag, str):
            raise TypeError("`tag` must be a string. Received: type(tag)")
        if rep is not None and not isinstance(rep, str):
            raise TypeError("`rep` must be a string. Received: type(rep)")

        # Get all available keys
        all_cols = self.get_all_keys(
            include_domain_prefix=True,
            include_rep_suffix=True,
        )
        all_cols.remove(DOMAIN_SAMPLE_ID)

        # Perform column selection (organized by domain)
        selected: dict[str, set[str]] = resolve_column_selectors(
            all_columns=all_cols,
            columns=[column] if column is not None else None,
            features=[feature] if feature is not None else None,
            targets=[target] if target is not None else None,
            tags=[tag] if tag is not None else None,
            rep=rep,
            include_all_if_empty=False,
        )
        flat_selected: set[str] = set().union(*selected.values())
        if len(flat_selected) != 1:
            msg = f"Cannot construct a column reference to more than one column. Received: {flat_selected}"
            raise ResolutionError(msg)

        domain, key, rep = next(iter(flat_selected)).split(".", maxsplit=2)
        return FeatureSetColumnReference(
            node_id=self.node_id,
            node_label=self.label,
            domain=domain,
            key=key,
            rep=rep,
        )

    # ================================================
    # Configurable
    # ================================================
    def get_config(self) -> dict[str, Any]:
        """Structural configuration of the FeatureSet and parents."""
        config = super().get_config()
        return config

    @classmethod
    def from_config(cls, config: dict[str, Any], *, register: bool = True) -> FeatureSet:
        """Instantiates a empty FeatureSet."""
        empty_table = pa.table({})
        return cls(table=empty_table, register=register, **config)

    # ================================================
    # Stateful
    # ================================================
    def get_state(self) -> dict[str, Any]:
        """Runtime state (i.e., PyArrow table and records) of the FeatureSet."""
        return {
            "sample_collection": self.collection,
            "table_hash": hash_pyarrow_table(self.collection.table),
            "splits": {k: self.get_split(k) for k in self.available_splits},
            "splitter_records": self._split_recs,
            "scaler_records": self._scaler_recs,
        }

    def set_state(self, state: dict[str, Any]):
        """Restore FeatureSet from semantic state."""
        from modularml.core.data.featureset_view import FeatureSetView

        # Clone sample collection (to prevent shared mutation)
        state_coll: SampleCollection = state["sample_collection"]
        self.collection = SampleCollection(
            table=pa.Table.from_pandas(state_coll.table.to_pandas()),
            schema=state_coll.schema,
        )

        # PyArrow table integrity check
        expected = state.get("table_hash")
        if expected is not None:
            actual = hash_pyarrow_table(self.collection.table)
            if actual != expected:
                msg = f"Arrow table integrity check failed: {actual} != {expected}"
                raise ValueError(msg)

        # Restore splits (copy all instances)
        state_splits: dict[str, FeatureSetView] = state["splits"]
        self._splits = {
            k: FeatureSetView(
                source=self,
                indices=fsv.indices,
                columns=fsv.columns,
                label=fsv.label,
            )
            for k, fsv in state_splits.items()
        }

        # TODO: splitter and scaler instances are not copied here
        # This may result in accidental mutation
        # Restore split records (copy all instances)
        self._split_recs = state["splitter_records"]

        # Restore scalers
        self._scaler_recs = state["scaler_records"]

    # ================================================
    # Serialization
    # ================================================
    def save(self, filepath: Path, *, overwrite: bool = False) -> Path:
        """
        Serializes this FeatureSet to the specified filepath.

        Args:
            filepath (Path):
                File location to save to. Note that the suffix may be overwritten
                to enforce the ModularML file extension schema.
            overwrite (bool, optional):
                Whether to overwrite any existing file at the save location.
                Defaults to False.

        Returns:
            Path: The actual filepath to write the FeatureSet is saved.

        """
        from modularml.core.io.serialization_policy import SerializationPolicy
        from modularml.core.io.serializer import serializer

        return serializer.save(
            self,
            filepath,
            policy=SerializationPolicy.BUILTIN,
            overwrite=overwrite,
        )

    @classmethod
    def load(
        cls,
        filepath: Path,
        *,
        allow_packaged_code: bool = False,
        overwrite: bool = True,
    ) -> FeatureSet:
        """
        Load a FeatureSet from file.

        Args:
            filepath (Path):
                File location of a previously saved FeatureSet.
            allow_packaged_code : bool
                Whether bundled code execution is allowed.
            overwrite (bool):
                Whether to replace any colliding node registrations in ExperimentContext
                If False, a new node_id is assigned to the reloaded FeatureSet. Otherwise,
                the existing FeatureSet is removed from the ExperimentContext registry.
                Defaults to True.

        Returns:
            FeatureSet: The reloaded FeatureSet.

        """
        from modularml.core.io.serializer import _enforce_file_suffix, serializer

        # Append proper sufficx only if no suffix is given
        if Path(filepath).suffix == "":
            filepath = _enforce_file_suffix(path=filepath, cls=cls)

        return serializer.load(
            filepath,
            allow_packaged_code=allow_packaged_code,
            overwrite=overwrite,
        )
