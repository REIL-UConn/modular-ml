from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pyarrow as pa

from modularml.core.data.sample_collection import SampleCollection
from modularml.core.data.sample_data import SampleData
from modularml.core.data.schema_constants import (
    DOMAIN_FEATURES,
    DOMAIN_SAMPLE_ID,
    DOMAIN_TAGS,
    DOMAIN_TARGETS,
)
from modularml.utils.data.conversion import convert_dict_to_format, convert_to_format
from modularml.utils.data.data_format import (
    _TENSORLIKE_FORMATS,
    DataFormat,
    format_is_tensorlike,
)
from modularml.utils.data.pyarrow_data import resolve_column_selectors

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from modularml.core.data.featureset import FeatureSet


class SampleCollectionMixin:
    """
    Mixin providing a uniform SampleCollection accessor API.

    Description:
        Exposes read-only access to SampleCollection-backed data for FeatureSet,
        and FeatureSetViews. The concrete class is responsible for defining
        where the SampleCollection lives and which row indices are active.

    Required hook:
        _resolve_caller_attributes() -> (SampleCollection, list[str] | None, np.ndarray | None)
    """

    # ================================================
    # Core resolution hook (must be implemented by host class)
    # ================================================
    def _resolve_caller_attributes(
        self,
    ) -> tuple[SampleCollection, list[str] | None, np.ndarray | None]:
        """Must resolve the source SampleCollection, list of columns, and row indices of the caller."""
        raise NotImplementedError

    def _resolve_collection(self) -> tuple[SampleCollection, np.ndarray | None]:
        """Returns SampleColection and mask over callers columns and indices."""
        collection, columns, indices = self._resolve_caller_attributes()
        if columns is not None and DOMAIN_SAMPLE_ID not in columns:
            columns.append(DOMAIN_SAMPLE_ID)

        # Column filtering
        sub_table = (
            collection.table if columns is None else collection.table.select(columns)
        )

        if indices is None:
            return SampleCollection(table=sub_table), None

        # Row filtering over only valid (non-negative indices)
        indices = np.asarray(indices)
        mask = indices >= 0
        valid_indices = indices[mask]
        sub_table = sub_table.take(pa.array(valid_indices))

        # Return collection of valid indices, and mask (zeros = areas to pad)
        return SampleCollection(table=sub_table), mask

    def _pad_output(
        self,
        data: dict[str, np.ndarray],
        mask: np.ndarray,
        *,
        fill_value: Any | None = None,
    ) -> dict[str, np.ndarray]:
        """
        Insert padded elements where mask == False.

        Args:
            data (dict[str, np.ndarray]):
                Mapping of column name -> array with leading dimension = n_samples.
            mask (np.ndarray):
                Boolean or 0/1 mask of length n_samples indicating valid rows.
            fill_value (Any):
                Value used to fill padded rows. Must be compatible with array dtype.
                If None, infers fill value based on numpy dtype of the data.

        Returns:
            dict[str, np.ndarray]:
                Padded arrays with the same shapes/dtypes as inputs.

        """

        def _default_fill_value(dtype: np.dtype):
            """Return a dtype-appropriate default fill value."""
            kind = dtype.kind

            if kind == "f":  # float
                return 0.0
            if kind in ("i", "u"):  # int / uint
                return 0
            if kind == "b":  # bool
                return False
            if kind in ("U", "S"):  # unicode / bytes
                return ""
            if kind == "O":  # object (e.g. strings, mixed metadata)
                return None

            msg = f"No default fill value defined for dtype {dtype}"
            raise TypeError(msg)

        mask = np.asarray(mask, dtype=bool)
        if len(mask) != self.n_samples:
            msg = f"Mask must have length equal to number of samples. {len(mask)} != {self.n_samples}"
            raise ValueError(msg)

        def _pad_array(arr: np.ndarray) -> np.ndarray:
            if arr is None:
                return None
            if arr.shape[0] != self.n_samples:
                msg = f"Data must have leading dimension equal to number of samples. {arr.shape[0]} != {self.n_samples}"
                raise ValueError(msg)

            # Get shape of samples
            sample_shape = arr.shape[1:]
            out_shape = (len(mask), *sample_shape)

            # Resolve fill value
            fv = (
                fill_value if fill_value is not None else _default_fill_value(arr.dtype)
            )

            # Allocate padded output using fill_value
            try:
                out = np.full(out_shape, fv, dtype=arr.dtype)
            except (TypeError, ValueError):
                # Fallback for incompatible fill_value (e.g. string into numeric)
                out = np.empty(out_shape, dtype=arr.dtype)
                out[:] = fv

            # Insert real data
            out[mask] = arr

            return out

        return {k: _pad_array(v) for k, v in data.items()}

    # ================================================
    # Basic properties
    # ================================================
    @property
    def n_samples(self) -> int:
        """Total number of samples (rows)."""
        collection, _ = self._resolve_collection()
        return collection.n_samples

    def __len__(self) -> int:
        return self.n_samples

    @property
    def sample_mask(self) -> NDArray[np.int8] | None:
        """Mask over samples: 1=valid sample, 0=padded sample."""
        _, _, indices = self._resolve_caller_attributes()
        if indices is None:
            return None
        return (np.asarray(indices) >= 0).astype(np.int8)

    # ================================================
    # Key accessors
    # ================================================
    def get_feature_keys(self, *, include_rep_suffix=True, include_domain_prefix=True):
        """
        Retrieves the feature column names.

        Description:
            Returns keys for the specified domain with fully
            explicit control over key formatting. No implicit suffix or
            prefix behavior is applied.

        Args:
            include_rep_suffix (bool):
                Whether to include representation suffixes (e.g., "raw")
            include_domain_prefix (bool):
                Whether to include domain prefixes (e.g., "features")

        Returns:
            List of all feature column names.

        """
        collection, _ = self._resolve_collection()
        return collection.get_feature_keys(
            include_rep_suffix=include_rep_suffix,
            include_domain_prefix=include_domain_prefix,
        )

    def get_target_keys(self, *, include_rep_suffix=True, include_domain_prefix=True):
        """
        Retrieves the target column names.

        Description:
            Returns keys for the specified domain with fully
            explicit control over key formatting. No implicit suffix or
            prefix behavior is applied.

        Args:
            include_rep_suffix (bool):
                Whether to include representation suffixes (e.g., "raw")
            include_domain_prefix (bool):
                Whether to include domain prefixes (e.g., "targets")

        Returns:
            List of all target column names.

        """
        collection, _ = self._resolve_collection()
        return collection.get_target_keys(
            include_rep_suffix=include_rep_suffix,
            include_domain_prefix=include_domain_prefix,
        )

    def get_tag_keys(self, *, include_rep_suffix=True, include_domain_prefix=True):
        """
        Retrieves the tag column names.

        Description:
            Returns keys for the specified domain with fully
            explicit control over key formatting. No implicit suffix or
            prefix behavior is applied.

        Args:
            include_rep_suffix (bool):
                Whether to include representation suffixes (e.g., "raw")
            include_domain_prefix (bool):
                Whether to include domain prefixes (e.g., "tags")

        Returns:
            List of all tag column names.

        """
        collection, _ = self._resolve_collection()
        return collection.get_tag_keys(
            include_rep_suffix=include_rep_suffix,
            include_domain_prefix=include_domain_prefix,
        )

    def get_all_keys(self, *, include_rep_suffix=True, include_domain_prefix=True):
        """
        Get all column names in the SampleCollection.

        Args:
            include_rep_suffix (bool):
                Whether to include the representation suffix in the keys
                (e.g., "voltage" or "voltage.raw"). Defaults to True.
            include_domain_prefix (bool): Wether to include the domain prefix in the
                keys (e.g., "cell_id" or "tags.cell_id"). Defaults to True.

        Returns:
            list[str]:
                All unique columns in the SampleCollection.

        """
        collection, _ = self._resolve_collection()
        return collection.get_all_keys(
            include_rep_suffix=include_rep_suffix,
            include_domain_prefix=include_domain_prefix,
        )

    # ================================================
    # Shape / dtype accessors
    # ================================================
    def get_feature_shapes(
        self,
        *,
        include_rep_suffix=True,
        include_domain_prefix=True,
    ):
        """
        Mapping of feature columns to their array shapes.

        Description:
            Returns a dictionary describing the shape of each feature representation
            in the FeatureSet.
            Keys are formatted depending on the bool flags and values are tuples
            representing the array shape (e.g., `(101,)`).
            The shape does not include the number of samples dimension.

        Args:
            include_rep_suffix (bool):
                Whether to include representation suffixes (e.g., "raw")
            include_domain_prefix (bool):
                Whether to include domain prefixes (e.g., "features")

        Returns:
            dict[str, tuple[int, ...]]:
                Mapping of feature column representation names to their shapes.

        """
        collection, _ = self._resolve_collection()
        return collection.get_feature_shapes(
            include_rep_suffix=include_rep_suffix,
            include_domain_prefix=include_domain_prefix,
        )

    def get_target_shapes(self, *, include_rep_suffix=True, include_domain_prefix=True):
        """
        Mapping of target columns to their array shapes.

        Description:
            Returns a dictionary describing the shape of each target representation
            in the FeatureSet.
            Keys are formatted depending on the bool flags and values are tuples
            representing the array shape (e.g., `(101,)`).
            The shape does not include the number of samples dimension.

        Args:
            include_rep_suffix (bool):
                Whether to include representation suffixes (e.g., "raw")
            include_domain_prefix (bool):
                Whether to include domain prefixes (e.g., "targets")

        Returns:
            dict[str, tuple[int, ...]]:
                Mapping of target column representation names to their shapes.

        """
        collection, _ = self._resolve_collection()
        return collection.get_target_shapes(
            include_rep_suffix=include_rep_suffix,
            include_domain_prefix=include_domain_prefix,
        )

    def get_tag_shapes(self, *, include_rep_suffix=True, include_domain_prefix=True):
        """
        Mapping of tag columns to their array shapes.

        Description:
            Returns a dictionary describing the shape of each tag representation
            in the FeatureSet.
            Keys are formatted depending on the bool flags and values are tuples
            representing the array shape (e.g., `(101,)`).
            The shape does not include the number of samples dimension.

        Args:
            include_rep_suffix (bool):
                Whether to include representation suffixes (e.g., "raw")
            include_domain_prefix (bool):
                Whether to include domain prefixes (e.g., "tags")

        Returns:
            dict[str, tuple[int, ...]]:
                Mapping of tag column representation names to their shapes.

        """
        collection, _ = self._resolve_collection()
        return collection.get_tag_shapes(
            include_rep_suffix=include_rep_suffix,
            include_domain_prefix=include_domain_prefix,
        )

    def get_feature_dtypes(
        self,
        *,
        include_rep_suffix=True,
        include_domain_prefix=True,
    ):
        """
        Mapping of feature columns to their data types.

        Description:
            Returns a dictionary describing the data type of each feature representation.
            Keys are formatted depending on the bool flags and values are string-based
            data-types (e.g., "float32").

        Args:
            include_rep_suffix (bool):
                Whether to include representation suffixes (e.g., "raw")
            include_domain_prefix (bool):
                Whether to include domain prefixes (e.g., "features")

        Returns:
            dict[str, str]:
                Mapping of feature column names to their data-types.

        """
        collection, _ = self._resolve_collection()
        return collection.get_feature_dtypes(
            include_rep_suffix=include_rep_suffix,
            include_domain_prefix=include_domain_prefix,
        )

    def get_target_dtypes(self, *, include_rep_suffix=True, include_domain_prefix=True):
        """
        Mapping of targets columns to their data types.

        Description:
            Returns a dictionary describing the data type of each targets representation.
            Keys are formatted depending on the bool flags and values are string-based
            data-types (e.g., "float32").

        Args:
            include_rep_suffix (bool):
                Whether to include representation suffixes (e.g., "raw")
            include_domain_prefix (bool):
                Whether to include domain prefixes (e.g., "targetss")

        Returns:
            dict[str, str]:
                Mapping of targets column names to their data-types.

        """
        collection, _ = self._resolve_collection()
        return collection.get_target_dtypes(
            include_rep_suffix=include_rep_suffix,
            include_domain_prefix=include_domain_prefix,
        )

    def get_tag_dtypes(self, *, include_rep_suffix=True, include_domain_prefix=True):
        """
        Mapping of tag columns to their data types.

        Description:
            Returns a dictionary describing the data type of each tag representation.
            Keys are formatted depending on the bool flags and values are string-based
            data-types (e.g., "float32").

        Args:
            include_rep_suffix (bool):
                Whether to include representation suffixes (e.g., "raw")
            include_domain_prefix (bool):
                Whether to include domain prefixes (e.g., "tags")

        Returns:
            dict[str, str]:
                Mapping of tag column names to their data-types.

        """
        collection, _ = self._resolve_collection()
        return collection.get_tag_dtypes(
            include_rep_suffix=include_rep_suffix,
            include_domain_prefix=include_domain_prefix,
        )

    # ================================================
    # Domain data access
    # ================================================
    def get_data(
        self,
        *,
        columns: str | list[str] | None = None,
        features: str | list[str] | None = None,
        targets: str | list[str] | None = None,
        tags: str | list[str] | None = None,
        rep: str | None = None,
        fmt: DataFormat = DataFormat.DICT_NUMPY,
        include_domain_prefix: bool = True,
        include_rep_suffix: bool = True,
    ) -> Any:
        """
        Retrieves a subset of data from this FeatureSet/View.

        Description:
            Data selection is performed via column-wise filtering and returned
            in the format specified by `fmt`.

            Selection supports:
                - Explicit fully-qualified columns (e.g. "features.voltage.raw")
                - Domain-based selectors via `features`, `targets`, and `tags`
                - Wildcards (e.g. "*.raw", "voltage.*", "*.*")
                - Automatic domain prefixing ("voltage.raw" -> "features.voltage.raw")
                - Optional default representation inference via `rep`

            If multiple columns are selected, and a non-keyed data formed is used,
            the columns are stacked in the following order:
                - Domain sorted with: features -> targets -> tags -> sample_id
                - Within each domain, columns are sorted alphabetically.

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

            fmt (DataFormat, optional):
                The format of the returned data. Defaults to a dict of numpy arrays
                (i.e., each key corresponds to a single column).

            include_domain_prefix (bool):
                Whether to include domain prefixes (e.g., "tags").
                Automatically included if multiple domains are included in the
                selected columns. Defaults to False.

            include_rep_suffix (bool):
                Whether to include representation suffixes (e.g., "raw").
                Automatically included if multiple representations are included in
                the selected columns. Defaults to True.

        Returns:
            Data from the specified columns, in the request DataFormat.

        """
        collection, mask = self._resolve_collection()

        # Extract real columns from collection
        all_cols: list[str] = collection.get_all_keys(
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

        # Order and flatten columns: features -> targets -> tags
        sel_cols: list[str] = []
        for d in [DOMAIN_FEATURES, DOMAIN_TARGETS, DOMAIN_TAGS, DOMAIN_SAMPLE_ID]:
            if d in selected:
                sel_cols.extend(sorted(selected[d]))

        # Grab data from valid indices
        data = collection.get_columns(
            columns=sel_cols,
            fmt=DataFormat.DICT_NUMPY,
            include_domain_prefix=include_domain_prefix,
            include_rep_suffix=include_rep_suffix,
        )

        # Pad invalid indices, if needed
        if mask is not None:
            data = self._pad_output(data=data, mask=mask)

        # Convert data to requested format
        return convert_dict_to_format(data=data, fmt=fmt, mode="stack", axis=1)

    def get_features(
        self,
        fmt: DataFormat = DataFormat.DICT_NUMPY,
        *,
        features: str | list[str] | None = None,
        rep: str | None = None,
        include_domain_prefix: bool = False,
        include_rep_suffix: bool = False,
    ):
        """
        Retrieves a subset of feature data from this FeatureSet/View.

        Description:
            Data selection is performed via column-wise filtering and returned
            in the format specified by `fmt`.

            Selection supports:
                - Explicit fully-qualified features (e.g. "voltage.raw")
                - Wildcards (e.g. "*.raw", "voltage.*", "*")
                - Optional default representation inference via `rep`

            If multiple columns are selected, and a non-keyed data formed is used,
            the columns are sorted alphabetically.

        Args:
            fmt (DataFormat, optional):
                The format of the returned data. Defaults to a dict of numpy arrays
                (i.e., each key corresponds to a single column).

            features (str | list[str] | None):
                Feature-domain selectors. May be bare keys ("voltage"),
                key/rep pairs ("voltage.raw"), or wildcards.
                The "features." prefix may be omitted.

            rep (str | None):
                Default representation suffix to apply when a selector omits a
                representation. Explicit representations are never overridden.

            include_domain_prefix (bool):
                Whether to include domain prefixes (e.g., "tags").
                Defaults to False.

            include_rep_suffix (bool):
                Whether to include representation suffixes (e.g., "raw").
                Automatically included if multiple representations are included in
                the selected columns. Defaults to False.

        Returns:
            Data from the specified columns, in the request DataFormat.

        """
        return self.get_data(
            features=features or "*",
            rep=rep,
            fmt=fmt,
            include_domain_prefix=include_domain_prefix,
            include_rep_suffix=include_rep_suffix,
        )

    def get_targets(
        self,
        fmt: DataFormat = DataFormat.DICT_NUMPY,
        *,
        targets: str | list[str] | None = None,
        rep: str | None = None,
        include_domain_prefix: bool = False,
        include_rep_suffix: bool = False,
    ):
        """
        Retrieves a subset of target data from this FeatureSet/View.

        Description:
            Data selection is performed via column-wise filtering and returned
            in the format specified by `fmt`.

            Selection supports:
                - Explicit fully-qualified targets (e.g. "soh.raw")
                - Wildcards (e.g. "*.raw", "soh.*", "*")
                - Optional default representation inference via `rep`

            If multiple columns are selected, and a non-keyed data formed is used,
            the columns are sorted alphabetically.

        Args:
            fmt (DataFormat, optional):
                The format of the returned data. Defaults to a dict of numpy arrays
                (i.e., each key corresponds to a single column).

            targets (str | list[str] | None):
                Feature-domain selectors. May be bare keys ("soh"),
                key/rep pairs ("soh.raw"), or wildcards.
                The "targets." prefix may be omitted.

            rep (str | None):
                Default representation suffix to apply when a selector omits a
                representation. Explicit representations are never overridden.

            include_domain_prefix (bool):
                Whether to include domain prefixes (e.g., "tags").
                Defaults to False.

            include_rep_suffix (bool):
                Whether to include representation suffixes (e.g., "raw").
                Automatically included if multiple representations are included in
                the selected columns. Defaults to False.

        Returns:
            Data from the specified columns, in the request DataFormat.

        """
        return self.get_data(
            targets=targets or "*",
            rep=rep,
            fmt=fmt,
            include_domain_prefix=include_domain_prefix,
            include_rep_suffix=include_rep_suffix,
        )

    def get_tags(
        self,
        fmt: DataFormat = DataFormat.DICT_NUMPY,
        *,
        tags: str | list[str] | None = None,
        rep: str | None = None,
        include_domain_prefix: bool = False,
        include_rep_suffix: bool = False,
    ):
        """
        Retrieves a subset of tag data from this FeatureSet/View.

        Description:
            Data selection is performed via column-wise filtering and returned
            in the format specified by `fmt`.

            Selection supports:
                - Explicit fully-qualified tags (e.g. "group_id.raw")
                - Wildcards (e.g. "*.raw", "group_id.*", "*")
                - Optional default representation inference via `rep`

            If multiple columns are selected, and a non-keyed data formed is used,
            the columns are sorted alphabetically.

        Args:
            fmt (DataFormat, optional):
                The format of the returned data. Defaults to a dict of numpy arrays
                (i.e., each key corresponds to a single column).

            tags (str | list[str] | None):
                Feature-domain selectors. May be bare keys ("group_id"),
                key/rep pairs ("group_id.raw"), or wildcards.
                The "tags." prefix may be omitted.

            rep (str | None):
                Default representation suffix to apply when a selector omits a
                representation. Explicit representations are never overridden.

            include_domain_prefix (bool):
                Whether to include domain prefixes (e.g., "tags").
                Defaults to False.

            include_rep_suffix (bool):
                Whether to include representation suffixes (e.g., "raw").
                Automatically included if multiple representations are included in
                the selected columns. Defaults to False.

        Returns:
            Data from the specified columns, in the request DataFormat.

        """
        return self.get_data(
            tags=tags or "*",
            rep=rep,
            fmt=fmt,
            include_domain_prefix=include_domain_prefix,
            include_rep_suffix=include_rep_suffix,
        )

    # ================================================
    # UUID access
    # ================================================
    def get_sample_uuids(self, fmt: DataFormat = DataFormat.NUMPY):
        """
        Retrieve sample UUIDs in this collection.

        Args:
            fmt (DataFormat):
                Desired output format (see :class:`DataFormat`).
                Defaults to a single dictionary of numpy arrays.

        Returns:
            Sample UUIDs in the requested format.
            Padded data is given a sample UUID of "N/A".

        """
        collection, mask = self._resolve_collection()

        data = collection.get_sample_uuids(fmt=DataFormat.NUMPY)

        # Pad invalid indices, if needed
        if mask is not None:
            d_data = self._pad_output(
                data={"sample_id": data},
                mask=mask,
                fill_value="N/A",
            )
            data = d_data["sample_id"]

        # Convert data to requested format
        return convert_to_format(data=data, fmt=fmt)

    # ================================================
    # Export
    # ================================================
    def to_arrow(self) -> pa.Table:
        collection, _ = self._resolve_collection()

        # Sort columns (not necessary but nicer for manual inspection)
        sorted_cols = collection.get_all_keys(
            include_domain_prefix=True,
            include_rep_suffix=True,
        )
        return collection.table.select(sorted_cols)

    def to_pandas(
        self,
        *,
        include_padded: bool = False,
        include_mask: bool = False,
    ) -> pd.DataFrame:
        # Case 1: no padding requested or no mask exists
        if not include_padded or self.sample_mask is None:
            df = self.to_arrow().to_pandas()
            if include_mask and self.sample_mask is not None:
                df["__mask__"] = 1
            return df

        # Case 2: include padded rows
        # get_data performs padded automatically if mask is not None
        data = self.get_data(fmt=DataFormat.DICT_NUMPY)

        # Flatten into DataFrame
        df = {}
        for k, v in data.items():
            # Each column may be multi-dim -> store as list
            df[k] = list(v)

        out = pd.DataFrame(df)
        if include_mask:
            out["__mask__"] = self.sample_mask.astype(int)

        return out

    def to_sample_collection(self) -> SampleCollection:
        coll, _ = self._resolve_collection()
        return coll

    def to_featureset(self, label: str) -> FeatureSet:
        from modularml.core.data.featureset import FeatureSet

        return FeatureSet(label=label, collection=self.to_sample_collection())

    def to_sample_data(self, fmt: DataFormat = DataFormat.NUMPY) -> SampleData:
        """
        Converts all columns referenced by this view to tensor-based SampleData.

        Args:
            fmt (DataFormat, optional):
                The data format to use for the tensors constructed from the
                "features" and "targets" columns. `fmt` must be TensorLike
                (e.g., "torch", "tf", "np"). Defaults to "numpy".

        """
        if not format_is_tensorlike(fmt):
            msg = f"DataFormat must be tensor-like. Received: {fmt}. Allowed formats: {_TENSORLIKE_FORMATS}."
            raise ValueError(msg)

        return SampleData(
            sample_uuids=self.get_sample_uuids(fmt=DataFormat.NUMPY),
            features=self.get_features(fmt=fmt),
            targets=self.get_targets(fmt=fmt),
            tags=self.get_tags(fmt=DataFormat.NUMPY),
        )
