"""Multi-axis containers for experiment metrics and tensor data."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Hashable, Iterable, Iterator
from dataclasses import dataclass
from typing import Any, ClassVar, Generic, TypeVar

import numpy as np

from modularml.utils.data.formatting import ensure_tuple
from modularml.utils.environment.optional_imports import check_tensorflow, check_torch

torch = check_torch()
tf = check_tensorflow()

# Data keys -> eg, (epoch, batch, edge)
AxisKey = tuple[Hashable, ...]

# Convenience key -> scalar when single axis, tuple otherwise
ExternalKey = Hashable | tuple[Hashable, ...]
T = TypeVar("T")


@dataclass
class AxisSeries(Generic[T]):
    """
    Container keyed by a fixed set of semantic axes (e.g., epoch, batch).

    Attributes:
        axes (tuple[str, ...]): Ordered axis names describing the coordinate system.
        _data (dict[AxisKey, T]): Values keyed by canonical axis tuples.

    """

    axes: tuple[str, ...]
    _data: dict[AxisKey, T]

    # Subclasses may override to restrict allowed string reducers
    # None = all string reducers are allowed
    supported_reduction_methods: ClassVar[set[str] | None] = None

    # ================================================
    # Axes properties
    # ================================================
    def axis_values(self, axis: str) -> list[Hashable]:
        """
        Return unique values of the specified axis.

        Args:
            axis (str): Axis name to inspect.

        Returns:
            list[Hashable]: Unique axis values.

        Raises:
            KeyError: If `axis` is not part of the series definition.

        """
        # Validate axis name
        try:
            idx = self.axes.index(axis)
        except ValueError as e:
            msg = f"Unknown axis '{axis}'. Available axes: {self.axes}."
            raise KeyError(msg) from e

        # Get unique values (use internal tuple keys)
        uniq = {k[idx] for k in self._data}
        return list(uniq)

    def axes_values(self) -> dict[str, list[Hashable]]:
        """
        Return all unique values for each axis.

        Returns:
            dict[str, list[Hashable]]: Mapping from axis name to its unique values.

        """
        uniq: dict[str, set] = defaultdict(set)
        for k in self._data:  # Use internal tuple keys
            for i, ax_key in enumerate(self.axes):
                uniq[ax_key].add(k[i])

        return {k: list(v) for k, v in uniq.items()}

    @property
    def shape(self) -> dict[str, int]:
        """
        Return the cardinality of each axis.

        Returns:
            dict[str, int]: Axis names mapped to the number of unique values.

        """
        return {k: len(v) for k, v in self.axes_values().items()}

    # ================================================
    # Direct access & querying
    # ================================================
    @property
    def data(self) -> dict[AxisKey, T]:
        """Return the keyed data as a normal dictionary."""
        return {self._unwrap_key(k): v for k, v in self._data.items()}

    def at(self, **coords) -> T:
        """
        Retrieve a value at an exact axis location.

        Values for all axes must be specified.

        Args:
            **coords: Axis name to value mappings for every axis.

        Returns:
            T: Value stored at the requested coordinates.

        Raises:
            KeyError: If any axis is missing or coordinate lookup fails.

        Example:
            >>> series.at(epoch=0, batch=3) Values for all axes must be specified.

        """
        key = self._make_key(coords)
        return self._data[key]

    def __getitem__(self, key: ExternalKey) -> T:
        """
        Retrieve a value using a scalar or tuple key.

        Args:
            key (ExternalKey): External axis key (scalar for single axis).

        Returns:
            T: Stored value.

        Raises:
            KeyError: If the key is not present.

        """
        internal_key = self._wrap_key(key)
        return self._data[internal_key]

    def where(self, **coords) -> AxisSeries[T]:
        """
        Filter the series by one or more axis values.

        Each coordinate may be a scalar or an iterable of allowed values.

        Args:
            **coords: Axis names mapped to scalar values, iterables of values, or predicates.

        Returns:
            AxisSeries[T]: Filtered series containing only matching entries.

        Example:
            >>> series.where(epoch=1)  # doctest: +SKIP
            >>> series.where(epoch=[1, 2, 3])  # doctest: +SKIP
            >>> series.where(epoch=[0, 1], batch=0)  # doctest: +SKIP

        """
        selectors = self._coord_selectors(coords)

        filtered = {
            k: v
            for k, v in self._data.items()
            if all(sel(k[idx]) for idx, sel in selectors)
        }
        series = AxisSeries(axes=self.axes, _data=filtered)
        return series

    def one(self) -> T:
        """
        Return exactly one value or raise.

        Returns:
            T: Sole stored value.

        Raises:
            ValueError: If the series contains zero or multiple values.

        """
        if len(self._data) != 1:
            msg = f"Expected exactly one item, found {len(self._data)}."
            raise ValueError(msg)
        return next(iter(self._data.values()))

    # ================================================
    # Axis slicing
    # ================================================
    def select(self, **coords) -> list[T]:
        """
        Select all values matching a partial axis specification.

        Args:
            **coords: Axis selectors defined via scalars, iterables, or predicates.

        Returns:
            list[T]: Values matching the partial specification.

        Example:
            >>> series.select(epoch=0)  # doctest: +SKIP
            >>> series.select(epoch=[1, 2])  # doctest: +SKIP

        """
        selectors = self._coord_selectors(coords)

        return [
            v
            for k, v in self._data.items()
            if all(sel(k[idx]) for idx, sel in selectors)
        ]

    # ================================================
    # Mapping properties
    # ================================================
    def values(self) -> list[T]:
        """
        Return all values in insertion order.

        Returns:
            list[T]: Stored values ordered by insertion.

        """
        return list(self._data.values())

    def keys(self) -> list[ExternalKey]:
        """
        Return all keys (scalar if single axis, tuple otherwise).

        Returns:
            list[ExternalKey]: External keys for each entry.

        """
        return [self._unwrap_key(k) for k in self._data]

    def items(self) -> Iterator[tuple[ExternalKey, T]]:
        """
        Iterate over key-value pairs (keys are scalar if single axis).

        Yields:
            tuple[ExternalKey, T]: Key-value pairs.

        """
        for k, v in self._data.items():
            yield self._unwrap_key(k), v

    def __iter__(self) -> Iterator[ExternalKey]:
        """
        Iterate over keys (scalar if single axis).

        Yields:
            ExternalKey: Each key in insertion order.

        """
        for k in self._data:
            yield self._unwrap_key(k)

    def __len__(self) -> int:
        """Return the number of stored entries."""
        return len(self._data)

    # ================================================
    # Views
    # ================================================
    def first(self, *, sort_by: list[AxisKey] | None = None) -> T:
        """
        Return the first value.

        Args:
            sort_by (list[AxisKey] | None): Axis keys to sort by before selection.

        Returns:
            T | ValueError: First value, or a ValueError object if the series is empty.

        """
        if len(self.values()) < 1:
            msg = "Cannot take first element of an empty series."
            return ValueError(msg)

        # Sort values by `sort_by` keys, then return first
        if sort_by is not None:
            if not isinstance(sort_by, list):
                sort_by = [sort_by]
            sort_by_idxs = [self.axes.index(x) for x in sort_by]
            sorted_keys = sorted(
                self.keys(),
                key=lambda x: (x[i] for i in sort_by_idxs),
            )
            sorted_vals = [self._data[k] for k in sorted_keys]
            return sorted_vals[0]

        # Return first value of values, as is
        return next(iter(self._data.values()))

    def last(self, *, sort_by: list[AxisKey] | None = None) -> T:
        """
        Return the last value.

        Args:
            sort_by (list[AxisKey] | None): Axis keys to sort by before selection.

        Returns:
            T | ValueError: Last value, or a ValueError object if the series is empty.

        """
        if len(self.values()) < 1:
            msg = "Cannot take last element of an empty series."
            raise ValueError(msg)

        # Sort values by `sort_by` keys, then return last
        if sort_by is not None:
            if not isinstance(sort_by, list):
                sort_by = [sort_by]
            sort_by_idxs = [self.axes.index(x) for x in sort_by]
            sorted_keys = sorted(
                self.keys(),
                key=lambda x: tuple(x[i] for i in sort_by_idxs),
            )
            sorted_vals = [self._data[k] for k in sorted_keys]
            return sorted_vals[-1]

        # Return last value of values, as is
        return next(reversed(self._data.values()))

    # ================================================
    # Reduction
    # ================================================
    def _resolve_reducer(
        self,
        reducer: str,
        data_type: type,
    ) -> Callable[[list[T]], T]:
        """
        Resolves a reduction method from label and data type.

        Args:
            reducer (str): Name of the reduction strategy (e.g., "mean").
            data_type (type): Class of underlying data to be reduced.

        Returns:
            Callable[[list[T]], T]: Callable that reduces lists of values.

        Raises:
            ValueError: If the reducer is unsupported.
            AttributeError: If a referenced reducer attribute does not exist.
            TypeError: If the resolved reducer is not callable.

        """

        def use_attribute(attr: str) -> Callable[[list[T]], T]:
            # Validate against supported_reduction_methods if defined
            supported = getattr(data_type, "supported_reduction_methods", None)
            if (supported is not None) and (reducer not in supported):
                msg = (
                    f"Reducer '{reducer}' is not supported for type "
                    f"'{data_type.__name__}'. Allowed: {supported}."
                )
                raise ValueError(msg)

            # Defaults if not an existing attribute:
            if not hasattr(data_type, attr):
                if attr == "sum":
                    return sum
                msg = (
                    f"Reducer '{attr}' not found on class '{data_type.__name__}'. "
                    "Expected a class-level reducer accepting Iterable[T]."
                )
                raise AttributeError(msg)

            # Get attribute
            reducer_fnc = getattr(data_type, attr)

            # Ensure attribute is callable
            if not callable(reducer_fnc):
                msg = (
                    f"Attribute '{attr}' on class '{data_type.__name__}' "
                    "is not callable."
                )
                raise TypeError(msg)

            return reducer_fnc

        if not isinstance(data_type, type):
            data_type = type(data_type)

        # Built in reduction types
        if reducer == "first":
            return lambda xs: xs[0]
        if reducer == "last":
            return lambda xs: xs[-1]
        if reducer == "mean":
            return use_attribute(attr=reducer)
        if reducer == "sum":
            return use_attribute(attr=reducer)

        if reducer == "concat":
            # Use backend specific concat method
            if (torch is not None) and issubclass(data_type, torch.Tensor):
                return lambda xs: torch.cat(xs, dim=0)
            if (tf is not None) and issubclass(data_type, tf.Tensor):
                return lambda xs: tf.concat(xs, axis=0)
            if issubclass(data_type, np.ndarray):
                return lambda xs: np.concatenate(xs, axis=0)

            # Otherwise, class must have "concat" attribute
            return use_attribute(attr=reducer)

        return use_attribute(attr=reducer)

    def collapse(
        self,
        axis: str,
        reducer: Callable[[list[T]], T] | str,
    ) -> AxisSeries[T]:
        """
        Collapse an axis by reducing value along it.

        Args:
            axis (str): Axis name to collapse.
            reducer (Callable[[list[T]], T] | str): Reduction callable or reducer label.

        Returns:
            AxisSeries[T]: New series with the specified axis removed.

        Example:
            >>> train_results.losses(...).collapse(  # doctest: +SKIP
            ...     "batch", reducer=LossCollection.mean
            ... )
            >>> train_results.losses(...).collapse(  # doctest: +SKIP
            ...     "batch", reducer="mean"
            ... )
            >>> train_results.tensors(...).collapse(  # doctest: +SKIP
            ...     "batch", reducer=torch.stack
            ... )

        """
        # Validate axis name
        try:
            axis_idx = self.axes.index(axis)
        except ValueError as e:
            msg = f"Unknown axis '{axis}'. Available axes: {self.axes}."
            raise KeyError(msg) from e

        # Group by axis & create new AxisKeys
        grouped: dict[AxisKey, list[T]] = defaultdict(list)
        for k, v in self._data.items():
            reduced_key = k[:axis_idx] + k[axis_idx + 1 :]
            grouped[reduced_key].append(v)
        if not grouped:
            msg = "Cannot collapse an empty AxisSeries."
            raise ValueError(msg)

        # Check reducer (try attribute access if string)
        if isinstance(reducer, str):
            # Infer from first value of first group
            cls = type(next(iter(grouped.values()))[0])
            reducer_fnc = self._resolve_reducer(reducer=reducer, data_type=cls)
        else:
            reducer_fnc = reducer

        # Reduce & return
        reduced = {k: reducer_fnc(vs) for k, vs in grouped.items()}
        new_axes = self.axes[:axis_idx] + self.axes[axis_idx + 1 :]
        return AxisSeries(axes=new_axes, _data=reduced)

    def squeeze(self) -> AxisSeries[T]:
        """
        Remove axes that have only one unique value.

        Returns:
            AxisSeries[T]: Series with singleton axes removed when possible.

        Example:
            >>> my_series.shape  # doctest: +SKIP
            >>> # ('epoch': 1, 'batch': 32)
            >>> my_series.squeeze().shape  # doctest: +SKIP
            >>> # ('batch': 32)

        """
        # Get axis labels with cardinality of 1
        single_axes = {k for k, v in self.shape.items() if v == 1}
        if not single_axes:
            return self

        # Get axis labels if new series
        keep_idxs = [i for i, ax in enumerate(self.axes) if ax not in single_axes]
        new_axes = tuple(self.axes[i] for i in keep_idxs)

        # All axes squeezed -> exactly one entry remains
        if not new_axes:
            value = next(iter(self._data.values()))
            return AxisSeries(axes=(), _data={(): value})

        # Created squeezed series
        new_data: dict[AxisKey, T] = {}
        for k, v in self._data.items():
            new_key = tuple(k[i] for i in keep_idxs)
            new_data[new_key] = v
        return AxisSeries(axes=new_axes, _data=new_data)

    # ================================================
    # Helpers
    # ================================================
    def _interpret_coords(
        self,
        coords: dict[str, Any],
    ) -> dict[str, Hashable]:
        """
        Provides a hook for subclasses to allow for flexible axis key values.

        Description:
            The keys of `coords` can be remapped to the true axes labels as
            defined in this :class:`AxisSeries`. For example, a series could be keyed
            by `ExperimentNode.node_id`. Instead of forcing the user to enter
            node IDs, it is more convenient to support node instances.

        Args:
            coords (dict[str, Any]): Coordinate mapping provided by the caller.

        Returns:
            dict[str, Hashable]: Normalized coordinates keyed by axis name.

        """
        return coords

    def _make_key(self, coords: dict[str, Hashable]) -> AxisKey:
        """
        Create a multi-axis key for supplied coordinates.

        Args:
            coords (dict[str, Hashable]): Axis-to-value mapping.

        Returns:
            AxisKey: Canonical tuple key.

        Raises:
            KeyError: If any axis value is missing.

        """
        # Pass through hook in case subclassed
        coords = self._interpret_coords(coords=coords)

        # Ensure all keys defined in coords
        try:
            return tuple(coords[a] for a in self.axes)
        except KeyError as e:
            msg = f"Missing axis '{e.args[0]}'. Required axes: {self.axes}."
            raise KeyError(msg) from e

    def _coord_selectors(
        self,
        coords: dict[str, Hashable | Callable],
    ) -> list[tuple[int, Callable]]:
        """
        Build per-axis selector functions.

        Each selector is a callable: value -> bool

        Args:
            coords (dict[str, Hashable | Callable]): Axis selectors to normalize.

        Returns:
            list[tuple[int, Callable]]: Axis index with predicate pairs.

        """
        # Pass through hook in case subclassed
        coords = self._interpret_coords(coords=coords)

        selectors = []
        for axis, value in coords.items():
            # Validate axis keys
            try:
                idx = self.axes.index(axis)
            except ValueError as e:
                msg = f"Unknown axis '{axis}'. Available: {self.axes}."
                raise KeyError(msg) from e

            # Check if index is iterable (not str)
            if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
                allowed = set(value)
                selectors.append((idx, lambda x, allowed=allowed: x in allowed))
            elif isinstance(value, Callable):
                selectors.append((idx, value))
            else:
                selectors.append((idx, lambda x, value=value: x == value))
        return selectors

    def _unwrap_key(self, key: AxisKey) -> ExternalKey:
        """
        Convert internal tuple key to external representation.

        Args:
            key (AxisKey): Internal tuple key.

        Returns:
            ExternalKey: Scalar key when there is a single axis, tuple otherwise.

        """
        if len(self.axes) == 1:
            return key[0]
        return key

    def _wrap_key(self, key: ExternalKey) -> AxisKey:
        """
        Convert external key to an internal tuple key.

        Args:
            key (ExternalKey): Key provided by the caller.

        Returns:
            AxisKey: Canonical tuple key.

        """
        if len(self.axes) == 1 and not isinstance(key, tuple):
            full_key = (key,)
        else:
            full_key = ensure_tuple(key)

        # Pass through hook in case subclassed
        coords = {self.axes[i]: v for i, v in enumerate(full_key)}
        coords = self._interpret_coords(coords=coords)
        return tuple(coords[k] for k in self.axes)

    # ================================================
    # Representation
    # ================================================
    def __repr__(self):
        """Return a developer-friendly string representation."""
        return f"AxisSeries(keyed_by={self.axes}, len={len(self)})"
