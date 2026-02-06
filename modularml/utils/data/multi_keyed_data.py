from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Hashable, Iterable, Iterator
from dataclasses import dataclass
from typing import Generic, TypeVar

import numpy as np

from modularml.core.training.loss_record import LossCollection
from modularml.utils.data.formatting import ensure_tuple
from modularml.utils.environment.optional_imports import check_tensorflow, check_torch

torch = check_torch()
tf = check_tensorflow()

AxisKey = tuple[Hashable, ...]  # eg, (epoch, batch, edge)
ExternalKey = (
    Hashable | tuple[Hashable, ...]
)  # scalar when single axis, tuple otherwise
T = TypeVar("T")


@dataclass
class AxisSeries(Generic[T]):
    """Container keyed by a fixed set of semantic axes (eg, epoch, batch)."""

    axes: tuple[str, ...]
    _data: dict[AxisKey, T]

    # ================================================
    # Key wrapping/unwrapping (single-axis convenience)
    # ================================================
    def _unwrap_key(self, key: AxisKey) -> ExternalKey:
        """Convert internal tuple key to external key (scalar if single axis)."""
        if len(self.axes) == 1:
            return key[0]
        return key

    def _wrap_key(self, key: ExternalKey) -> AxisKey:
        """Convert external key to internal tuple key."""
        if len(self.axes) == 1 and not isinstance(key, tuple):
            return (key,)
        return ensure_tuple(key)

    # ================================================
    # Axes properties
    # ================================================
    def axis_values(self, axis: str) -> list[Hashable]:
        """Unique values of the specified axis."""
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
            dict[str, list[Hashable]]:
                Mapping from axis name to its unique values.

        """
        uniq: dict[str, set] = defaultdict(set)
        for k in self._data:  # Use internal tuple keys
            for i, ax_key in enumerate(self.axes):
                uniq[ax_key].add(k[i])

        return {k: list(v) for k, v in uniq.items()}

    def shape(self) -> dict[str, int]:
        """Returns the cardinality of each axis."""
        return {k: len(v) for k, v in self.axes_values().items()}

    # ================================================
    # Direct access & querying
    # ================================================
    def at(self, **coords) -> T:
        """
        Retrieve a value at an exact axis location.

        Values for all axes must be specified.

        Examples:
        ```python
            series.at(epoch=0, batch=3)
        ```

        """
        key = self._make_key(coords)
        return self._data[key]

    def __getitem__(self, key: ExternalKey) -> T:
        internal_key = self._wrap_key(key)
        return self._data[internal_key]

    def where(self, **coords) -> AxisSeries[T]:
        """
        Filter the series by one or more axis values.

        Each coordinate may be a scalar or an iterable of allowed values.

        Examples:
        ```python
            series.where(epoch=1)
            series.where(epoch=[1, 2, 3])
            series.where(epoch=[0, 1], batch=0)
        ```

        """
        selectors = self._coord_selectors(coords)

        filtered = {
            k: v
            for k, v in self._data.items()
            if all(sel(k[idx]) for idx, sel in selectors)
        }
        series = AxisSeries(axes=self.axes, _data=filtered)
        if len(series) == 0:
            msg = "The provided filters result in zero returned items."
            raise ValueError(msg)
        return series

    def one(self) -> T:
        """Return exactly one value or raise."""
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

        Examples:
        ```python
            series.select(epoch=0)
            series.select(epoch=[1, 2])
        ```

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
        """Return all values in insertion order."""
        return list(self._data.values())

    def keys(self) -> list[ExternalKey]:
        """Return all keys (scalar if single axis, tuple otherwise)."""
        return [self._unwrap_key(k) for k in self._data]

    def items(self) -> Iterator[tuple[ExternalKey, T]]:
        """Iterate over key-value pairs (keys are scalar if single axis)."""
        for k, v in self._data.items():
            yield self._unwrap_key(k), v

    def __iter__(self) -> Iterator[ExternalKey]:
        """Iterable over keys (scalar if single axis)."""
        for k in self._data:
            yield self._unwrap_key(k)

    def __len__(self) -> int:
        return len(self._data)

    # ================================================
    # Views
    # ================================================
    def first(self) -> T:
        """Return the first value."""
        return next(iter(self._data.values()))

    def last(self) -> T:
        """Return the last value."""
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
            reducer (str):
                Name of the reduction strategy.
            data_type (type):
                Class of underlying data to be reduced.

        """

        def use_attribute(attr: str) -> Callable[[list[T]], T]:
            # Ensure attribute exists
            try:
                reducer_fnc = getattr(data_type, attr)
            except AttributeError as e:
                msg = (
                    f"Reducer '{attr}' not found on class '{data_type.__name__}'. "
                    "Expected a class-level reducer accepting Iterable[T]."
                )
                raise AttributeError(msg) from e

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
            # LossCollection directly support mean of multiple LCs
            if issubclass(data_type, LossCollection):
                return LossCollection.mean

            # Otherwise, class must have "mean" attribute
            return use_attribute(attr=reducer)
        if reducer == "sum":
            # LossCollection directly support mean of multiple LCs
            if issubclass(data_type, LossCollection):
                return LossCollection.merge

            # Otherwise, class must have "mean" attribute
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
            axis (str):
                Axis name to collapse.
            reducer (Callable[[list[T]], T] | str):
                Function that reduces a list of values into one. Can be a custom
                callable, an attribute name of a class-level method, or one of
                several built-ins (e.g., "first", "last", "mean", "concat", "sum").

        Returns:
            AxisSeries[T]: New series with the specified axis removed.

        Examples:
        ```python
            train_results.losses(...).collapse("batch", reducer=LossCollection.mean)
            train_results.losses(...).collapse("batch", reducer="mean")
            train_results.tensors(...).collapse("batch", reducer=torch.stack)
        ```

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

    # ================================================
    # Helpers
    # ================================================
    def _make_key(self, coords: dict[str, Hashable]) -> AxisKey:
        try:
            return tuple(coords[a] for a in self.axes)
        except KeyError as e:
            msg = f"Missing axis '{e.args[0]}'. Required axes: {self.axes}."
            raise KeyError(msg) from e

    def _coord_selectors(
        self,
        coords: dict[str, Hashable],
    ) -> list[tuple[int, Callable]]:
        """
        Build per-axis selector functions.

        Each selector is a callable: value -> bool
        """
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
            else:
                selectors.append((idx, lambda x, value=value: x == value))

        return selectors

    # ================================================
    # Helpers
    # ================================================
    def __repr__(self):
        return f"AxisSeries(keyed_by={self.axes}, len={len(self)})"
