"""Sampled view container for aligned :class:`BatchView` streams."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass

from modularml.core.data.batch_view import BatchView
from modularml.utils.representation.summary import Summarizable


@dataclass(frozen=True)
class SampledView(Mapping[str, list[BatchView]], Summarizable):
    """
    Immutable container for sampler outputs.

    Description:
        Wraps a mapping of named sampler output streams to lists of
        :class:`BatchView` objects. Each stream represents a lazily-evaluated
        sequence of batches.

        Iteration over :class:`SampledView` yields *aligned batch groups*
        (``dict[str, BatchView]`` — one ``BatchView`` per stream at the same index).

    Example:
        ```python
        sampled = SampledView({"main": [...], "aux": [...]})
        sampled["main"]  # list[BatchView]
        sampled.main  # list[BatchView]
        for batch_map in sampled:
            ... batch_map["main"]  # BatchView
        ```

    """

    streams: dict[str, list[BatchView]]

    # ================================================
    # Validation
    # ================================================
    def __post_init__(self):
        if not self.streams:
            raise ValueError("SampledView must contain at least one stream.")

        lengths = set()
        for name, batches in self.streams.items():
            if not isinstance(name, str):
                msg = f"Stream keys must be str, got {type(name)}"
                raise TypeError(msg)
            if not isinstance(batches, list):
                msg = f"Stream '{name}' must map to list[BatchView]"
                raise TypeError(msg)
            for b in batches:
                if not isinstance(b, BatchView):
                    msg = f"Stream '{name}' contains non-BatchView item: {type(b)}"
                    raise TypeError(msg)
            lengths.add(len(batches))

        if len(lengths) > 1:
            msg = f"All streams must have the same number of batches. Got stream lengths: {sorted(lengths)}"
            raise ValueError(msg)

    # ================================================
    # Mapping interface
    # ================================================
    def __getitem__(self, key: str) -> list[BatchView]:
        """Return the :class:`BatchView` list for the given stream name."""
        return self.streams[key]

    def __iter__(self) -> Iterator[dict[str, BatchView]]:
        """
        Iterate over aligned BatchViews across all streams.

        Yields:
            dict[str, BatchView]:
                Mapping of stream name to BatchView for a single batch index.

        """
        n_batches = self.num_batches
        stream_names = list(self.streams.keys())

        for i in range(n_batches):
            yield {name: self.streams[name][i] for name in stream_names}

    def __len__(self) -> int:
        """Return the number of streams."""
        return len(self.streams)

    # ================================================
    # Stream accessors
    # ================================================
    @property
    def stream_names(self) -> list[str]:
        """Names of all output streams."""
        return list(self.streams.keys())

    @property
    def num_streams(self) -> int:
        """Number of aligned streams."""
        return len(self.streams)

    @property
    def num_batches(self) -> int:
        """Number of aligned batches across all streams."""
        first_stream = next(iter(self.streams.values()))
        return len(first_stream)

    def get_stream(self, name: str) -> list[BatchView]:
        """
        Explicit stream accessor.

        Args:
            name (str): Stream name.

        Returns:
            list[BatchView]: The :class:`BatchView` list for the stream.

        """
        return self.streams[name]

    # ================================================
    # Attribute access
    # ================================================
    def __getattr__(self, name: str) -> list[BatchView]:
        # Guard against recursion
        if name == "streams":
            raise AttributeError(name)

        # Try to grab attribute from stream
        if name in self.streams:
            return self.streams[name]

        msg = f"{self.__class__.__name__} has no stream '{name}'. Available streams: {self.stream_names}"
        raise AttributeError(msg)

    # ================================================
    # Utilities
    # ================================================
    def to_dict(self) -> dict[str, list[BatchView]]:
        """Unwrap to a plain ``dict[str, list[BatchView]]``."""
        return dict(self.streams)

    # ================================================
    # Representation
    # ================================================
    def _summary_rows(self) -> list[tuple]:
        rows: list[tuple] = []
        rows.append(
            ("streams", [(k, f"{len(v)} batches") for k, v in self.streams.items()]),
        )

        for name, batches in self.streams.items():
            if batches:
                rows.append((name, batches[0]._summary_rows()))
            else:
                rows.append((name, "empty"))

        return rows

    def __repr__(self) -> str:
        return (
            f"SampledView(streams={self.stream_names}, num_batches={self.num_batches})"
        )
