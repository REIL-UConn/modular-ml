"""Abstract base sampler with batching utilities and state tracking."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from modularml.core.data.batch_view import BatchView
from modularml.core.data.featureset import FeatureSet
from modularml.core.data.featureset_view import FeatureSetView
from modularml.core.data.sampled_view import SampledView
from modularml.core.data.schema_constants import ROLE_DEFAULT, STREAM_DEFAULT
from modularml.core.io.protocols import Configurable, Stateful
from modularml.core.sampling.batcher import Batcher
from modularml.utils.data.comparators import deep_equal
from modularml.utils.data.formatting import ensure_list
from modularml.utils.environment.environment import IN_NOTEBOOK
from modularml.utils.progress_bars.progress_task import ProgressTask

if TYPE_CHECKING:
    from collections.abc import Iterator

    from numpy.typing import NDArray


@dataclass(frozen=True)
class Samples:
    """
    Container describing aligned role indices (and optional weights) for a stream.

    Attributes:
        role_indices (dict[str, NDArray[np.int_]]):
            Absolute sample indices keyed by role name.
        role_weights (dict[str, NDArray[np.float32]] | None):
            Optional per-role weights aligned with ``role_indices``.

    """

    role_indices: dict[str, NDArray[np.int_]]
    role_weights: dict[str, NDArray[np.float32]] | None = None


@dataclass(frozen=True)
class SamplerStreamSpec:
    """
    Specification of sampler output streams and roles.

    Description:
        Sampler streams should be a static property on each sampler class so that
        downstream consumers know which streams/roles exist before sampling.

    Attributes:
        stream_names (tuple[str, ...]): Names for each logical output stream.
        roles (tuple[str, ...]): Ordered list of roles produced within each stream.

    """

    stream_names: tuple[str, ...]
    roles: tuple[str, ...]


class BaseSampler(Configurable, Stateful, ABC):
    """
    Base class for all samplers that emit aligned :class:`BatchView` objects.

    Description:
        Accepts one or more :class:`FeatureSet` or :class:`FeatureSetView` inputs,
        normalizes them to views, and produces zero-copy :class:`BatchView` objects.

    Attributes:
        sources (dict[str, FeatureSetView] | None): Bound source views keyed by label.
        rng (np.random.Generator): Random generator used for shuffling.
        show_progress (bool): Whether materialization uses progress reporting.
        batcher (Batcher): Helper responsible for turning indices into batches.

    """

    __SPECS__ = SamplerStreamSpec(
        stream_names=(STREAM_DEFAULT,),
        roles=(ROLE_DEFAULT,),
    )

    def __init__(
        self,
        *,
        batch_size: int = 1,
        shuffle: bool = False,
        group_by: list[str] | None = None,
        group_by_role: str = ROLE_DEFAULT,
        stratify_by: list[str] | None = None,
        stratify_by_role: str = ROLE_DEFAULT,
        strict_stratification: bool = True,
        drop_last: bool = False,
        seed: int | None = None,
        show_progress: bool = True,
        sources: list[FeatureSet | FeatureSetView] | None = None,
    ):
        """
        Initialize sampler hyperparameters and optional sources.

        Args:
            batch_size (int):
                Number of samples per batch.

            shuffle (bool):
                Whether to shuffle role indices prior to batching.

            group_by (list[str] | None):
                Column keys used to keep groups together per batch.

            group_by_role (str):
                Role whose indices determine grouping when multiple roles exist.

            stratify_by (list[str] | None):
                Column keys used to balance strata per batch.

            stratify_by_role (str):
                Role used when interpreting `stratify_by`.

            strict_stratification (bool):
                Whether batching stops when any stratum exhausts.

            drop_last (bool):
                Whether to drop incomplete batches.

            seed (int | None):
                Random-seed passed to the internal RNG.

            show_progress (bool):
                Whether to enable progress updates during materialization.

            sources (list[FeatureSet | FeatureSetView] | None):
                Optional data sources to bind immediately.

        Raises:
            ValueError:
                If both `group_by` and `stratify_by` are provided simultaneously.

        """
        self.sources: dict[str, FeatureSetView] | None = None
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)
        self.show_progress = show_progress

        if stratify_by and group_by:
            msg = "Both `group_by` and `stratify_by` cannot be applied at the same."
            raise ValueError(msg)

        self.batcher = Batcher(
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            group_by=ensure_list(group_by),
            group_by_role=group_by_role,
            stratify_by=ensure_list(stratify_by),
            stratify_by_role=stratify_by_role,
            strict_stratification=strict_stratification,
            seed=seed,
        )

        # list of BatchViews keyed by output stream
        self._sampled: SampledView | None = None
        self._stream_to_source_label: dict[str, str] = {}

        # Optional progress bar
        self._progress_task = ProgressTask(
            style="sampling",
            description=f"{type(self).__name__}",
            total=None,
            enabled=show_progress,
            persist=IN_NOTEBOOK,
        )

        # Define expected output streams
        self._stream_names: list[str] = [STREAM_DEFAULT]

        if sources is not None:
            self.bind_sources(sources)

    def __eq__(self, other):
        if not isinstance(other, BaseSampler):
            msg = (
                "Equality can only be compared between two samplers. "
                f"Received: {type(other)}."
            )
            raise TypeError(msg)

        # Check config first
        if self.get_config() != other.get_config():
            return False

        # Then check state
        return deep_equal(self.get_state(), other.get_state())

    __hash__ = None

    # =====================================================
    # Properties
    # =====================================================
    @property
    def is_bound(self) -> bool:
        """Whether sources has been attached."""
        return self.sources is not None

    @property
    def is_materialized(self) -> bool:
        """Whether batches have been materialized."""
        return self.is_bound and (self._sampled is not None)

    @property
    def sampled(self) -> SampledView:
        """
        Return the materialized :class:`SampledView` (materializing if needed).

        Returns:
            SampledView: Object containing batches for each stream.

        Raises:
            RuntimeError: If sources have not been bound yet.

        """
        if self.is_bound and not self.is_materialized:
            self.materialize_batches()
        if not self.is_bound:
            raise RuntimeError("Sampler has no bound source. Call bind_source().")
        return self._sampled

    @property
    def stream_names(self) -> list[str]:
        """Names of all output streams."""
        return self.__SPECS__.stream_names

    @property
    def num_streams(self) -> int:
        """Number of output streams generated by this sampler."""
        return len(self.stream_names)

    @property
    def role_names(self) -> list[str]:
        """
        Names of roles produced by this sampler.

        All output streams have the same set of roles.
        """
        return self.__SPECS__.roles

    @property
    def num_roles(self) -> int:
        """
        Number of sample roles per output stream.

        All output streams have the same set of roles.
        """
        return len(self.role_names)

    # ================================================
    # SampledView interface
    # ================================================
    def __iter__(self) -> Iterator[dict[str, BatchView]]:
        """
        Iterate over aligned BatchViews across all streams.

        Yields:
            dict[str, BatchView]:
                Mapping of stream name to BatchView for a single batch index.

        """
        yield from self.sampled

    @property
    def num_batches(self) -> int:
        """
        Number of aligned batches across all streams.

        Notes:
            This will trigger batch materialization if a source if bound but batches
            have not been created. It is recommended to first call `materialize_batches`
            directly to configure the execution process.

        """
        return self.sampled.num_batches

    def get_stream(self, name: str) -> list[BatchView]:
        """
        Return the batches for a specific stream.

        Args:
            name (str): Stream label returned by :attr:`stream_names`.

        Returns:
            list[BatchView]: Sequence of batches for the requested stream.

        Raises:
            KeyError: If the stream name does not exist.

        """
        return self.sampled.get_stream(name=name)

    def get_batches(self, stream: str | None = None) -> list[BatchView]:
        """
        Return the batches for a stream, defaulting when only one exists.

        Args:
            stream (str | None):
                Stream label. When omitted, only valid if a single stream exists.

        Returns:
            list[BatchView]: Batches emitted for the requested stream.

        Raises:
            ValueError: If `stream` is omitted but multiple streams exist.

        """
        if stream is None:
            if self.num_streams != 1:
                msg = "This sampler outputs multiple streams. `stream` cannot be None."
                raise ValueError(msg)
            stream = self.stream_names[0]
        return self.get_stream(name=stream)

    # ================================================
    # Source Binding (batch instantiation)
    # ================================================
    def _build_sampled_view(self) -> SampledView:
        """
        Construct and return a :class:`SampledView` from bound sources.

        Returns:
            SampledView: Materialized batches for each stream.

        Raises:
            RuntimeError: If :meth:`build_samples` fails to produce indices.

        """
        if self._progress_task is not None:
            self._progress_task.reset()
            self._progress_task.description = f"Sampling {list(self.sources)}"

        try:
            samples_by_stream: dict[tuple[str, str], Samples] = self.build_samples()
        finally:
            # Ensure cleanup even if never started
            self._progress_task.finish()

        # Construct SampledView
        streams: dict[str, list[BatchView]] = {}
        self._stream_to_source_label = {}
        for (stream_lbl, source_lbl), samples in samples_by_stream.items():
            streams[stream_lbl] = self.batcher.batch(
                view=self.sources[source_lbl],
                role_indices=samples.role_indices,
                role_weights=samples.role_weights,
            )
            self._stream_to_source_label[stream_lbl] = source_lbl

        return SampledView(streams=streams)

    def bind_sources(self, sources: list[FeatureSet | FeatureSetView]):
        """
        Attach :class:`FeatureSet` or :class:`FeatureSetView` sources to this sampler.

        Args:
            sources (list[FeatureSet | FeatureSetView]):
                Objects to bind; each is converted to a view internally.

        Raises:
            TypeError: If any source is neither a FeatureSet nor a FeatureSetView.

        """
        src_views: dict[str, FeatureSetView] = {}
        for src in ensure_list(sources):
            if isinstance(src, FeatureSet):
                view = src.to_view()
            elif isinstance(src, FeatureSetView):
                view = src
            else:
                raise TypeError(
                    "Sampler source must be a FeatureSet or FeatureSetView.",
                )
            src_views[view.source.label] = view

        self.sources = src_views
        self._sampled = None

    def materialize_batches(
        self,
        *,
        show_progress: bool = True,
    ):
        """
        Execute sampling of bound sources and instantiate batches.

        Args:
            show_progress (bool): Whether to display progress updates during materialization.

        Raises:
            RuntimeError: If sources have not been bound via :meth:`bind_sources`.

        """
        if not self.is_bound:
            raise RuntimeError("Sampler has no bound source. Call bind_source() first.")
        self.show_progress = show_progress
        if not self.is_materialized:
            self._sampled = self._build_sampled_view()

    # ================================================
    # Abstract Methods
    # ================================================
    @abstractmethod
    def build_samples(self) -> dict[tuple[str, str], Samples]:
        """
        Build sample streams from bound source data.

        Returns:
            dict[tuple[str, str], Samples]:
                Mapping with keys of `(stream_name, source_label)`.

        Raises:
            NotImplementedError: Must be supplied by subclasses.

        """
        raise NotImplementedError

    # ================================================
    # Configurable
    # ================================================
    def get_config(self) -> dict[str, Any]:
        """
        Return configuration required to reconstruct this sampler.

        Returns:
            dict[str, Any]: Configuration excluding bound sources and batch state.

        """
        return {
            "batch_size": self.batcher.batch_size,
            "shuffle": self.batcher.shuffle,
            "group_by": self.batcher.group_by,
            "group_by_role": self.batcher.group_by_role,
            "stratify_by": self.batcher.stratify_by,
            "stratify_by_role": self.batcher.stratify_by_role,
            "strict_stratification": self.batcher.strict_stratification,
            "drop_last": self.batcher.drop_last,
            "seed": self.batcher.seed,
            "show_progress": self.show_progress,
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> BaseSampler:
        """
        Construct a sampler from configuration.

        Args:
            config (dict[str, Any]):
                Serialized sampler configuration (must include `sampler_name`).

        Returns:
            BaseSampler: Unfitted sampler instance.

        Raises:
            KeyError: If `sampler_name` is missing.

        """
        from modularml.samplers import sampler_registry

        if "sampler_name" not in config:
            msg = (
                "Sampler config must store 'sampler_name' if "
                "using BaseSampler to instantiate."
            )
            raise KeyError(msg)

        sampler_cls: BaseSampler = sampler_registry[str(config["sampler_name"])]
        return sampler_cls.from_config(config)

    # ================================================
    # Stateful
    # ================================================
    def get_state(self) -> dict[str, Any]:
        """
        Return runtime state (sources, RNG, and batches) for this sampler.

        Returns:
            dict[str, Any]: State payload consumable by :meth:`set_state`.

        """
        # Construct reference of each source
        # This is later re-linked via ExperimentContext
        source_cfgs: dict[str, dict[str, Any]] | None = None
        if self.sources is not None:
            source_cfgs = {}
            for src_lbl, src_view in self.sources.items():
                source_cfgs[src_lbl] = src_view.get_config()
        state = {
            "sources_config": source_cfgs,
            "sources_label_map": self._stream_to_source_label,
            "sampled_config": None,
        }

        # Record batch views, if already materialized
        if self._sampled is not None:
            sampled_cfg = defaultdict(list)
            for stream_lbl, bv_list in self._sampled.streams.items():
                for bv in bv_list:
                    bv_cfg = {
                        "role_indices": bv.role_indices,
                        "role_indice_weights": bv.role_indice_weights,
                    }
                    sampled_cfg[stream_lbl].append(bv_cfg)
            state["sampled_config"] = sampled_cfg

        return state

    def set_state(self, state: dict[str, Any]) -> None:
        """
        Restore runtime state captured via :meth:`get_state`.

        Args:
            state (dict[str, Any]): State payload produced by :meth:`get_state`.

        """
        # If state has sources, rebuild views
        if state.get("sources_config") is None:
            # Can't set anything without a source
            # Sampler is in a config-only runtime state
            return

        # To rebuild sources, we also need to know which sources each stream used
        stream_to_source_label: dict[str, str] = state.get("sources_label_map")
        if stream_to_source_label is None:
            msg = "Sampler state is missing required keyword: 'sources_label_map'."
            raise RuntimeError(msg)
        self._stream_to_source_label = stream_to_source_label

        # Rebuild sources
        sources: dict[str, FeatureSetView] = {}
        for src_lbl, src_view_cfg in state["sources_config"].items():
            sources[src_lbl] = FeatureSetView.from_config(src_view_cfg)

        # If state has batches saved, we can manually set source and batches
        sampled_cfg: dict[str, list[dict]] | None = state.get("sampled_config")
        if sampled_cfg is not None:
            # Construct SampledView from psuedo-config
            # Need to relink source view to stream
            streams: dict[str, list[BatchView]] = defaultdict(list)
            for stream_lbl, bv_cfg_list in sampled_cfg.items():
                for bv_cfg in bv_cfg_list:
                    bv = BatchView(
                        source=sources[self._stream_to_source_label[stream_lbl]].source,
                        role_indices=bv_cfg["role_indices"],
                        role_indice_weights=bv_cfg["role_indice_weights"],
                    )
                    streams[stream_lbl].append(bv)
            self._sampled = SampledView(streams=streams)
            self.sources = sources

        # Otherwise, we should call bind_sources
        else:
            self.bind_sources(sources=list(sources.values()))

    # ================================================
    # Serialization
    # ================================================
    def save(self, filepath: Path, *, overwrite: bool = False) -> Path:
        """
        Serialize this sampler to disk.

        Args:
            filepath (Path):
                Destination path; suffix may be adjusted to match ModularML
                conventions.
            overwrite (bool):
                Whether to overwrite existing files.

        Returns:
            Path: Actual file path written by the serializer.

        """
        from modularml.core.io.serialization_policy import SerializationPolicy
        from modularml.core.io.serializer import serializer

        return serializer.save(
            self,
            filepath,
            policy=SerializationPolicy.REGISTERED,
            overwrite=overwrite,
        )

    @classmethod
    def load(cls, filepath: Path, *, allow_packaged_code: bool = False) -> BaseSampler:
        """
        Load a sampler from disk.

        Args:
            filepath (Path):
                File path to a serialized sampler artifact.
            allow_packaged_code (bool):
                Whether packaged code execution is allowed.

        Returns:
            BaseSampler: Reloaded sampler instance.

        """
        from modularml.core.io.serializer import _enforce_file_suffix, serializer

        # Append proper sufficx only if no suffix is given
        if Path(filepath).suffix == "":
            filepath = _enforce_file_suffix(path=filepath, cls=cls)

        return serializer.load(filepath, allow_packaged_code=allow_packaged_code)
