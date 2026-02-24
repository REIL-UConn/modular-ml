"""Base results container utilities shared across phases."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Literal, TypeVar

import numpy as np

from modularml.core.data.batch import Batch
from modularml.core.data.execution_context import ExecutionContext
from modularml.core.data.schema_constants import ROLE_DEFAULT
from modularml.core.experiment.callbacks.callback import CallbackResult
from modularml.core.experiment.experiment_context import ExperimentContext
from modularml.core.experiment.results.metric_store import MetricDataSeries, MetricStore
from modularml.core.references.execution_reference import TensorLike
from modularml.core.topology.graph_node import GraphNode
from modularml.core.training.loss_record import LossRecord
from modularml.utils.data.data_format import DataFormat, normalize_format
from modularml.utils.data.multi_keyed_data import AxisSeries
from modularml.utils.data.scaling import unscale_sample_data
from modularml.utils.topology.graph_search_utils import find_upstream_featuresets

if TYPE_CHECKING:
    from collections.abc import Hashable

    from modularml.core.data.featureset import FeatureSet
    from modularml.core.data.featureset_view import FeatureSetView
    from modularml.core.data.sample_data import SampleData

T = TypeVar("T")


# ================================================
# AxisSeries Wrappers
# ================================================
@dataclass
class ExecutionDataSeries(AxisSeries[ExecutionContext]):
    """
    ExecutionContext objects keyed by (epoch, batch).

    Attributes:
        axes (tuple[str, ...]): Axis labels describing the series dimensions.
        _data (dict[tuple, ExecutionContext]): Underlying mapping of keys to contexts.
        supported_reduction_methods (ClassVar[set[str]]):
            Allowed reducers for :meth:`AxisSeries.collapse`.

    """

    supported_reduction_methods: ClassVar[set[str]] = {"first", "last"}

    def __repr__(self):
        return f"ExecutionDataSeries(keyed_by={self.axes}, len={len(self)})"


@dataclass
class BatchDataSeries(AxisSeries[Batch]):
    """
    Batch objects keyed by (epoch, batch).

    Attributes:
        axes (tuple[str, ...]): Axis labels describing the series dimensions.
        _data (dict[tuple, Batch]): Underlying mapping of batch keys to Batch objects.
        supported_reduction_methods (ClassVar[set[str]]):
            Allowed reducers for :meth:`AxisSeries.collapse`.

    """

    supported_reduction_methods: ClassVar[set[str]] = {"first", "last"}

    # ================================================
    # Data Casting
    # ================================================
    def to_format(self, fmt: DataFormat) -> BatchDataSeries:
        """
        Cast all underlying tensors to the specified format.

        Args:
            fmt (DataFormat): Target tensor format (e.g., `torch`, `np`).

        Returns:
            BatchDataSeries: New series with converted batches.

        """
        data = {k: b.to_format(fmt=fmt) for k, b in self.data.items()}
        return BatchDataSeries(
            axes=self.axes,
            _data=data,
        )

    # ================================================
    # Representation
    # ================================================
    def __repr__(self):
        return f"BatchDataSeries(keyed_by={self.axes}, len={len(self)})"


@dataclass
class TensorDataSeries(AxisSeries[TensorLike]):
    """
    TensorLike objects keyed by (epoch, batch).

    Attributes:
        axes (tuple[str, ...]): Axis labels describing the series dimensions.
        _data (dict[tuple, TensorLike]): Underlying mapping of axis keys to tensors.
        supported_reduction_methods (ClassVar[set[str]]):
            Allowed reducers for :meth:`AxisSeries.collapse`.

    """

    supported_reduction_methods: ClassVar[set[str]] = {"first", "last", "concat"}

    def __repr__(self):
        return f"TensorDataSeries(keyed_by={self.axes}, len={len(self)})"


@dataclass
class LossDataSeries(AxisSeries[LossRecord]):
    """
    Loss records keyed by (epoch, batch, label).

    Attributes:
        axes (tuple[str, ...]): Axis labels describing the series dimensions.
        _data (dict[tuple, LossRecord]): Mapping of axis keys to loss records.
        supported_reduction_methods (ClassVar[set[str]]): Allowed reducers for
            :meth:`AxisSeries.collapse`.

    """

    supported_reduction_methods: ClassVar[set[str]] = {
        "mean",
        "sum",
        "first",
        "last",
    }

    # ================================================
    # Data Casting
    # ================================================
    def to_float(self) -> LossDataSeries:
        """
        Cast all underlying loss record values to floats.

        Returns:
            LossDataSeries: Series with float-valued loss records.

        """
        return LossDataSeries(
            axes=self.axes,
            _data={k: lr.to_float() for k, lr in self.data.items()},
        )

    # ================================================
    # Accessors / Querying
    # ================================================
    @property
    def trainable(self) -> float:
        """
        Retrieves the total trainable loss value of all records in this collection.

        Returns:
            float: Sum of trainable components.

        Example:
            To get the trainable loss for a specific epoch/batch/label, use:

            >>> series = LossDataSeries(...)  # doctest: +SKIP
            >>> series.where(epoch=1, ...).trainable # doctest: +SKIP

        """
        vals = [lr.trainable for lr in self.values() if lr.trainable is not None]
        return sum(vals) if vals else 0.0

    @property
    def auxiliary(self) -> float:
        """
        Retrieves the total auxiliary loss value of all records in this collection.

        Returns:
            float: Sum of auxiliary components.

        Example:
            To get the auxiliary loss for a specific epoch/batch/label, use:

            >>> series = LossDataSeries(...)  # doctest: +SKIP
            >>> series.where(epoch=1, ...).auxiliary # doctest: +SKIP

        """
        vals = [lr.auxiliary for lr in self.values() if lr.auxiliary is not None]
        return sum(vals) if vals else 0.0

    # ================================================
    # Representation
    # ================================================
    def __repr__(self):
        return f"LossDataSeries(keyed_by={self.axes}, len={len(self)})"


@dataclass
class CallbackDataSeries(AxisSeries[CallbackResult]):
    """
    CallbackResult objects keyed by (kind, label, epoch, batch, edge).

    Attributes:
        axes (tuple[str, ...]): Axis labels describing the series dimensions.
        _data (dict[tuple, CallbackResult]): Mapping of keys to callback entries.
        supported_reduction_methods (ClassVar[set[str]]):
            Allowed reducers for :meth:`AxisSeries.collapse`.

    """

    supported_reduction_methods: ClassVar[set[str]] = {"first", "last"}

    def __repr__(self):
        return f"CallbackDataSeries(keyed_by={self.axes}, len={len(self)})"


# ================================================
# PhaseResults
# ================================================
@dataclass
class PhaseResults:
    """
    Base container for per-phase execution data, callbacks, and metrics.

    Attributes:
        label (str): Phase label.
        _execution (list[ExecutionContext]): Ordered execution contexts.
        _callbacks (list[CallbackResult]): Recorded callback outputs.
        _metrics (MetricStore): Store of scalar metrics logged during execution.
        _series_cache (dict[tuple, Any]): Cache of memoized AxisSeries queries.

    """

    label: str

    # Phase contexts ordered by execution time
    _execution: list[ExecutionContext] = field(default_factory=list, repr=False)

    # Results produced by callbacks; keyed by callback label
    _callbacks: list[CallbackResult] = field(default_factory=list, repr=False)

    # Named scalar metrics recorded during execution
    _metrics: MetricStore = field(default_factory=MetricStore, repr=False)

    # Memoized AxisSeries objects (invalidated on mutation)
    _series_cache: dict[tuple[Hashable, ...], Any] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    # ================================================
    # Runtime Modifiers
    # ================================================
    def add_execution_context(self, ctx: ExecutionContext):
        """
        Record a new execution context.

        Args:
            ctx (ExecutionContext): Context to append.

        """
        # Ensure all tensors are detached/copied
        for k in ctx.outputs:
            # Detached in-place
            ctx.outputs[k].detach_tensors()

        # Break any loss links too
        ctx.losses = ctx.losses.to_float()

        # Record cleaned ctx
        self._execution.append(ctx)
        self._series_cache.clear()

    def add_callback_result(self, cb_res: CallbackResult):
        """
        Record a new callback result.

        Args:
            cb_res (CallbackResult): Result emitted by a callback.

        """
        self._callbacks.append(cb_res)
        self._series_cache.clear()

    # ================================================
    # Helpers
    # ================================================
    def _resolve_node(self, node: str | GraphNode) -> GraphNode:
        """Resolve a GraphNode from a string ID/label or GraphNode instance."""
        exp_ctx = ExperimentContext.get_active()
        if isinstance(node, str):
            return exp_ctx.get_node(val=node, enforce_type="GraphNode")
        if isinstance(node, GraphNode):
            return node
        msg = (
            f"Invalid `node` type. Must be string or GraphNode. Received: {type(node)}."
        )
        raise TypeError(msg)

    def _cache_get(self, key: tuple[Hashable, ...]):
        """Get cached data for key."""
        return self._series_cache.get(key)

    def _cache_set(self, key: tuple[Hashable, ...], value):
        """Set cache for key."""
        self._series_cache[key] = value
        return value

    # ================================================
    # Representation
    # ================================================
    def __repr__(self):
        n_exec = len(self._execution)
        n_cb = len(self._callbacks)
        return (
            f"PhaseResults(label='{self.label}', executions={n_exec}, callbacks={n_cb})"
        )

    # ================================================
    # Execution Data & Loss Querying
    # ================================================
    def execution_contexts(self) -> ExecutionDataSeries:
        """
        Returns a query interface for execution contexts.

        Data is keyed by `(epoch, batch)`.
        """
        # Check cache
        cache_key = ("execution_contexts",)
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        keyed_data: dict[tuple[int, int], ExecutionContext] = {
            (ctx.epoch_idx, ctx.batch_idx): ctx for ctx in self._execution
        }
        return self._cache_set(
            cache_key,
            ExecutionDataSeries(axes=("epoch", "batch"), _data=keyed_data),
        )

    def batches(self, node: str | GraphNode) -> BatchDataSeries:
        """
        Returns a query interface for batches on a specific node.

        Args:
            node (str | GraphNode):
                The node to filter batches to. Can be the node instance, its ID,
                or its label.

        Returns:
            BatchDataSeries:
                A keyed iterable over all batches executed on `node`.
                Data is keyed by `(epoch, batch)`.

        """
        # Resolve node
        graph_node = self._resolve_node(node=node)

        # Check cache
        cache_key = ("batches", graph_node.node_id)
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        # Key batches by execution scope
        keyed_data: dict[tuple[int, int], Batch] = {
            (ctx.epoch_idx, ctx.batch_idx): ctx.outputs[graph_node.node_id]
            for ctx in self._execution
        }
        return self._cache_set(
            cache_key,
            BatchDataSeries(axes=("epoch", "batch"), _data=keyed_data),
        )

    def tensors(
        self,
        node: str | GraphNode,
        domain: Literal["outputs", "targets", "tags", "sample_uuids"],
        *,
        role: str = ROLE_DEFAULT,
        fmt: DataFormat | None = None,
        unscale: bool = False,
    ) -> TensorDataSeries:
        """
        Returns a query interface for tensors related to a specific domain.

        Args:
            node (str | GraphNode):
                The node to filter data to. Can be the node instance, its ID, or
                its label.
            domain (Literal["outputs", "targets", "tags"]):
                The domain of data to return:
                * outputs: the tensors produced by the node forward pass
                * targets: the expected output tensors (only meaningful
                    for tail nodes)
                * tags: any tracked tags during the node's forward pass
            role (str, optional):
                If the data executed during this phase was produced by a multi-role
                sampler, the role of the data to returned must be specified.
                Defaults to `ROLE_DEFAULT`.
            fmt (DataFormat, optional):
                The format to cast the returned tensors to. If None, the as-produced
                format is used. Defaults to None.
            unscale (bool, optional):
                Whether to inverse any applied scalers to these tensors.
                Note that this is only possible when `node` refers to a tail node,
                and `domain` is one of `["outputs", "targets"]`.

        Returns:
            TensorDataSeries:
                A keyed iterable over all tensors related to `node`.
                Data is keyed by `(epoch, batch)`.

        """
        # Resolve node
        graph_node = self._resolve_node(node=node)
        fmt = normalize_format(fmt) if fmt is not None else None

        # Check cache
        cache_key = (
            "tensors",
            graph_node.node_id,
            domain,
            role,
            fmt.value if isinstance(fmt, DataFormat) else None,
            unscale,
        )
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        # Get batch data
        batch_series = self.batches(node=graph_node)

        keyed_tensors: dict[tuple[int, int], TensorLike] = {}
        for ax_key, b in batch_series.data.items():
            # Get sample data for role
            sd: SampleData = b.get_data(role=role)

            # Cast to fmt (if defined)
            if fmt is not None:
                sd = sd.to_format(fmt=fmt)

            # Unscale (if defined)
            if unscale:
                sd = unscale_sample_data(data=sd, from_node=node)

            # Get tensor-like data from domain
            tlike = sd.get_domain_data(domain=domain)

            keyed_tensors[ax_key] = tlike

        return self._cache_set(
            cache_key,
            TensorDataSeries(axes=("epoch", "batch"), _data=keyed_tensors),
        )

    def losses(self, node: str | GraphNode) -> LossDataSeries:
        """
        Returns a query interface for losses applied to a specific node.

        Args:
            node (str | GraphNode):
                The node to filter losses to. Can be the node instance,
                its ID, or its label.

        Returns:
            LossDataSeries:
                A keyed iterable over all loss records.
                Data is keyed by `(epoch, batch, label)`

        """
        # Resolve node
        graph_node = self._resolve_node(node=node)

        # Check cache
        cache_key = ("losses", graph_node.node_id)
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        # Key losses by execution scope
        keyed_lrs: dict[tuple[int, int, str], LossRecord] = {}
        for ctx in self._execution:
            if ctx.losses is None:
                continue
            lrs = ctx.losses.select(node=graph_node.node_id)
            for lr in lrs:
                key = (ctx.epoch_idx, ctx.batch_idx, lr.label)
                if key in keyed_lrs:
                    msg = f"Key already exists for LossRecord: {key}"
                    raise KeyError(msg)
                keyed_lrs[key] = lr

        return self._cache_set(
            cache_key,
            LossDataSeries(axes=("epoch", "batch", "label"), _data=keyed_lrs),
        )

    # ================================================
    # Source Data Access
    # ================================================
    def source_views(
        self,
        node: str | GraphNode,
        *,
        role: str = ROLE_DEFAULT,
        epoch: int | None = None,
        batch: int | None = None,
    ) -> dict[str, FeatureSetView]:
        """
        Get the source FeatureSetViews that contributed data to the given node.

        Description:
            Traces the node back to its upstream FeatureSets, collects all
            unique sample UUIDs from execution results, and returns a view
            of each upstream FeatureSet filtered to only the samples used.

            When no `epoch` or `batch` is specified, only a single epoch
            is scanned since all epochs draw from the same sample pool.

            Note that the returned views contain only unique sample UUIDs used
            in generating these phase results. They are not a 1-to-1 mapping
            of result sample to source sample. Use `tensors()` to get exact
            execution data.

        Args:
            node (str | GraphNode):
                The node to trace upstream from. Can be the node instance,
                its ID, or its label.
            role (str, optional):
                Restrict to samples from this role only. Defaults to `ROLE_DEFAULT`.
            epoch (int | None, optional):
                Restrict to samples from this epoch only.
            batch (int | None, optional):
                Restrict to samples from this batch only.

        Returns:
            dict[str, FeatureSetView]:
                A mapping of FeatureSet label to FeatureSetView containing
                only the samples used during execution.

        """
        graph_node = self._resolve_node(node=node)

        # Determine which context to concat
        if epoch is None and batch is None:
            # All epochs use same samples -> scan only 1st epoch
            first_epoch = self._execution[0].epoch_idx
            ctxs = self.execution_contexts().select(epoch=first_epoch)
        elif epoch is not None and batch is None:
            ctxs = self.execution_contexts().select(epoch=epoch)
        elif epoch is None and batch is not None:
            ctxs = self.execution_contexts().select(batch=batch)
        else:
            ctxs = self.execution_contexts().select(epoch=epoch, batch=batch)

        # Collect unique sample ids from matching contexts
        all_uuids: set[str] = set()
        for ctx in ctxs:
            batch_uuids = (
                ctx.outputs[graph_node.node_id].get_data(role=role).sample_uuids
            )
            all_uuids.update(np.asarray(batch_uuids).flatten().tolist())

        # Trace upstream FeatureSets and create filtered views
        upstream_refs = find_upstream_featuresets(node=graph_node)
        exp_ctx = ExperimentContext.get_active()

        views: dict[str, FeatureSetView] = {}
        uuid_list = list(all_uuids)
        for ref in upstream_refs:
            fs: FeatureSet = exp_ctx.get_node(
                node_id=ref.node_id,
                enforce_type="FeatureSet",
            )
            views[fs.label] = fs.take_sample_uuids(uuid_list)

        return views

    def source_view(
        self,
        node: str | GraphNode,
        *,
        role: str = "default",
        epoch: int | None = None,
        batch: int | None = None,
    ) -> FeatureSetView:
        """
        Get the single source FeatureSetView for the given node.

        Description:
            Convenience method for the common case where a node has exactly
            one upstream FeatureSet. Raises `ValueError` if multiple
            upstream FeatureSets exist.

            Note that the returned views contain only unique sample UUIDs used
            in generating these phase results. They are not a 1-to-1 mapping
            of result sample to source sample. Use `tensors()` to get exact
            execution data.

        Args:
            node (str | GraphNode):
                The node to trace upstream from.
            role (str, optional):
                Restrict to samples from this role only. Defaults to "default".
            epoch (int | None, optional):
                Restrict to samples from this epoch only.
            batch (int | None, optional):
                Restrict to samples from this batch only.

        Returns:
            FeatureSetView:
                A view of the single upstream FeatureSet filtered to only
                the samples used during execution.

        Raises:
            ValueError:
                If the node has multiple upstream FeatureSets.

        """
        views = self.source_views(node=node, role=role, epoch=epoch, batch=batch)
        if len(views) != 1:
            msg = (
                f"Node has {len(views)} upstream FeatureSets: "
                f"{list(views.keys())}. Use source_views() instead."
            )
            raise ValueError(msg)
        return next(iter(views.values()))

    # ================================================
    # Callback & Metric Querying
    # ================================================
    def callbacks(self) -> CallbackDataSeries:
        """
        Returns a query interface for callback results.

        Data is keyed by `(kind, label, epoch, batch, edge)`.
        """
        # Check cache
        cache_key = ("callbacks",)
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        # Key callbacks by label and kind
        keyed_data: dict[tuple, CallbackResult] = {}
        for cb_res in self._callbacks:
            key = (
                cb_res.kind,
                cb_res.callback_label,
                cb_res.epoch_idx,
                cb_res.batch_idx,
                cb_res.edge,
            )
            if key in keyed_data:
                msg = f"Key already exists for CallbackResult: {key}"
                raise KeyError(msg)
            keyed_data[key] = cb_res

        return self._cache_set(
            cache_key,
            CallbackDataSeries(
                axes=("kind", "label", "epoch", "batch", "edge"),
                _data=keyed_data,
            ),
        )

    def metrics(self) -> MetricDataSeries:
        """
        Get all metric entries.

        Returns:
            MetricDataSeries:
                Entries keyed by `(name, epoch, batch)`.
                Epoch-level entries have `batch=None`.

        """
        return self._metrics.entries()

    def metric_names(self) -> list[str]:
        """All unique metric names recorded in these results."""
        return self.metrics().axis_values(axis="name")
