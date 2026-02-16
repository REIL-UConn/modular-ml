from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from modularml.core.data.featureset import FeatureSet
from modularml.core.experiment.experiment_context import ExperimentContext
from modularml.core.topology.graph_node import GraphNode
from modularml.utils.data.data_format import DataFormat, normalize_format
from modularml.utils.data.multi_keyed_data import AxisSeries
from modularml.utils.data.scaling import unscale_sample_data
from modularml.utils.topology.graph_search_utils import find_upstream_featuresets

if TYPE_CHECKING:
    from collections.abc import Hashable

    from modularml.core.data.batch import Batch
    from modularml.core.data.execution_context import ExecutionContext
    from modularml.core.data.featureset_view import FeatureSetView
    from modularml.core.data.sample_data import SampleData
    from modularml.core.experiment.callback import CallbackResult
    from modularml.core.references.execution_reference import TensorLike
    from modularml.core.training.loss_record import LossCollection


@dataclass
class PhaseResults:
    label: str

    # Phase contexts ordered by execution time
    _execution: list[ExecutionContext] = field(default_factory=list, repr=False)

    # Results produced by callbacks; keyed by callback label
    _callbacks: dict[str, list[CallbackResult]] = field(
        default_factory=dict,
        repr=False,
    )

    # Memoized AxisSeries objects (invalidated on mutation)
    _series_cache: dict[tuple[Hashable, ...], Any] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    # ================================================
    # Representation
    # ================================================
    def __repr__(self):
        n_exec = len(self._execution)
        cb_labels = list(self._callbacks.keys())
        return (
            f"PhaseResults(label='{self.label}', "
            f"executions={n_exec}, "
            f"callbacks={cb_labels})"
        )

    # ================================================
    # Discoverability
    # ================================================
    @property
    def node_ids(self) -> list[str]:
        """Unique node IDs that have recorded outputs in these results."""
        seen: dict[str, None] = {}
        for ctx in self._execution:
            for nid in ctx.outputs:
                if nid not in seen:
                    seen[nid] = None
        return list(seen)

    @property
    def loss_node_ids(self) -> list[str]:
        """Unique node IDs that have recorded losses in these results."""
        seen: dict[str, None] = {}
        for ctx in self._execution:
            for nid in ctx.losses:
                if nid not in seen:
                    seen[nid] = None
        return list(seen)

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
        return self._series_cache.get(key)

    def _cache_set(self, key: tuple[Hashable, ...], value):
        self._series_cache[key] = value
        return value

    # ================================================
    # Runtime Modifiers
    # ================================================
    def add_execution_context(self, ctx: ExecutionContext):
        """Record a new execution context."""
        self._execution.append(ctx)
        self._series_cache.clear()

    def add_callback_result(self, cb_res: CallbackResult):
        """Record a new callback result."""
        if cb_res.callback_label not in self._callbacks:
            self._callbacks[cb_res.callback_label] = []
        self._callbacks[cb_res.callback_label].append(cb_res)
        self._series_cache.clear()

    # ================================================
    # Execution Data & Loss Querying
    # ================================================
    def execution_contexts(self) -> AxisSeries[ExecutionContext]:
        """Returns a query interface for execution contexts."""
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
            AxisSeries(axes=("epoch", "batch"), _data=keyed_data),
        )

    def batches(self, node: str | GraphNode) -> AxisSeries[Batch]:
        """
        Returns a query interface for batches on a specific node.

        Args:
            node (str | GraphNode):
                The node to filter batches to. Can be the node instance, its ID, or
                its label.

        Returns:
            AxisSeries[Batch]:
                A keyed iterable over all batches executed on `node`.
                Keyed by epoch and batch index.

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
            AxisSeries(axes=("epoch", "batch"), _data=keyed_data),
        )

    def tensors(
        self,
        node: str | GraphNode,
        domain: Literal["outputs", "targets", "tags", "sample_uuids"],
        *,
        role: str = "default",
        fmt: DataFormat | None = None,
        unscale: bool = False,
    ) -> AxisSeries[TensorLike]:
        """
        Returns a query interface for tensors related to a specific node.

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
                Defaults to `'default'`.
            fmt (DataFormat, optional):
                The format to cast the returned tensors to. If None, the as-produced
                format is used. Defaults to None.
            unscale (bool, optional):
                Whether to inverse any applied scalers to these tensors.
                Note that this is only possible when `node` refers to a tail node,
                and `domain` is one of `["outputs", "targets"]`.

        Returns:
            AxisSeries[TensorLike]:
                A keyed iterable over all tensors related to `node`.
                Keyed by epoch and batch index.

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
        for ax_key, b in batch_series.items():
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
            AxisSeries(axes=("epoch", "batch"), _data=keyed_tensors),
        )

    def losses(self, node: str | GraphNode) -> AxisSeries[LossCollection]:
        """
        Returns a query interface for losses applied to a specific node.

        Args:
            node (str | GraphNode):
                The node to filter losses to. Can be the node instance,
                its ID, or its label.

        Returns:
            AxisSeries[LossCollection]:
                A keyed iterable over all losses applied to `node`.
                Keyed by epoch, batch, and loss label.

        """
        # Resolve node
        graph_node = self._resolve_node(node=node)

        # Check cache
        cache_key = ("losses", graph_node.node_id)
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        # Key losses by execution scope
        keyed_lcs: dict[tuple[int, int, str], LossCollection] = {}
        for ctx in self._execution:
            lbl_lcs = ctx.losses[graph_node.node_id].by_label()
            for k, lc in lbl_lcs.items():
                keyed_lcs[(ctx.epoch_idx, ctx.batch_idx, k)] = lc

        return self._cache_set(
            cache_key,
            AxisSeries(axes=("epoch", "batch", "label"), _data=keyed_lcs),
        )

    # ================================================
    # Source Data Access
    # ================================================
    def source_views(
        self,
        node: str | GraphNode,
        *,
        role: str = "default",
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
                Restrict to samples from this role only. Defaults to "default".
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
    # Callback Querying
    # ================================================
    @property
    def callback_labels(self) -> list[str]:
        """Returns the unique labels of recorded callbacks."""
        return list(self._callbacks.keys())

    @property
    def callback_kinds(self) -> list[str]:
        """Returns the unique kinds of recorded callbacks."""
        return list({k[1] for k in self.callbacks()})

    def callbacks(self) -> AxisSeries[list[CallbackResult]]:
        """Returns a query interface for callback results."""
        # Check cache
        cache_key = ("callbacks",)
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        # Key callbacks by label and kind
        keyed_data: dict[tuple[str, str], list[CallbackResult]] = {
            (lbl, v[0].kind): v for lbl, v in self._callbacks.items()
        }
        return self._cache_set(
            cache_key,
            AxisSeries(axes=("label", "kind"), _data=keyed_data),
        )
