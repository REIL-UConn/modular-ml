from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from modularml.context.experiment_context import ExperimentContext
from modularml.core.topology.graph_node import GraphNode
from modularml.utils.data.data_format import DataFormat, normalize_format
from modularml.utils.data.multi_keyed_data import AxisSeries
from modularml.utils.data.scaling import unscale_sample_data

if TYPE_CHECKING:
    from collections.abc import Hashable

    from modularml.context.execution_context import ExecutionContext
    from modularml.core.data.batch import Batch
    from modularml.core.data.sample_data import SampleData
    from modularml.core.experiment.callback import CallbackResult
    from modularml.core.references.execution_reference import TensorLike
    from modularml.core.training.loss_record import LossCollection


@dataclass
class PhaseResults:
    phase_label: str

    # Phase contexts ordered by execution time
    _execution: list[ExecutionContext] = field(default_factory=list)

    # Results produced by callbacks; keyed by callback label
    _callbacks: dict[str, list[CallbackResult]] = field(default_factory=dict)

    # Memoized AxisSeries objects (invalidated on mutation)
    _series_cache: dict[tuple[Hashable, ...], Any] = field(
        default_factory=dict,
        init=False,
    )

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
