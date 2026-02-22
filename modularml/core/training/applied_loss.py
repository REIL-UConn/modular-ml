"""Utilities for applying configured losses to model outputs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np

from modularml.core.data.sample_schema import DOMAIN_TARGETS
from modularml.core.data.schema_constants import DOMAIN_OUTPUTS
from modularml.core.experiment.experiment_context import ExperimentContext
from modularml.core.references.experiment_reference import ResolutionError
from modularml.core.references.featureset_reference import FeatureSetColumnReference
from modularml.core.references.model_io_reference import ModelOutputReference
from modularml.core.topology.model_node import ModelNode
from modularml.utils.data.conversion import align_ranks, convert_to_format, to_numpy
from modularml.utils.data.data_format import get_data_format_for_backend
from modularml.utils.data.shape_utils import ensure_tuple_shape
from modularml.utils.environment.optional_imports import ensure_tensorflow, ensure_torch
from modularml.utils.errors.exceptions import BackendMismatchError
from modularml.utils.nn.backend import Backend
from modularml.utils.representation.summary import Summarizable

if TYPE_CHECKING:
    from modularml.core.data.batch import Batch
    from modularml.core.data.batch_view import BatchView
    from modularml.core.data.execution_context import ExecutionContext
    from modularml.core.references.execution_reference import TensorLike
    from modularml.core.references.reference_like import ReferenceLike
    from modularml.core.topology.graph_node import GraphNode
    from modularml.core.training.loss import Loss


class AppliedLoss(Summarizable):
    """
    Bind a :class:`Loss` to a :class:`ModelNode` with resolved inputs.

    Attributes:
        loss (Loss):
            Wrapped :class:`Loss` instance evaluated for each execution step.
        weight (float):
            Scalar multiplier applied to computed loss values before aggregation.
        label (str | None):
            Friendly label used when logging summaries for this applied loss.
        node_id (str):
            Identifier of the :class:`ModelNode` targeted by this loss.
        inputs (dict[str, ReferenceLike]):
            Mapping of loss argument names to runtime references.

    """

    def __init__(
        self,
        loss: Loss,
        on: str | ModelNode,
        inputs: list[ReferenceLike] | dict[str, ReferenceLike],
        *,
        weight: float = 1.0,
        label: str | None = None,
    ):
        """
        Define a :class:`Loss` applied on a specified :class:`ModelNode`.

        Args:
            loss (Loss):
                Loss instance to apply for each execution step.
            on (str | ModelNode):
                Node label/ID or :class:`ModelNode` object indicating where
                the loss attaches.
            inputs (list[ReferenceLike] | dict[str, ReferenceLike]):
                Positional or keyword references resolved as loss arguments.
            weight (float):
                Scalar multiplier applied to the computed loss value.
            label (str | None):
                Optional label used when logging or summarizing this applied loss.

        Raises:
            TypeError:
                If `on` is not a :class:`ModelNode` or string, or `inputs` has
                an unsupported type.

        """
        self.loss = loss
        self.weight = float(weight)
        self.label = label or loss.name

        # Resolve ModelNode
        if isinstance(on, ModelNode):
            self.node_id = on.node_id
        elif isinstance(on, str):
            exp_ctx = ExperimentContext.get_active()
            self.node_id = exp_ctx.get_node(
                val=on,
                enforce_type="ModelNode",
            ).node_id
        else:
            msg = f"`on` must be a ModelNode or string. Received: {type(on)}."
            raise TypeError(msg)

        # Normalize inputs to dict[str, str]
        if isinstance(inputs, list):
            self.inputs = {str(i): v for i, v in enumerate(inputs)}
        elif isinstance(inputs, Mapping):
            self.inputs = dict(inputs)
        else:
            raise TypeError("`inputs` must be list[str] or dict[str, str]")

    def __eq__(self, other: AppliedLoss):
        if not isinstance(other, AppliedLoss):
            msg = f"Cannot compare equality between AppliedLoss and {type(other)}."
            raise TypeError(msg)

        return self.get_config() == other.get_config()

    __hash__ = None

    # ================================================
    # Properties
    # ================================================
    @property
    def backend(self) -> Backend:
        """
        Backend used by the underlying :class:`Loss`.

        Returns:
            Backend: Backend declared on the wrapped :class:`Loss`.

        """
        return self.loss.backend

    # ================================================
    # Input resolution
    # ================================================
    def _resolve_input(
        self,
        spec: str,
        ctx: ExecutionContext,
    ) -> tuple[TensorLike, TensorLike, TensorLike]:
        """
        Resolve a loss input reference into tensor data, weights, and masks.

        Args:
            spec (str): Reference string such as `outputs.default` or a FeatureSet column path.
            ctx (ExecutionContext): Execution context containing upstream batches and outputs.

        Returns:
            tuple[TensorLike, TensorLike, TensorLike]: Tuple of tensor-like data, weights, and masks.

        Raises:
            BackendMismatchError: If the :class:`Loss` backend does not match the :class:`ModelNode`.
            ResolutionError: If the reference cannot be resolved to a model output or FeatureSet column.

        """
        exp_ctx = ExperimentContext.get_active()

        # Validate node & backend
        node: ModelNode = exp_ctx.get_node(
            node_id=self.node_id,
            enforce_type="ModelNode",
        )
        if self.loss.backend != node.backend:
            msg = (
                f"ModelNode ('{node.label}') and Loss ('{self.loss.name}') "
                f"backends do not match. {node.backend} != {self.loss.backend}."
            )
            raise BackendMismatchError(message=msg)

        # Remove any references to `on_node.label`
        spec = spec.replace(f"{node.label}.", "")

        # Convert to OutputRef input string starts with "outputs" or "targets"
        if any(spec.startswith(x) for x in [DOMAIN_OUTPUTS, DOMAIN_TARGETS]):
            return self._resolve_model_output(spec=spec, node=node, ctx=ctx)

        # Otherwise, convert to a FeatureSetColumnReference
        return self._resolve_featureset_column(spec=spec, node=node, ctx=ctx)

    def _resolve_model_output(
        self,
        spec: str,
        node: ModelNode,
        ctx: ExecutionContext,
    ) -> tuple[TensorLike, TensorLike, TensorLike]:
        """
        Resolve a model output reference into tensor, weight, and mask tuples.

        Args:
            spec (str):
                Domain/role specification referencing :attr:`ExecutionContext.outputs`.
            node (ModelNode):
                Node supplying the outputs referenced by this loss.
            ctx (ExecutionContext):
                Execution context that holds model outputs for the current step.

        Returns:
            tuple[TensorLike, TensorLike, TensorLike]: Tuple containing tensor data, weights, and masks.

        Raises:
            ResolutionError: If the reference is malformed or the role cannot be inferred uniquely.

        """
        # Extract role key
        domain, role = spec, None
        if "." in spec:
            parts = spec.split(".")
            if len(parts) == 1:
                domain, role = parts[0], None
            elif len(parts) == 2:
                domain, role = parts
            else:
                msg = (
                    f"AppliedLoss input '{spec}' could not resolved. Too many "
                    f"components: {parts}."
                )
                raise ResolutionError(msg)

        # Get model outputs
        output_batch = ctx.outputs[node.node_id]
        if role is None:
            if len(output_batch.available_roles) != 1:
                msg = (
                    f"Applied loss spec '{spec}' must specify a `role` when multiple "
                    "roles exist in the output data. Available roles: "
                    f"{output_batch.available_roles}."
                )
                raise ResolutionError(msg)
            role = output_batch.available_roles[0]

        # Resolve reference to tensor like data
        ref = ModelOutputReference(
            node_label=node.label,
            node_id=node.node_id,
            role=role,
            domain=domain,
        )
        tensor_like = ref.resolve(ctx=ctx)

        # Grab weights and mask from batch
        weights = output_batch.role_weights[role]
        mask = output_batch.role_masks[role]

        return tensor_like, weights, mask

    def _resolve_featureset_column(
        self,
        spec: str,
        node: ModelNode,
        ctx: ExecutionContext,
    ) -> tuple[TensorLike, TensorLike, TensorLike]:
        """
        Resolve a FeatureSet column reference into tensor, weight, and mask tuples.

        Args:
            spec (str):
                Column path referencing upstream FeatureSet data.
            node (ModelNode):
                Node consuming the upstream batch derived from the FeatureSet.
            ctx (ExecutionContext):
                Execution context providing FeatureSet :class:`BatchView` instances.

        Returns:
            tuple[TensorLike, TensorLike, TensorLike]:
                Tuple containing tensor data, weights, and masks.

        Raises:
            ResolutionError:
                If the upstream FeatureSet cannot be inferred or lacks a usable role.

        """
        bv: BatchView = self._get_upstream_view(node=node, ctx=ctx)
        ref = FeatureSetColumnReference.from_string(
            val=spec,
            experiment=ExperimentContext.get_active(),
            known_attrs={
                "node_id": bv.source.node_id,
                "node_label": bv.source.label,
            },
        )
        # Materialize batch
        b: Batch = bv.materialize_batch(
            fmt=get_data_format_for_backend(backend=self.backend),
            columns=[f"{ref.domain}.{ref.key}.{ref.rep}"],
        )
        # Infer role
        role = None
        if len(b.available_roles) == 1:
            role = b.available_roles[0]
        elif "default" in b.available_roles:
            role = "default"
        elif "anchor" in b.available_roles:
            role = "anchor"
        else:
            msg = (
                f"AppliedLoss input '{spec}' could not resolved. Role must be "
                f"specified when multiple exists. Available: {b.available_roles}."
            )
            raise ResolutionError(msg)

        # Get domain data (tensor like)
        tensor_like = b.role_data.get_data(role=role, domain=ref.domain)

        # Create dummy weights (no masking available)
        weights = np.ones(shape=ensure_tuple_shape(tensor_like.shape))
        mask = np.ones(shape=len(weights), dtype=np.int8)

        return tensor_like, weights, mask

    def _get_upstream_view(
        self,
        node: ModelNode,
        ctx: ExecutionContext,
    ) -> BatchView:
        """
        Determine the :class:`BatchView` feeding the head node on this branch.

        Args:
            node (ModelNode):
                Node whose upstream FeatureSet should be located.
            ctx (ExecutionContext):
                Execution context storing the head inputs per node.

        Returns:
            BatchView:
                Upstream view that supplies data to the requested node.

        Raises:
            ResolutionError: If zero or multiple upstream FeatureSets feed the node.

        """
        exp_ctx = ExperimentContext.get_active()
        # All head node IDs in this ExecutionContext
        head_node_ids = [x[0] for x in ctx.inputs]

        # Get all upstream head nodes of `node`
        upstream_views: list[BatchView] = []
        visited: set[str] = set()

        def _get_input_view(n: GraphNode):
            # Record visited to protect against incidental loops
            if n.node_id in visited:
                return
            visited.add(n.node_id)

            # If this is a head node, add all inputs and return
            if n.node_id in head_node_ids:
                bvs: list[BatchView] = [
                    bv for inp_key, bv in ctx.inputs.items() if inp_key[0] == n.node_id
                ]
                upstream_views.extend(bvs)
                return

            # Otherwise, recurse on upstream node
            for ref in n._upstream_refs:
                up_n = ref.resolve(ctx=exp_ctx)
                _get_input_view(n=up_n)

        _get_input_view(n=node)

        if len(upstream_views) != 1:
            msg = (
                "FeatureSet-column-based loss inputs require that the applied-to-node "
                f"has exactly one upstream FeatureSet. Detected: {len(upstream_views)}."
            )
            raise ResolutionError(msg)

        return next(iter(upstream_views))

    # ================================================
    # Computation
    # ================================================
    def _apply_weights(self, raw_loss: Any, weights: Any) -> Any:
        """
        Apply sample weights to the raw backend loss output.

        Args:
            raw_loss (Any):
                Backend-specific tensor or array returned by :class:`Loss`.
            weights (Any):
                Per-sample weights aligned with `raw_loss`.

        Returns:
            Any: Weighted scalar compatible with the configured backend.

        """
        # Apply sample weighting -> convert mean_weights to correct backend tensor
        if self.backend == Backend.TORCH:
            torch = ensure_torch()
            # Ensure loss has shape (batch_size, )
            raw_loss = raw_loss.view(-1)
            w = torch.as_tensor(weights, device=raw_loss.device)
            return torch.sum(raw_loss * w) * self.weight / len(raw_loss)

        if self.backend == Backend.TENSORFLOW:
            tf = ensure_tensorflow()
            # Ensure loss has shape (batch_size, )
            raw_loss = tf.reshape(raw_loss, [-1])
            w = tf.convert_to_tensor(weights, dtype=raw_loss.dtype)
            return tf.reduce_sum(raw_loss * w) * self.weight / len(raw_loss)

        # Assume NumPy
        raw_loss = np.reshape(raw_loss, (-1,))
        w = np.reshape(weights, (-1,))
        return np.sum(raw_loss * w) * self.weight / len(raw_loss)

    def compute(self, ctx: ExecutionContext) -> Any:
        """
        Compute the weighted loss for a single execution step.

        Args:
            ctx (ExecutionContext):
                Execution context supplying model outputs and upstream batches.

        Returns:
            Any: Backend-specific scalar/tensor representing the weighted loss value.

        Raises:
            BackendMismatchError:
                If the :class:`Loss` backend differs from the :class:`ModelNode`
                backend.
            ResolutionError:
                If any configured reference cannot be resolved from the execution
                context.

        """
        # Map self.inputs.keys() to batch tensor data
        kw_data: dict[str, Any] = {}
        kw_weights: list[np.ndarray] = []
        kw_masks: list[np.ndarray] = []

        # Collect required input(s) for each loss argument
        for arg, spec in self.inputs.items():
            data, weights, mask = self._resolve_input(spec=spec, ctx=ctx)
            kw_data[arg] = convert_to_format(
                data=data,
                fmt=get_data_format_for_backend(backend=self.backend),
            )
            kw_weights.append(to_numpy(weights))
            kw_masks.append(to_numpy(mask).astype(bool))

        # Ensure all kwargs have matching shapes (aligns singletons)
        ref_key = next(iter(kw_data.keys()))
        for k in [x for x in kw_data if x != ref_key]:
            kw_data[ref_key], kw_data[k] = align_ranks(
                kw_data[ref_key],
                kw_data[k],
                backend=self.backend,
            )

        # Combine masks (logical AND across inputs)
        combined_mask = np.logical_and.reduce(kw_masks)  # shape: (n_samples, )

        # Combine weights (mean across inputs, then apply mask)
        mean_weights = np.mean(
            np.stack(kw_weights, axis=0),
            axis=0,
        ).reshape(-1)  # shape: (n_samples, )
        mean_weights = mean_weights * combined_mask.astype(mean_weights.dtype)

        # Call loss function (convert to positional args if needed)
        if all(k.isdigit() for k in kw_data):
            args = [kw_data[str(i)] for i in range(len(kw_data))]
            raw = self.loss(*args)
        else:
            raw = self.loss(**kw_data)

        # Apply weighting
        return self._apply_weights(raw, mean_weights)

    # ================================================
    # Representation
    # ================================================
    def _summary_rows(self) -> list[tuple]:
        """
        Return summary table rows representing this applied loss configuration.

        Returns:
            list[tuple]: Sequence of key/value tuples rendered in summaries.

        """
        rows: list[tuple] = [
            ("label", str(self.label)),
            (
                "loss",
                self.loss._summary_rows()
                if hasattr(self.loss, "_summary_rows")
                else f"{self.loss!r}",
            ),
            ("inputs", str(self.inputs)),
            ("weight", str(self.weight)),
        ]
        return rows

    def __repr__(self):
        return (
            f"AppliedLoss(label={self.label!r}, loss={self.loss.name!r}, "
            f"inputs={self.inputs!r}, weight={self.weight})"
        )

    # ================================================
    # Configurable
    # ================================================
    def get_config(self) -> dict[str, Any]:
        """
        Return configuration required to reconstruct this applied loss.

        Returns:
            dict[str, Any]:
                Serialized dictionary capturing loss, node, inputs, weight, and label.

        """
        return {
            "loss": self.loss,  # not JSON safe
            "on": self.node_id,
            "inputs": self.inputs,  # not JSON safe
            "weight": self.weight,
            "label": self.label,
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> AppliedLoss:
        """
        Construct an :class:`AppliedLoss` from configuration.

        Args:
            config (dict[str, Any]):
                Dictionary produced by :meth:`get_config`.

        Returns:
            AppliedLoss: Rehydrated applied loss instance.

        """
        return cls(**config)
