from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from modularml.core.data.batch import Batch
from modularml.core.experiment.results.phase_results import PhaseResults

if TYPE_CHECKING:
    from modularml.core.data.featureset_view import FeatureSetView
    from modularml.core.references.execution_reference import TensorLike
    from modularml.core.topology.graph_node import GraphNode
    from modularml.utils.data.data_format import DataFormat


@dataclass
class EvalResults(PhaseResults):
    """
    Results container for a single forward-pass evaluation phase.

    Description:
        EvalResults wraps the outputs of an EvalPhase, which executes a single
        epoch (epoch=0) over multiple batches. This class provides convenience
        methods for:

        - Automatic tensor stacking across batches
        - Loss aggregation (sum/mean) over batches

        All methods leverage the base PhaseResults query interface and use
        AxisSeries.collapse() for batch aggregation.

    Examples:
        ```python
        # Run evaluation
        eval_results = experiment.run_evaluation(phase=eval_phase)

        # Get stacked outputs for a node (all batches concatenated)
        outputs = eval_results.stacked_tensors(node="output_node", domain="outputs")

        # Get total loss across all batches
        total_loss = eval_results.aggregated_losses(node="output_node")

        # Get all source data utilized in evaluation
        source_view = eval_results.source_view(node="output_node")
        ```

    """

    # ================================================
    # Representation
    # ================================================
    def __repr__(self):
        n_batches = self.n_batches if self._execution else 0
        return f"EvalResults(label='{self.label}', batches={n_batches})"

    # ================================================
    # Properties
    # ================================================
    @property
    def batch_indices(self) -> list[int]:
        """
        Sorted list of recorded batch indices.

        Returns:
            list[int]: Batch indices in ascending order.

        """
        batch_vals = self.execution_contexts().axis_values("batch")
        return sorted(int(e) for e in batch_vals)

    @property
    def n_batches(self) -> int:
        """The number of batches executed during evaluation."""
        return len(self.batch_indices)

    # ================================================
    # Execution Data & Loss Querying
    # ================================================
    def stacked_tensors(
        self,
        node: str | GraphNode,
        domain: Literal["outputs", "targets", "tags", "sample_uuids"],
        *,
        role: str = "default",
        fmt: DataFormat | None = None,
        unscale: bool = False,
    ) -> TensorLike:
        """
        Retrieve tensors for a node, concatenated across all batches.

        Description:
            Collects tensors from the specified domain across all evaluation
            batches and concatenates them along the batch dimension using
            backend-aware concatenation (torch.cat, np.concatenate, or tf.concat).

            This is the primary method for retrieving complete evaluation outputs
            or targets in a single tensor.

        Args:
            node (str | GraphNode):
                The node to retrieve tensors for. Can be the node instance,
                its ID, or its label.
            domain (Literal["outputs", "targets", "tags", "sample_uuids"]):
                The domain of data to return:
                - outputs: the tensors produced by the node forward pass
                - targets: the expected output tensors (only for tail nodes)
                - tags: any tracked tags during the node's forward pass
                - sample_uuids: the sample identifiers
            role (str, optional):
                If multi-role data, specifies which role to return.
                Defaults to "default".
            fmt (DataFormat | None, optional):
                Format to cast returned tensors to. If None, uses as-produced
                format. Defaults to None.
            unscale (bool, optional):
                Whether to inverse any applied scalers. Only valid for tail
                nodes with domain in ["outputs", "targets"]. Defaults to False.

        Returns:
            TensorLike:
                A single tensor containing concatenated data from all batches.

        Examples:
            ```python
            # Get all predictions stacked
            predictions = eval_results.stacked_tensors(
                node="output_node",
                domain="outputs",
            )

            # Get targets, unscaled, as numpy
            targets = eval_results.stacked_tensors(
                node="output_node",
                domain="targets",
                fmt="np",
                unscale=True,
            )
            ```

        """
        tensor_series = self.tensors(
            node=node,
            domain=domain,
            role=role,
            fmt=fmt,
            unscale=unscale,
        )
        # Collapse batch axis, then get single value (only epoch=0)
        collapsed = tensor_series.collapse(axis="batch", reducer="concat")
        return collapsed.one()

    def stacked_batches(
        self,
        node: str | GraphNode,
        *,
        fmt: DataFormat | None = None,
    ) -> Batch:
        """
        Retrieve all batches for a node, concatenated into a single Batch.

        Description:
            Collects Batch objects from all evaluation batches and concatenates
            them using Batch.concat(). This provides access to all data domains
            (outputs, targets, tags, sample_uuids) plus role weights and masks
            in a single container.

        Args:
            node (str | GraphNode):
                The node to retrieve batches for.
            fmt (DataFormat | None, optional):
                Format to cast tensor data to. Defaults to None.

        Returns:
            Batch:
                A single Batch containing concatenated data from all batches.

        Examples:
            ```python
            batch = eval_results.stacked_batches(node="output_node")
            print(f"Total samples: {batch.batch_size}")
            print(f"Outputs shape: {batch.outputs.shape}")
            ```

        """
        batch_series = self.batches(node=node)
        batches = list(batch_series.values())

        if fmt is not None:
            batches = [b.to_format(fmt) for b in batches]

        return Batch.concat(*batches, fmt=fmt)

    # ================================================
    # Source Data Access
    # ================================================
    def source_views(
        self,
        node: str | GraphNode,
        *,
        role: str = "default",
        batch: int | None = None,
    ) -> dict[str, FeatureSetView]:
        """
        Get the source FeatureSetViews that contributed data to the given node.

        Description:
            Traces the node back to its upstream FeatureSets, collects all
            unique sample UUIDs from execution results, and returns a view
            of each upstream FeatureSet filtered to only the samples used.

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
            batch (int | None, optional):
                Restrict to samples from this batch only.

        Returns:
            dict[str, FeatureSetView]:
                A mapping of FeatureSet label to FeatureSetView containing
                only the samples used during execution.

        """
        return super().source_views(
            node=node,
            role=role,
            epoch=None,
            batch=batch,
        )

    def source_view(
        self,
        node: str | GraphNode,
        *,
        role: str = "default",
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
        views = self.source_views(node=node, role=role, batch=batch)
        if len(views) != 1:
            msg = (
                f"Node has {len(views)} upstream FeatureSets: "
                f"{list(views.keys())}. Use source_views() instead."
            )
            raise ValueError(msg)
        return next(iter(views.values()))
