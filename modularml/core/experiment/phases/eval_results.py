from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from modularml.core.data.batch import Batch
from modularml.core.experiment.phases.phase_result import PhaseResults

if TYPE_CHECKING:
    from modularml.core.references.execution_reference import TensorLike
    from modularml.core.topology.graph_node import GraphNode
    from modularml.core.training.loss_record import LossCollection
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
        ```

    """

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
    # Tensor Stacking
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
    # Loss Aggregation
    # ================================================
    def aggregated_losses(
        self,
        node: str | GraphNode,
        *,
        reducer: Literal["sum", "mean"] = "mean",
        by_label: bool = False,
    ) -> LossCollection | dict[str, LossCollection]:
        """
        Retrieve losses for a node, aggregated across all batches.

        Description:
            Collects LossCollection objects from all evaluation batches and
            reduces them using the specified strategy:

            - "sum": Merges all loss records (LossCollection.merge)
            - "mean": Computes elementwise average (LossCollection.mean)

            Optionally groups results by loss label.

        Args:
            node (str | GraphNode):
                The node to retrieve losses for. Can be the node instance,
                its ID, or its label.
            reducer (Literal["sum", "mean"], optional):
                Reduction strategy for aggregating losses across batches.
                - "sum": Concatenates all individual loss records
                - "mean": Computes the mean loss value per label
                Defaults to "mean".
            by_label (bool, optional):
                If True, returns a dict mapping loss labels to their
                aggregated LossCollection. If False, returns a single
                aggregated LossCollection. Defaults to False.

        Returns:
            LossCollection | dict[str, LossCollection]:
                Aggregated losses. If by_label=True, returns a dict keyed
                by loss label.

        Examples:
            ```python
            # Get total loss across all batches (sum)
            lc = eval_results.aggregated_losses(node="output_node")
            print(f"Total loss: {lc.total}")

            # Get mean loss across all batches
            lc = eval_results.aggregated_losses(node="output_node", reducer="mean")
            print(f"Mean loss: {lc.total}")

            # Get losses grouped by label
            lcs_by_label = eval_results.aggregated_losses(
                node="output_node",
                by_label=True,
            )
            for label, lc in lcs_by_label.items():
                print(f"{label}: {lc.total}")
            ```

        """
        loss_series = self.losses(node=node)

        # Collapse batch axis first
        batch_collapsed = loss_series.collapse(axis="batch", reducer=reducer)

        if by_label:
            # Return dict keyed by label
            # batch_collapsed is keyed by (epoch, label) -> (0, label)
            result: dict[str, LossCollection] = {}
            for label in batch_collapsed.axis_values("label"):
                result[label] = batch_collapsed.where(label=label).one()
            return result

        # Collapse label axis to get single LossCollection
        label_collapsed = batch_collapsed.collapse(axis="label", reducer=reducer)
        return label_collapsed.one()
