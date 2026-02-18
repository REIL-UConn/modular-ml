"""Execution context for a single batch pass through the model graph."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from modularml.core.training.loss_record import LossCollection

if TYPE_CHECKING:
    from modularml.core.data.batch import Batch
    from modularml.core.data.batch_view import BatchView
    from modularml.core.references.featureset_reference import FeatureSetReference


@dataclass
class ExecutionContext:
    """
    Execution-time container for a single batch.

    Description:
        Stores all inputs, outputs, losses, and metrics produced during a
        single forward/backward pass through the ModelGraph.

    Attributes:
        phase_label (str): Label identifying the current phase.
        epoch_idx (int): Current epoch index.
        batch_idx (int): Current batch index within the epoch.
        inputs (dict): Inputs to head nodes, keyed by (node_id, upstream_ref).
        outputs (dict[str, Batch]): Outputs of graph nodes, keyed by node ID.
        losses (LossCollection | None): Losses computed in this batch.

    """

    # Identity
    phase_label: str
    epoch_idx: int
    batch_idx: int

    # Inputs to head nodes (keyed by node ID + upstream ref)
    inputs: dict[tuple[str, FeatureSetReference], BatchView] = field(
        default_factory=dict,
    )

    # Outputs of GraphNodes (keyed by node ID)
    outputs: dict[str, Batch] = field(default_factory=dict)

    # Losses computed in this batch
    losses: LossCollection | None = None

    # ================================================
    # Attribute Updating
    # ================================================
    def set_input(
        self,
        *,
        node_id: str,
        upstream: FeatureSetReference,
        batch_view: BatchView,
    ):
        """
        Register an input :class:`BatchView` for a head node.

        Args:
            node_id (str): Node identifier.
            upstream (FeatureSetReference): Upstream reference for the node.
            batch_view (BatchView): The input :class:`BatchView`.

        """
        if node_id not in self.inputs:
            self.inputs[node_id] = {}
        self.inputs[(node_id, upstream)] = batch_view

    def set_output(self, *, node_id: str, batch: Batch):
        """
        Sets the tracked outputs for a given node ID.

        Args:
            node_id (str): Node ID to set.
            batch (Batch): Batch data to record.

        """
        if node_id in self.outputs:
            msg = f"Data already set for node: '{node_id}'."
            raise ValueError(msg)
        self.outputs[node_id] = batch

    def add_losses(self, lc: LossCollection):
        """
        Update the tracked losses with this collection.

        Args:
            lc (LossCollection): Loss records to merge.

        """
        if self.losses is None:
            self.losses = lc

        # Combine collections
        else:
            self.losses = LossCollection(
                records=[*lc.values(), *self.losses.values()],
            )
