from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modularml.core.data.batch import Batch
    from modularml.core.data.batch_view import BatchView
    from modularml.core.references.featureset_reference import FeatureSetReference
    from modularml.core.training.loss_record import LossCollection


@dataclass
class ExecutionContext:
    """
    Execution-time container for a single batch.

    Description:
        Stores all inputs, outputs, losses, and metrics produced during a
        single forward/backward pass through the ModelGraph.
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

    # Losses computed in this batch (keyed by node ID)
    losses: dict[str, LossCollection] = field(default_factory=dict)

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
        if node_id not in self.inputs:
            self.inputs[node_id] = {}
        self.inputs[(node_id, upstream)] = batch_view

    def set_output(self, *, node_id: str, batch: Batch):
        self.outputs[node_id] = batch

    def set_losses(self, *, node_id: str, loss: LossCollection):
        self.losses[node_id] = loss
