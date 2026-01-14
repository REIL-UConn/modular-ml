from __future__ import annotations

from typing import TYPE_CHECKING

from modularml.core.experiment.phase import ExperimentPhase, InputBinding

if TYPE_CHECKING:
    from modularml.core.topology.graph_node import GraphNode


class EvalPhase(ExperimentPhase):
    def __init__(
        self,
        label: str,
        input_sources: list[InputBinding],
        active_nodes: list[GraphNode] | None = None,
    ):
        """
        Initiallizes a new evaluation phase for the experiment.

        Notes:
            All `input_sources` must originate from the same upstream FeatureSet.
            If multiple FeatureSets need to be evaluated, they must be done so in
            separate EvalPhases.

        Args:
            label (str):
                A label to assign to this phase of the experiment. Used for logging.

            input_sources (list[InputBinding]):
                Input bindings for each head node in ModelGraph. All bindings must
                resolve to the same FeatureSet.

            active_nodes (list[GraphNode] | None, optional):
                A list of active GraphNodes in this phase of the experiment. Nodes can
                be listed via their ID, label, or with the actual node instance. If
                None, all nodes comprising the ModelGraph are used. Defaults to None.

        """
        super().__init__(
            label=label,
            input_sources=input_sources,
            active_nodes=active_nodes,
        )
        self._validate_single_featureset()

    def _validate_single_featureset(self):
        """Ensure all input sources originate from same FeatureSet."""
        if len(self.input_sources) <= 1:
            return

        fs_node_ids = {binding.upstream_ref.node_id for binding in self.input_sources}
        if len(fs_node_ids) > 1:
            fs_lbls = {
                binding.upstream_ref.node_label for binding in self.input_sources
            }
            msg = (
                "All `input_sources` of an EvalPhase must resolve to a single upstream "
                f"FeatureSet. Detected multiple: {fs_lbls}."
            )
            raise ValueError(msg)

    def __repr__(self):
        return f"EvalPhase(label='{self.label}')"
