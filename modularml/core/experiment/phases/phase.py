"""Core definitions for experiment phases and input bindings."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from modularml.core.data.schema_constants import STREAM_DEFAULT
from modularml.core.experiment.experiment_context import ExperimentContext
from modularml.core.references.featureset_reference import FeatureSetReference
from modularml.core.sampling.base_sampler import BaseSampler
from modularml.utils.data.formatting import ensure_list, find_duplicates
from modularml.utils.errors.error_handling import ErrorMode
from modularml.visualization.visualizer.visualizer import Visualizer

if TYPE_CHECKING:
    from collections.abc import Iterator

    from modularml.core.data.batch_view import BatchView
    from modularml.core.data.execution_context import ExecutionContext
    from modularml.core.data.featureset import FeatureSet
    from modularml.core.data.featureset_view import FeatureSetView
    from modularml.core.experiment.callbacks.callback import Callback
    from modularml.core.experiment.results.phase_results import PhaseResults
    from modularml.core.topology.graph_node import GraphNode
    from modularml.core.training.applied_loss import AppliedLoss


def _get_upstream_featureset_refs_for_node(node_id: str) -> list[FeatureSetReference]:
    # Resolve node
    exp_ctx = ExperimentContext.get_active()
    node: GraphNode = exp_ctx.get_node(node_id=node_id, enforce_type="GraphNode")

    # Get all upstream FeatureSetReference of this node
    ups_fs_refs: list[FeatureSetReference] = [
        ref
        for ref in node.get_upstream_refs(error_mode=ErrorMode.IGNORE)
        if isinstance(ref, FeatureSetReference)
    ]
    return ups_fs_refs


def _resolve_upstream_featureset_ref(
    node_id: str,
    val: str | FeatureSetReference | FeatureSetView | FeatureSet | None = None,
) -> FeatureSetReference:
    from modularml.core.data.featureset import FeatureSet
    from modularml.core.data.featureset_view import FeatureSetView

    # Resolve node
    exp_ctx = ExperimentContext.get_active()
    node: GraphNode = exp_ctx.get_node(node_id=node_id, enforce_type="GraphNode")

    # Get all upstream FeatureSetReference of this node
    ups_fs_refs: list[FeatureSetReference] = _get_upstream_featureset_refs_for_node(
        node_id=node_id,
    )
    if len(ups_fs_refs) == 0:
        msg = (
            f"There are no upstream FeatureSets of '{node.label}'. Cannot "
            "generate binding."
        )
        raise RuntimeError(msg)

    if val is None:
        if len(ups_fs_refs) > 1:
            msg = (
                f"GraphNode '{node.label}' has multiple upstream FeatureSets. "
                f"You must specify `upstream` explicitly."
            )
            raise ValueError(msg)
        return ups_fs_refs[0]

    if isinstance(val, str):
        for ref in ups_fs_refs:
            if ref.node_id == val:
                return ref
            if ref.node_label == val:
                return ref
        msg = (
            f"No upstream FeatureSet of node '{node.label}' found with ID or "
            f"label of '{val}'."
        )
        raise ValueError(msg)
    if isinstance(val, FeatureSetReference):
        if val not in ups_fs_refs:
            msg = f"No matching FeatureSetReference exists on node '{node.label}'."
            raise ValueError(msg)
        return val
    if isinstance(val, FeatureSetView):
        val = val.source
    if isinstance(val, FeatureSet):
        for ref in ups_fs_refs:
            if ref.node_id == val.node_id:
                return ref
            if ref.node_label == val.label:
                return ref
        msg = (
            f"No upstream FeatureSet of node '{node.label}' matches given "
            "the given FeatureSet/FeatureSetView."
        )
        raise ValueError(msg)

    msg = (
        "`upstream` must of type str, FeatureSet, or FeatureSetView. "
        f"Received: {type(val)}."
    )
    raise TypeError(msg)


def _normalize_node_value_to_id(value: str | GraphNode) -> str:
    """Converts a node ID, label, or instance to its node ID."""
    from modularml.core.topology.graph_node import GraphNode

    exp_ctx = ExperimentContext.get_active()

    if isinstance(value, GraphNode):
        return value.node_id
    if isinstance(value, str):
        if exp_ctx.has_node(node_id=value):
            return value
        if exp_ctx.has_node(label=value):
            gnode: GraphNode = exp_ctx.get_node(
                label=value,
                enforce_type="GraphNode",
            )
            return gnode.node_id
        msg = f"The given GraphNode value ('{value}') does not correspond to any node IDs or labels in the active ExperimentContext."
        raise ValueError(msg)

    msg = f"GraphNode values must be instances, node IDs, or node labels. Received: {type(value)}."
    raise TypeError(msg)


@dataclass(frozen=True)
class InputBinding:
    """
    A phase-specific binding of input data to a head GraphNode.

    Description:
        An InputBinding exists within a single Experiment phase. It defines
        an attachment of a sampler (or direct pass-through) to an existing
        graph edge between a FeatureSet and a head GraphNode, optionally
        restricted to a FeatureSet split.

    Attributes:
        node_id (str):
            ID of the head GraphNode on which we are defining input for.
        upstream_ref (FeatureSetReference):
            Which upstream reference of the head node this binding applies to.
        split (str, optional):
            If defined, only data from this split is used.
        sampler (BaseSampler, optional):
            A sampler to use in feeding the source data to the head node.
        stream (str, optional):
            If a sampler with multiple output streams is used, this defines the
            exact stream of data to feed to the head node.

    """

    node_id: str
    upstream_ref: FeatureSetReference
    split: str | None = None
    sampler: BaseSampler | None = None
    stream: str = STREAM_DEFAULT

    # ================================================
    # Constructors
    # ================================================
    @classmethod
    def for_training(
        cls,
        *,
        node: GraphNode | str,
        sampler: BaseSampler,
        upstream: FeatureSet | FeatureSetView | str | None,
        split: str | None = None,
        stream: str = STREAM_DEFAULT,
    ) -> InputBinding:
        """
        Create an InputBinding for a training phase.

        Description:
            This method creates a phase-specific binding that attaches a sampler
            between an upstream FeatureSet and a head GraphNode.

            Conceptually, this binding modifies an existing graph edge
            (FeatureSet -> GraphNode) by inserting a sampler that controls how
            samples are batched and fed into the node during training.

            No data is materialized at construction time. The sampler is only
            executed when the training phase runs.

        Args:
            node (GraphNode | str):
                The head GraphNode that will receive input data during training.
                Accepted values:
                - GraphNode instance
                - GraphNode label (str)
                - GraphNode ID (str)

            sampler (BaseSampler):
                The sampler used to generate batches from the upstream FeatureSet
                (e.g., random batches, contrastive roles, paired samples).

            upstream (FeatureSet | FeatureSetView | str | None):
                Identifies which upstream FeatureSet connection of `node` this
                binding applies to.
                Accepted values:
                - FeatureSet instance
                - FeatureSetView instance
                - FeatureSet node ID or label (str)
                - None, only if `node` has exactly one upstream FeatureSet

                If the node has multiple upstream FeatureSets, this argument
                is required to disambiguate which input is being bound.

            split (str, optional):
                Optional split name of the upstream FeatureSet (e.g. "train", "val").
                If provided, only rows from this split are sampled.
                If None, the entire FeatureSet is used.

            stream (str, optional):
                Output stream name from the sampler to feed into `node`.
                Required only if the sampler produces multiple streams.
                Defaults to STREAM_DEFAULT.

        Returns:
            InputBinding:
                A fully specified training InputBinding that can be passed
                directly to a TrainPhase.

        """
        from modularml.core.sampling.base_sampler import BaseSampler

        # Validate node
        node_id = _normalize_node_value_to_id(value=node)

        # Validate sampler and stream
        if not isinstance(sampler, BaseSampler):
            msg = f"Sampler must be of tyep BaseSampler. Received: {type(sampler)}."
            raise TypeError(msg)
        if stream not in sampler.stream_names:
            msg = (
                f"No stream '{stream}' exists in sampler. "
                f"Available: {sampler.stream_names}."
            )
            raise ValueError(msg)

        # Resolve FeatureSetReference
        ups_ref = _resolve_upstream_featureset_ref(node_id=node_id, val=upstream)

        # Validate split name, if defined
        if split is not None:
            fs = ups_ref.resolve().source
            if split not in fs.available_splits:
                msg = (
                    f"Split '{split}' does not exist in FeatureSet '{fs.label}'. "
                    f"Available splits: {fs.available_splits}."
                )
                raise ValueError(msg)

        # Return binding
        return InputBinding(
            node_id=node_id,
            upstream_ref=ups_ref,
            split=split,
            sampler=sampler,
            stream=stream,
        )

    @classmethod
    def for_evaluation(
        cls,
        *,
        node: GraphNode | str,
        upstream: FeatureSet | FeatureSetView | str | None,
        split: str | None = None,
    ) -> InputBinding:
        """
        Create an InputBinding for an evaluation phase.

        Description:
            This method creates a phase-specific binding that directly feeds
            data from an upstream FeatureSet into a head GraphNode without
            applying a sampler.

            Evaluation bindings typically iterate over all samples in a split
            (or the full FeatureSet) and are used for validation, testing, or
            inference.

        Args:
            node (GraphNode | str):
                The head GraphNode that will receive input data during evaluation.
                Accepted values:
                - GraphNode instance
                - GraphNode label (str)
                - GraphNode ID (str)

            upstream (FeatureSet | FeatureSetView | str | None):
                Identifies which upstream FeatureSet connection of `node` this
                binding applies to.
                Accepted values:
                - FeatureSet instance
                - FeatureSetView instance
                - FeatureSet node ID or label (str)
                - None, only if `node` has exactly one upstream FeatureSet

                If the node has multiple upstream FeatureSets, this argument
                is required to disambiguate which input is being bound.

            split (str, optional):
                Optional split name of the upstream FeatureSet (e.g. "val", "test").
                If provided, only rows from this split are used.
                If None, the entire FeatureSet is evaluated.

        Returns:
            InputBinding:
                A fully specified evaluation InputBinding that can be passed
                directly to an EvalPhase.

        """
        # Validate node
        node_id = _normalize_node_value_to_id(value=node)

        # Resolve FeatureSetReference
        ups_ref = _resolve_upstream_featureset_ref(node_id=node_id, val=upstream)

        # Validate split name, if defined
        if split is not None:
            fs = ups_ref.resolve().source
            if split not in fs.available_splits:
                msg = (
                    f"Split '{split}' does not exist in FeatureSet '{fs.label}'. "
                    f"Available splits: {fs.available_splits}."
                )
                raise ValueError(msg)

        # Return binding (no sampler, no stream semantics)
        return InputBinding(
            node_id=node_id,
            upstream_ref=ups_ref,
            split=split,
            sampler=None,
            stream=STREAM_DEFAULT,
        )

    # ================================================
    # Runtime Resolution
    # ================================================
    def resolve_input_view(self) -> FeatureSetView:
        """
        Resolves the FeatureSetView for the `upstream_ref`.

        Returns:
            FeatureSetView:
                A view of the FeatureSet specified by `upstream_ref`. If `split` is
                defined, the returned view is restricted to only the indices of the
                `split`.

        """
        # Get upstream FeatureSet
        # This is only a column-wise view over the FeatureSet
        fsv: FeatureSetView = self.upstream_ref.resolve()

        # If split is defined, need to intersect view with split row indices
        if self.split is not None:
            split_view: FeatureSetView = fsv.source.get_split(split_name=self.split)
            fsv = fsv.take_intersection(other=split_view)

        return fsv

    def materialize_batches(
        self,
        *,
        show_progress: bool = True,
    ) -> list[BatchView]:
        """
        Executes sampling of the source data defined by this binding.

        Args:
            show_progress (bool, optional):
                Whether to show a progress bar of the batch construction process.

        Returns:
            list[BatchView]:
                The materialized batches for the sampler and stream defined by this binding.

        """
        if self.sampler is None:
            raise ValueError("Cannot materialize batches for a `sampler` of None.")

        # Bind resolved source to sampler
        fsv = self.resolve_input_view()
        self.sampler.bind_sources(sources=[fsv])

        # Create batches for all streams defined by sampler
        self.sampler.materialize_batches(show_progress=show_progress)

        # Return only the batches for the specified stream
        return self.sampler.get_batches(stream=self.stream)

    # ================================================
    # Configurable
    # ================================================
    def get_config(self) -> dict[str, Any]:
        """
        Return configuration details required to reconstruct this binding.

        Returns:
            dict[str, Any]:
                Configuration used to reconstruct the binding.

        """
        return {
            "node_id": self.node_id,
            "upstream_ref": self.upstream_ref.get_config(),
            "split": self.split,
            "sampler": self.sampler.get_config() if self.sampler is not None else None,
            "stream": self.stream,
        }

    @classmethod
    def from_config(cls, config: dict) -> ExperimentPhase:
        """
        Construct a phase from a configuration dictionary.

        Args:
            config (dict[str, Any]):
                Configuration details. Keys must be strings.

        Returns:
            ExperimentPhase: Reconstructed phase.

        """
        sampler = None
        if config.get("sampler") is not None:
            sampler = BaseSampler.from_config(config=config["sampler"])
        return cls(
            node_id=config["node_id"],
            upstream_ref=FeatureSetReference.from_config(config=config["upstream_ref"]),
            split=config["split"],
            sampler=sampler,
            stream=config["stream"],
        )


class ExperimentPhase(ABC):
    """Abstract base for executable experiment phases."""

    def __init__(
        self,
        label: str,
        input_sources: list[InputBinding],
        losses: list[AppliedLoss] | None = None,
        active_nodes: list[GraphNode] | None = None,
        callbacks: list[Callback] | None = None,
    ):
        """
        Initiallizes a new phase of the experiment.

        Args:
            label (str):
                A label to assign to this phase of the experiment. Used for logging.

            input_sources (list[InputBinding]):
                Input bindings for each head node in ModelGraph.

            losses (list[AppliedLoss], optional):
                A list of losses to be applied during this experiment phase.

            active_nodes (list[GraphNode] | None, optional):
                A list of active GraphNodes in this phase of the experiment. Nodes can
                be listed via their ID, label, or with the actual node instance. If
                None, all nodes comprising the ModelGraph are used. Defaults to None.

            callbacks (list[Callback] | None, optional):
                An optional list of Callbacks to run during phase execution.

        """
        self.label = label
        self.input_sources = self._normalize_input_sources(sources=input_sources)
        self.losses: list[AppliedLoss] = ensure_list(losses)
        self._validate_losses()
        self.callbacks: list[Callback] = ensure_list(callbacks)
        self._validate_callbacks()
        self.active_nodes = self._resolve_active_nodes(active_nodes)
        self._validate_inputs_for_head_nodes()

    def __repr__(self):
        return f"ExperimentPhase(label={self.label})"

    # ================================================
    # Convenience Constructors
    # ================================================
    @classmethod
    def _build_input_sources_from_split(
        cls,
        *,
        split: str,
        sampler: BaseSampler | None = None,
        active_nodes: list[str | GraphNode] | None = None,
    ) -> list[InputBinding]:
        """
        Build InputBindings automatically from a split name.

        Rules:
            - All active head nodes must resolve to exactly one upstream FeatureSet
            - All must resolve to the same FeatureSet
            - The FeatureSet must contain the given split

        Args:
            split (str):
                Split name of the upstream FeatureSet (e.g. "train", "val").
                Onnly rows from this split are use for phase execution.

            sampler (BaseSampler, optional):
                An optional sampler to use to generate batches from this split.
                Required if this binding is for a TrainPhase.

            active_nodes (list[GraphNode] | None, optional):
                A list of active GraphNodes in this phase of the experiment. Nodes can
                be listed via their ID, label, or with the actual node instance. If
                None, all nodes comprising the ModelGraph are used. Defaults to None.

        """
        # Validate environment
        exp_ctx = ExperimentContext.get_active()
        mg = exp_ctx.model_graph
        if mg is None:
            msg = "Cannot infer input sources without a registered ModelGraph."
            raise RuntimeError(msg)

        # Resolve active nodes
        active_node_ids = cls._resolve_active_nodes(nodes=active_nodes)

        # Identify active head nodes
        head_nodes: list[GraphNode] = [
            exp_ctx.get_node(node_id=n_id, enforce_type="GraphNode")
            for n_id in mg.head_nodes
            if n_id in active_node_ids
        ]
        if not head_nodes:
            msg = "No active head nodes found for phase."
            raise ValueError(msg)

        # Resolve upstream fsv per head node
        fs_refs: list[FeatureSetReference] = []
        for node in head_nodes:
            ups = _get_upstream_featureset_refs_for_node(node.node_id)
            if len(ups) != 1:
                msg = (
                    f"Head node '{node.label}' has {len(ups)} upstream FeatureSets. "
                    "Automatic split-based binding requires exactly one. "
                    "Use the default phase constructor instead."
                )
                raise ValueError(msg)
            fs_refs.append(ups[0])

        # Ensure all refs point to same FeatureSet
        fs_ids = {ref.node_id for ref in fs_refs}
        if len(fs_ids) != 1:
            msg = (
                "Automatic split-based binding requires all head nodes to "
                "share the same upstream FeatureSet. "
                "Use the default phase constructor instead."
            )
            raise ValueError(msg)

        # Validate split exists
        fs = fs_refs[0].resolve().source
        if split not in fs.available_splits:
            msg = (
                f"Split '{split}' does not exist in FeatureSet '{fs.label}'. "
                f"Available splits: {fs.available_splits}."
            )
            raise ValueError(msg)

        # Create bindings
        return [
            node.create_input_binding(
                split=split,
                sampler=sampler,
            )
            for node in head_nodes
        ]

    # ================================================
    # Validation
    # ================================================
    def _validate_losses(self):
        from modularml.core.training.applied_loss import AppliedLoss

        # Validate loss type
        loss_lbls = []
        for ls in self.losses:
            if not isinstance(ls, AppliedLoss):
                msg = f"Loss entries must be of type AppliedLoss. Received: {type(ls)}."
                raise TypeError(msg)
            loss_lbls.append(ls.label)

        # Ensure unique loss labels (used for tracking)
        dup_lbls = find_duplicates(items=loss_lbls)
        if len(dup_lbls) > 0:
            msg = (
                f"Multiple AppliedLosses have the same label: {dup_lbls}. "
                "Loss labels must be unique."
            )
            raise ValueError(msg)

    def _validate_callbacks(self):
        from modularml.core.experiment.callbacks.callback import Callback

        # Validate callback type
        callback_lbls = []
        for cb in self.callbacks:
            if not isinstance(cb, Callback):
                msg = (
                    f"Callback entries must be of type Callback. Received: {type(cb)}."
                )
                raise TypeError(msg)
            callback_lbls.append(cb.label)

        # Ensure unique callback labels (used for tracking)
        dup_lbls = find_duplicates(items=callback_lbls)
        if len(dup_lbls) > 0:
            msg = (
                f"Multiple Callbacks have the same label: {dup_lbls}. "
                "Callbacks labels must be unique."
            )
            raise ValueError(msg)

        # Ensure callbacks are ordered for execution
        self.callbacks.sort(key=lambda cb: cb._exec_order)

    def _normalize_input_sources(
        self,
        sources: list[InputBinding],
    ) -> list[InputBinding]:
        """
        Validate input sources.

        Returns:
            list[InputBinding]:
                Validated/cleaned bindings.

        """
        sources = ensure_list(sources)
        clean_sources: list[InputBinding] = []
        for binding in sources:
            # Validate binding
            if not isinstance(binding, InputBinding):
                msg = (
                    f"Input source values must be of type InputBinding. "
                    f"Received: {type(binding)}."
                )
                raise TypeError(msg)
            clean_sources.append(binding)

        return clean_sources

    @staticmethod
    def _resolve_active_nodes(
        nodes: list[str | GraphNode] | None,
    ) -> list[str]:
        """
        Resolve active GraphNode.

        Returns:
            list[str]
                List of node IDs of active nodes in this phase.

        """
        exp_ctx = ExperimentContext.get_active()
        # If None, use all nodes in the active ModelGraph
        if nodes is None:
            mg = exp_ctx.model_graph
            if mg is None:
                msg = "No ModelGraph has been set in the active ExperimentContext. Either explictly list out `active_nodes`, or register a ModelGraph."
                raise ValueError(msg)

            # Get all node IDs of the nodes comprising the ModelGraph
            return list(mg.nodes.keys())

        # Otherwise, normalize each node value to a known node ID
        node_ids: list[str] = []
        for n in ensure_list(nodes):
            n_id = _normalize_node_value_to_id(value=n)
            node_ids.append(n_id)

        return node_ids

    def _validate_inputs_for_head_nodes(self):
        """Validates that all head nodes have defined inputs."""
        # Get active ModelGraph, must be defined prior to phase init
        exp_ctx = ExperimentContext.get_active()
        mg = exp_ctx.model_graph
        if mg is None:
            msg = "Cannot define an ExperimentPhase before a ModelGraph has been registered."
            raise RuntimeError(msg)

        # Check that all active head nodes have all inputs defined
        for n_id in mg.head_nodes:
            # Skip if not active
            if n_id not in self.active_nodes:
                continue

            node = exp_ctx.get_node(node_id=n_id, enforce_type="GraphNode")

            # Get all upstream FeatureSetRefs of this head node
            ups_fs_refs = _get_upstream_featureset_refs_for_node(node_id=n_id)
            req_refs = [
                inp.upstream_ref for inp in self.input_sources if inp.node_id == n_id
            ]
            missing: list[FeatureSetReference] = [
                ref for ref in ups_fs_refs if ref not in req_refs
            ]
            if missing:
                msg = (
                    f"Head node '{node.label}' is missing an input binding for "
                    f"upstream FeatureSet(s): '{[r.node_label for r in missing]}'."
                )
                raise ValueError(msg)

    # ================================================
    # Stop Control
    # ================================================
    def request_stop(self) -> None:
        """Request early termination of this phase."""
        raise NotImplementedError

    # ================================================
    # Execution
    # ================================================
    @abstractmethod
    def iter_execution(
        self,
        *,
        results: PhaseResults | None = None,
        **kwargs,
    ) -> Iterator[ExecutionContext]:
        """Iterate over execution steps for this phase."""
        ...

    # ================================================
    # Configurable
    # ================================================
    def get_config(self) -> dict[str, Any]:
        """
        Return configuration details required to reconstruct this phase.

        Returns:
            dict[str, Any]:
                Configuration used to reconstruct the phase.

        """
        losses_cfg = None
        if self.losses is not None:
            losses_cfg = [ls.get_config() for ls in self.losses]

        return {
            "label": self.label,
            "input_sources": [inp.get_config() for inp in self.input_sources],
            "losses": losses_cfg,
            "active_nodes": self.active_nodes,
            "callbacks": [cb.get_config() for cb in ensure_list(self.callbacks)],
        }

    @classmethod
    def from_config(cls, config: dict) -> ExperimentPhase:
        """
        Construct a phase from a configuration dictionary.

        Args:
            config (dict[str, Any]):
                Configuration details. Keys must be strings.

        Returns:
            ExperimentPhase: Reconstructed phase.

        """
        if "phase_type" not in config:
            raise ValueError("ExperimentPhase config must include `phase_type`")

        # Create subclasses directly
        phase_type = config["phase_type"]
        if phase_type == "EvalPhase":
            from modularml.core.experiment.phases.eval_phase import EvalPhase

            return EvalPhase.from_config(config=config)

        if phase_type == "TrainPhase":
            from modularml.core.experiment.phases.train_phase import TrainPhase

            return TrainPhase.from_config(config=config)

        if phase_type == "FitPhase":
            from modularml.core.experiment.phases.fit_phase import FitPhase

            return FitPhase.from_config(config=config)

        msg = f"Unknown ExperimentPhase subclass: {phase_type}."
        raise ValueError(msg)

    # ================================================
    # Visualizer
    # ================================================
    def visualize(self):
        """Displays a mermaid diagram for this phase."""
        return Visualizer(self).display_mermaid()
