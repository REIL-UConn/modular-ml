from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from modularml.callbacks.eval_loss_metric import EvalLossMetric
from modularml.core.experiment.callbacks.callback import Callback, CallbackResult
from modularml.core.experiment.phases.eval_phase import EvalPhase
from modularml.utils.logging.warnings import warn

if TYPE_CHECKING:
    from modularml.callbacks.metric import EvaluationMetric
    from modularml.core.data.batch import Batch
    from modularml.core.data.execution_context import ExecutionContext
    from modularml.core.data.featureset_view import FeatureSetView
    from modularml.core.experiment.experiment import Experiment
    from modularml.core.experiment.phases.phase import ExperimentPhase
    from modularml.core.experiment.results.eval_results import EvalResults
    from modularml.core.experiment.results.phase_results import PhaseResults
    from modularml.core.references.execution_reference import TensorLike
    from modularml.core.topology.graph_node import GraphNode
    from modularml.core.training.applied_loss import AppliedLoss
    from modularml.utils.data.data_format import DataFormat


class Evaluation(Callback):
    """
    Periodically run an EvalPhase during training and emit evaluation results.

    Description:
        This callback is designed to be attached to a TrainPhase. At configured
        intervals (e.g., every N epochs), it triggers execution of an EvalPhase.

        The callback returns a :class:`CallbackResult` whose payload is an
        :class:`EvaluationResult` containing the produced eval results and the
        epoch/batch scope at which the evaluation was triggered.

    """

    def __init__(
        self,
        *,
        eval_phase: EvalPhase,
        every_n_epochs: int = 1,
        run_on_start: bool = False,
        label: str | None = None,
        metrics: list[EvaluationMetric] | None = None,
    ):
        """
        Initialize an Evaluation callback.

        Args:
            eval_phase (EvalPhase):
                The evaluation phase definition to execute when triggered.

            every_n_epochs (int, optional):
                Run evaluation every N epochs. Must be >= 1. Defaults to 1.

            run_on_start (bool, optional):
                If True, run evaluation once at the start of training (epoch 0,
                before any training occurs). Defaults to False.

            label (str | None, optional):
                Stable identifier for this callback within a results container.
                If None, defaults to `eval_phase.label`.

            metrics (list[EvaluationMetric] | None, optional):
                A list of EvaluationMetric instances to attach to this callback.
                Each metric's `EvaluationMetric.extract` method will be
                called automatically whenever this Evaluation produces results.
                Defaults to None.

        """
        super().__init__(label=label or eval_phase.label)
        if every_n_epochs < 1:
            raise ValueError("every_n_epochs must be >= 1")
        # Ensure eval phase has no callbacks
        self.eval_phase = eval_phase
        if self.eval_phase.callbacks:
            msg = (
                "The EvalPhase provided to Evaluation has its own callbacks. Nested"
                "callbacks are not currently supported, and will not be executed."
            )
            warn(msg)

        self.every_n_epochs = every_n_epochs
        self.run_on_start = run_on_start
        self._metrics: list[EvaluationMetric] = list(metrics or [])

        # Ensure any losses defined in metrics are attached to phase
        loss_metrics = [m for m in self._metrics if isinstance(m, EvalLossMetric)]
        for lm in loss_metrics:
            if lm._loss not in self.eval_phase.losses:
                self.eval_phase.losses.append(lm._loss)

    # ================================================
    # Convenience Constructors
    # ================================================
    @classmethod
    def from_split(
        cls,
        *,
        label: str,
        split: str,
        losses: list[AppliedLoss] | None = None,
        active_nodes: list[GraphNode] | None = None,
        batch_size: int | None = None,
        every_n_epochs: int = 1,
        run_on_start: bool = False,
        metrics: list[EvaluationMetric] | None = None,
    ) -> Evaluation:
        """
        Initialize an Evaluation callback on a specific FeatureSet split.

        Args:
            label (str):
                Stable identifier for this callback within a results container.

            split (str):
                The FeatureSet split name to run evaluation on.

            losses (list[AppliedLoss] | None, optional):
                A list of losses to be applied during this evaluation phase.
                Defaults to None.

            active_nodes (list[GraphNode] | None, optional):
                A list of GraphNodes to run a forward phase on. Nodes can
                be listed via their ID, label, or with the actual node instance. If
                None, all nodes comprising the ModelGraph are used. Defaults to None.

            batch_size (int | None, optional):
                If defined, limits the number of samples to using during a single
                forward pass. Otherwise, all samples are passed at once.

            every_n_epochs (int, optional):
                Run evaluation every N epochs. Must be >= 1. Defaults to 1.

            run_on_start (bool, optional):
                If True, run evaluation once at the start of training (epoch 0,
                before any training occurs). Defaults to False.

            metrics (list[EvaluationMetric] | None, optional):
                A list of EvaluationMetric instances to attach to this callback.
                Each metric's `EvaluationMetric.extract` method will be
                called automatically whenever this Evaluation produces results.
                Defaults to None.

        """
        return cls(
            eval_phase=EvalPhase.from_split(
                label=label,
                split=split,
                losses=losses,
                active_nodes=active_nodes,
                batch_size=batch_size,
            ),
            every_n_epochs=every_n_epochs,
            run_on_start=run_on_start,
            label=label,
            metrics=metrics,
        )

    # ================================================
    # Metrics
    # ================================================
    @property
    def metrics(self) -> list[EvaluationMetric]:
        """The list of attached EvaluationMetric instances."""
        return list(self._metrics)

    def attach_metric(self, metric: EvaluationMetric) -> None:
        """
        Attach an EvaluationMetric to this callback.

        Args:
            metric (EvaluationMetric):
                A metric whose `EvaluationMetric.extract` method will
                be called whenever this Evaluation produces results.

        """
        self._metrics.append(metric)

    def _run_metrics(
        self,
        *,
        eval_result: EvaluationCallbackResult,
        phase: ExperimentPhase,
        exec_ctx: ExecutionContext | None = None,
        results: PhaseResults | None = None,
    ) -> None:
        """
        Run all attached metrics against an evaluation result.

        Description:
            For each attached EvaluationMetric, calls its :meth:`extract`
            method with the evaluation result. Any returned MetricResult
            is stored in the phase results and logged to the MetricStore.

        """
        if results is None:
            return

        for metric in self._metrics:
            mr = metric.extract(eval_result=eval_result, exec_ctx=exec_ctx)
            if mr is None:
                continue

            mr.bind_scope(
                callback_label=metric.label,
                phase_label=phase.label,
                epoch_idx=exec_ctx.epoch_idx if exec_ctx is not None else None,
                batch_idx=None,
                edge="end" if exec_ctx is not None else "start",
            )
            results._metrics.log(
                name=mr.metric_name,
                value=mr.metric_value,
                epoch_idx=mr.epoch_idx,
                batch_idx=mr.batch_idx,
            )

    # ================================================
    # Lifecycle Hooks
    # ================================================
    def on_phase_start(
        self,
        *,
        experiment: Experiment,
        phase: ExperimentPhase,
        results: PhaseResults | None = None,
    ) -> EvaluationCallbackResult | None:
        """
        Optionally run EvalPhase once at the start of a phase.

        Args:
            experiment (Experiment):
                The Experiment on which the phase is attached.
                The model graph can be accessed via `experiment.model_graph`.
            phase (ExperimentPhase):
                The current phase.
            results (PhaseResults | None):
                The results container for this phase, if available.

        Returns:
            EvaluationCallbackResult | None:
                Optional eval results produced at phase start.

        """
        if not self.run_on_start:
            return None

        eval_result = EvaluationCallbackResult(
            callback_label=self.label,
            eval_results=experiment.preview_phase(phase=self.eval_phase),
        )
        self._run_metrics(
            eval_result=eval_result,
            phase=phase,
            results=results,
        )
        return eval_result

    def on_epoch_end(
        self,
        *,
        experiment: Experiment,
        phase: ExperimentPhase,
        exec_ctx: ExecutionContext,
        results: PhaseResults | None = None,
    ) -> EvaluationCallbackResult | None:
        """
        Run evaluation at the end of selected epochs.

        Args:
            experiment (Experiment):
                The Experiment on which the callback is attached.
                The model graph can be accessed via `experiment.model_graph`.
            phase (ExperimentPhase):
                The current phase on which the callback is attached.
            exec_ctx (ExecutionContext):
                The last execution context of the epoch.
            results (PhaseResults | None):
                The results container for this phase, if available.

        Returns:
            EvaluationCallbackResult | None:
                Eval results produced at epoch end (if selected epoch)

        """
        epoch_idx = exec_ctx.epoch_idx
        if epoch_idx is None:
            return None

        # Run on epoch indices: N-1, 2N-1, 3N-1, etc
        if ((epoch_idx + 1) % self.every_n_epochs) != 0:
            return None

        eval_result = EvaluationCallbackResult(
            callback_label=self.label,
            eval_results=experiment.preview_phase(phase=self.eval_phase),
        )
        self._run_metrics(
            eval_result=eval_result,
            phase=phase,
            exec_ctx=exec_ctx,
            results=results,
        )
        return eval_result

    # ================================================
    # Configurable
    # ================================================
    def get_config(self) -> dict[str, Any]:
        """
        Return configuration details required to reconstruct this callback.

        Returns:
            dict[str, Any]:
                Configuration used to reconstruct the callback.

        """
        return {
            "callback_type": self.__class__.__qualname__,
            "label": self.label,
            "eval_phase": self.eval_phase.get_config(),
            "every_n_epochs": self.every_n_epochs,
            "run_on_start": self.run_on_start,
            "metrics": [m.get_config() for m in self._metrics],
        }

    @classmethod
    def from_config(cls, config: dict) -> Evaluation:
        """
        Construct an Evaluation callback from config data.

        Args:
            config (dict[str, Any]):
                Configuration details. Keys must be strings.

        Returns:
            Evaluation: Reconstructed callback.

        """
        if config.get("callback_type") != cls.__qualname__:
            msg = f"Invalid config data for {cls.__qualname__} callback."
            raise ValueError(msg)

        metrics = [Callback.from_config(m_cfg) for m_cfg in config.get("metrics", [])]
        return cls(
            eval_phase=EvalPhase.from_config(config=config["eval_phase"]),
            every_n_epochs=config["every_n_epochs"],
            run_on_start=config["run_on_start"],
            label=config["label"],
            metrics=metrics,
        )


@dataclass
class EvaluationCallbackResult(CallbackResult):
    kind: ClassVar[str] = "evaluation"

    # Raw EvalResults
    eval_results: EvalResults | None = None

    def get_eval_results(self) -> EvalResults:
        if self.eval_results is None:
            msg = "This callback has no receorded results."
            raise ValueError(msg)

        return self.eval_results

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
        return self.get_eval_results().batch_indices

    @property
    def n_batches(self) -> int:
        """The number of batches executed during evaluation."""
        return self.get_eval_results().n_batches

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

        Example:
            >>> # Get all predictions stacked
            >>> predictions = eval_cb_results.stacked_tensors( # doctest: +SKIP
            ...    node="output_node",
            ...    domain="outputs",
            ...)

            >>> # Get targets, unscaled, as numpy
            >>> targets = eval_cb_results.stacked_tensors(  # doctest: +SKIP
            ...     node="output_node",
            ...     domain="targets",
            ...     fmt="np",
            ...     unscale=True,
            ... )

        """
        return self.get_eval_results().stacked_tensors(
            node=node,
            domain=domain,
            role=role,
            fmt=fmt,
            unscale=unscale,
        )

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

        Example:
            Batches are accessed for a specified node:

            >>> batch = eval_cb_results.stacked_batches(  # doctest: +SKIP
            ...     node="output_node"
            ... )
            >>> print(f"Total samples: {batch.batch_size}")  # doctest: +SKIP
            >>> print(f"Outputs shape: {batch.outputs.shape}")  # doctest: +SKIP

        """
        return self.get_eval_results().stacked_batches(node=node, fmt=fmt)

    def aggregated_losses(
        self,
        node: str | GraphNode,
        *,
        reducer: Literal["mean", "sum"] = "mean",
    ) -> dict[str, float]:
        """
        Aggregates losses over all batches within this eval phase.

        Args:
            node (str | GraphNode):
                The node to filter losses to. Can be the node instance,
                its ID, or its label.

            reducer (Literal['mean', 'sum']):
                How losses should be aggregated. Defaults to "mean".

        Returns:
            dict[str, float]:
                Aggregated lossed, keyed by the AppliedLoss label.

        """
        return self.get_eval_results().aggregated_losses(
            node=node,
            reducer=reducer,
        )

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
        return self.get_eval_results().source_views(
            node=node,
            role=role,
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
        return self.get_eval_results().source_view(
            node=node,
            role=role,
            batch=batch,
        )
