from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

from modularml.core.experiment.callback import Callback, CallbackResult
from modularml.core.experiment.phases.eval_phase import EvalPhase

if TYPE_CHECKING:
    from modularml.context.execution_context import ExecutionContext
    from modularml.core.experiment.experiment import Experiment
    from modularml.core.experiment.phases.eval_results import EvalResults
    from modularml.core.experiment.phases.phase import ExperimentPhase
    from modularml.core.experiment.phases.phase_result import PhaseResults
    from modularml.core.references.execution_reference import TensorLike
    from modularml.core.topology.graph_node import GraphNode
    from modularml.core.training.loss_record import LossCollection
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
                If None, defaults to the class name.

        """
        super().__init__(label=label)
        if every_n_epochs < 1:
            raise ValueError("every_n_epochs must be >= 1")
        self.eval_phase = eval_phase
        self.every_n_epochs = every_n_epochs
        self.run_on_start = run_on_start

    # ================================================
    # Lifecycle Hooks
    # ================================================
    def on_phase_start(
        self,
        *,
        experiment: Experiment,
        phase: ExperimentPhase,
    ) -> PhaseResults | None:
        """
        Optionally run EvalPhase once at the start of a phase.

        Args:
            experiment (Experiment):
                The Experiment on which the phase is attached.
                The model graph can be accessed via `experiment.model_graph`.
            phase (ExperimentPhase):
                The current phase.

        Returns:
            PhaseResults | None:
                Optional eval results produced at phase start.

        """
        if not self.run_on_start:
            return None

        return EvaluationCallbackResult(
            callback_label=self.label,
            eval_results=experiment.run_evaluation(
                phase=self.eval_phase,
                record=True,  # TODO: add eval execution to Experiment history?
            ),
        )

    def on_epoch_end(
        self,
        *,
        experiment: Experiment,
        phase: ExperimentPhase,
        exec_ctx: ExecutionContext,
    ) -> PhaseResults | None:
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

        Returns:
            PhaseResults | None:
                Eval results produced at epoch end (if selected epoch)

        """
        epoch_idx = exec_ctx.epoch_idx
        if epoch_idx is None:
            return None

        # Run on epoch indices: N-1, 2N-1, 3N-1, etc
        if ((epoch_idx + 1) % self.every_n_epochs) != 0:
            return None

        return EvaluationCallbackResult(
            callback_label=self.label,
            eval_results=experiment.run_evaluation(
                phase=self.eval_phase,
                record=True,  # TODO: add eval execution to Experiment history?
            ),
        )

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
            "eval_phase": self.eval_phase.get_config(),
            "every_n_epochs": self.every_n_epochs,
            "run_on_start": self.run_on_start,
            "label": self.label,
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
        return cls(
            eval_phase=EvalPhase.from_config(config=config["eval_phase"]),
            every_n_epochs=config["every_n_epochs"],
            run_on_start=config["run_on_start"],
            label=config["label"],
        )


@dataclass
class EvaluationCallbackResult(CallbackResult):
    kind: ClassVar[str] = "evaluation"

    # Raw EvalResults
    eval_results: EvalResults | None = None

    # ================================================
    # EvalResults Pass-Through
    # ================================================
    def get_node_outputs(
        self,
        node: str | GraphNode | None = None,
        *,
        fmt: DataFormat | None = None,
        unscale: bool = False,
    ) -> TensorLike | dict[str, TensorLike]:
        """
        Retrieves the node outputs (ie predictions) during this evaluation.

        Args:
            node (str | GraphNode | None, optional):
                Graph node (or identifier) whose outputs should be retrieved.
                If None, outputs from all nodes are returned. Defaults to None.
            fmt (DataFormat | None, optional):
                Format to return target data as. If None, the backend is inferred
                from the stored data when possible. Defaults to None.
            unscale (bool, optional):
                Whether to inverse any applied scalers to these node targets.
                Note that this is only possible when `node` is both provided and
                refers to a tail node.

        Returns:
            TensorLike | dict[str, TensorLike]:
                Concatenated outputs for a single node if `node` is specified, or a
                mapping from node IDs to concatenated data otherwise.

        """
        if self.eval_results is None:
            raise ValueError("This callback results has no recorded results.")
        return self.eval_results.get_node_outputs(node=node, fmt=fmt, unscale=unscale)

    def get_node_targets(
        self,
        node: str | GraphNode | None = None,
        *,
        fmt: DataFormat | None = None,
        unscale: bool = False,
    ) -> TensorLike | dict[str, TensorLike]:
        """
        Retrieves the node targets during this evaluation.

        Args:
            node (str | GraphNode | None, optional):
                Graph node (or identifier) whose targets should be retrieved.
                If None, targets from all nodes are returned. Defaults to None.
            fmt (DataFormat | None, optional):
                Target format to use for concatenation. If None, the backend is inferred
                from the stored data when possible. Defaults to None.
            unscale (bool, optional):
                Whether to inverse any applied scalers to these node targets.
                Note that this is only possible when `node` is both provided and
                refers to a tail node.

        Returns:
            TensorLike | dict[str, TensorLike]:
                Concatenated targets for a single node if `node` is specified, or a
                mapping from node IDs to concatenated data otherwise.

        """
        if self.eval_results is None:
            raise ValueError("This callback results has no recorded results.")
        return self.eval_results.get_node_targets(node=node, fmt=fmt, unscale=unscale)

    def get_node_losses(
        self,
        node: str | GraphNode | None = None,
    ) -> LossCollection | dict[str, LossCollection]:
        """
        Retrieve losses recorded during evaluation.

        Description:
            Collects and concatenates recorded losses from all evaluation
            batches. If a node is specified, only losses applied to that node
            are returned. Otherwise, losses for all nodes are returned keyed
            by node ID.

        Args:
            node (str | GraphNode | None, optional):
                Graph node (or identifier) whose losses should be retrieved.
                If None, losses from all nodes are returned. Defaults to None.

        Returns:
            LossCollection | dict[str, LossCollection]:
                Concatenated losses for a single node if `node` is specified, or a
                mapping from node IDs to losses otherwise.

        Raises:
            TypeError:
                If `node` is not a string or GraphNode.
            ValueError:
                If the specified node has no applied losses.

        """
        if self.eval_results is None:
            raise ValueError("This callback results has no recorded results.")
        return self.eval_results.get_node_losses(node=node)
