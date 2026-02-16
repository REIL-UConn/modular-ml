from __future__ import annotations

from typing import TYPE_CHECKING, overload

from modularml.core.experiment.experiment_context import (
    ExperimentContext,
    RegistrationPolicy,
)
from modularml.core.experiment.phases.eval_phase import EvalPhase
from modularml.core.experiment.phases.phase import ExperimentPhase
from modularml.core.experiment.phases.phase_group import PhaseGroup
from modularml.core.experiment.phases.train_phase import TrainPhase
from modularml.core.experiment.results.eval_results import EvalResults
from modularml.core.experiment.results.group_results import PhaseGroupResults
from modularml.core.experiment.results.train_results import TrainResults
from modularml.utils.environment.environment import IN_NOTEBOOK

if TYPE_CHECKING:
    from modularml.core.topology.model_graph import ModelGraph


class Experiment:
    def __init__(
        self,
        label: str,
        registration_policy: RegistrationPolicy | str | None = None,
        ctx: ExperimentContext | None = None,
    ):
        """
        Constructs a new Experiment.

        Args:
            label (str):
                A name to assign to this experiment.

            registration_policy (RegistrationPolicy | str, optional):
                Default registration policy for nodes created after this Experiment
                is constructed.

            ctx (ExperimentContext, optional):
                Context to associate with this Experiment. If None, a new context
                is created and activated.

        """
        self.label = label

        # Initialize / attach context
        if ctx is None:
            ctx = ExperimentContext(
                experiment=self,
                registration_policy=registration_policy,
            )
            ExperimentContext._set_active(ctx)
        else:
            ctx.set_experiment(self)
        self._ctx = ctx

        # Initialize phase registry
        self._exec_plan = PhaseGroup(label=self.label)

    # ================================================
    # Constructors
    # ================================================
    @classmethod
    def from_active_context(
        cls,
        label: str,
        registration_policy: RegistrationPolicy | str | None = None,
    ) -> Experiment:
        """
        Construct an Experiment using the active ExperimentContext.

        Description:
            Creates a new Experiment instance, but retains all nodes that have been
            registered in the current ExperimentContext.

        Args:
            label (str):
                A name to assign to this experiment.

            registration_policy (RegistrationPolicy | str | None, optional):
                Default registration policy for nodes created after this Experiment
                is constructed.

        Returns:
            Experiment: A new Experiment utilizing the active context.

        """
        active_ctx = ExperimentContext.get_active()
        if active_ctx._experiment_ref is not None:
            msg = "An Experiment has already been associated with the active context."
            raise ValueError(msg)

        return cls(
            label=label,
            registration_policy=registration_policy,
            ctx=active_ctx,
        )

    # ================================================
    # Properties
    # ================================================
    @property
    def ctx(self) -> ExperimentContext:
        return self._ctx

    @property
    def model_graph(self) -> ModelGraph | None:
        """The ModelGraph defined in this Experiment."""
        return self._ctx.model_graph

    @property
    def execution_plan(self) -> PhaseGroup:
        """Group of phases (and sub-groups) to be executed."""
        return self._exec_plan

    # ================================================
    # Execution
    # ================================================
    def run_training(
        self,
        phase: TrainPhase,
        *,
        show_sampler_progress: bool = True,
        show_training_progress: bool = True,
        persist_progress: bool = IN_NOTEBOOK,
        persist_epoch_progress: bool = IN_NOTEBOOK,
    ) -> TrainResults:
        """
        Executes a training phase on this experiment.

        Args:
            phase (TrainPhase):
                Training phase to be executed.
            show_sampler_progress (bool, optional):
                Whether to show a progress bar for sampler batching.
                Defaults to True.
            show_training_progress (bool, optional):
                Whether to show a progress bar for training execution.
                Defaults to True.
            persist_progress (bool, optional):
                Whether to leave all epoch progress bars shown after they complete.
                Defaults to `IN_NOTEBOOK` (True if working in a notebook, False if in
                a Python script).
            persist_epoch_progress (bool, optional):
                Whether to leave all per-epoch training bars shown after they complete.
                Defaults to `IN_NOTEBOOK` (True if working in a notebook, False if in
                a Python script).

        Returns:
            TrainResults: Tracked results from training.

        """
        # Ensure active nodes are not frozen
        self.model_graph.unfreeze(phase.active_nodes)

        # Run training and track results
        res = TrainResults(label=phase.label)
        for ctx in phase.iter_execution(
            results=res,
            show_sampler_progress=show_sampler_progress,
            show_training_progress=show_training_progress,
            persist_progress=persist_progress,
            persist_epoch_progress=persist_epoch_progress,
        ):
            self.model_graph.train_step(
                ctx=ctx,
                losses=phase.losses,
                active_nodes=phase.active_nodes,
            )
            res.add_execution_context(ctx=ctx)

        res.validation_callback_labels = [cb.label for cb in phase.validation_callbacks]

        # TODO: should this be returned or tracked internally?
        return res

    def run_evaluation(
        self,
        phase: EvalPhase,
        *,
        show_eval_progress: bool = False,
        persist_progress: bool = IN_NOTEBOOK,
    ) -> EvalResults:
        """
        Executes an evaluation phase on this experiment.

        Args:
            phase (EvalPhase):
                Evaluation phase to be executed.
            show_eval_progress (bool, optional):
                Whether to show a progress bar for eval batches. Defaults to False.
            persist_progress (bool, optional):
                Whether to leave all eval progress bars shown after they complete.
                Defaults to `IN_NOTEBOOK` (True if working in a notebook, False if in
                a Python script).

        Returns:
            EvalResults: Tracked results from evaluation.

        """
        # Ensure all nodes are frozen
        self.model_graph.freeze()

        # Run evaluation and track results
        res = EvalResults(label=phase.label)
        for ctx in phase.iter_execution(
            results=res,
            show_eval_progress=show_eval_progress,
            persist_progress=persist_progress,
        ):
            self.model_graph.eval_step(
                ctx=ctx,
                losses=phase.losses,
                active_nodes=phase.active_nodes,
            )
            res.add_execution_context(ctx=ctx)

        # TODO: should this be returned or tracked internally?
        return res

    @overload
    def run_phase(self, phase: TrainPhase, **kwargs) -> TrainResults: ...
    @overload
    def run_phase(self, phase: EvalPhase, **kwargs) -> EvalResults: ...
    def run_phase(
        self,
        phase: ExperimentPhase,
        **kwargs,
    ) -> TrainResults | EvalResults:
        """
        Executes a given phase.

        Args:
            phase (ExperimentPhase):
                The phase to run.
            **kwargs:
                Display flags forwarded to the phase-specific run method.

        Returns:
            TrainResults | EvalResults: Phase results.

        """
        if isinstance(phase, TrainPhase):
            train_keys = {
                "show_sampler_progress",
                "show_training_progress",
                "persist_progress",
                "persist_epoch_progress",
            }
            return self.run_training(
                phase,
                **{k: v for k, v in kwargs.items() if k in train_keys},
            )

        if isinstance(phase, EvalPhase):
            eval_keys = {"show_eval_progress", "persist_progress"}
            return self.run_evaluation(
                phase,
                **{k: v for k, v in kwargs.items() if k in eval_keys},
            )

        msg = f"Unknown phase type: {type(phase)}"
        raise TypeError(msg)

    def run_phase_group(
        self,
        group: PhaseGroup,
        **kwargs,
    ) -> PhaseGroupResults:
        """
        Execute all phases in a PhaseGroup.

        Args:
            group (PhaseGroup):
                The PhaseGroup to execute.
            **kwargs:
                Display flags forwarded to each phase's run method.

        Returns:
            PhaseGroupResults:
                Results of the executed phase group.

        """
        results = PhaseGroupResults(label=group.label)
        for val in group.all:
            if isinstance(val, ExperimentPhase):
                res = self.run_phase(phase=val, **kwargs)
                results.add_result(result=res)
            elif isinstance(val, PhaseGroup):
                res = self.run_phase_group(group=val, **kwargs)
                results.add_result(result=res)
            else:
                msg = (
                    "Unsupported group element. Expected ExperimentPhase "
                    f"or PhaseGroup. Received: {type(val)}."
                )
                raise TypeError(msg)

        return results
