from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from typing import TYPE_CHECKING, Any, overload

from modularml.core.experiment.experiment_context import (
    ExperimentContext,
    RegistrationPolicy,
)
from modularml.core.experiment.phases.eval_phase import EvalPhase
from modularml.core.experiment.phases.phase import ExperimentPhase
from modularml.core.experiment.phases.phase_group import PhaseGroup
from modularml.core.experiment.phases.train_phase import TrainPhase
from modularml.core.experiment.results.eval_results import EvalResults
from modularml.core.experiment.results.execution_meta import (
    PhaseExecutionMeta,
    PhaseGroupExecutionMeta,
)
from modularml.core.experiment.results.experiment_run import ExperimentRun
from modularml.core.experiment.results.group_results import PhaseGroupResults
from modularml.core.experiment.results.train_results import TrainResults
from modularml.utils.environment.environment import IN_NOTEBOOK

if TYPE_CHECKING:
    from modularml.core.experiment.results.phase_results import PhaseResults
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
            ctx.set_registration_policy(registration_policy)
        self._ctx = ctx

        # Initialize phase registry
        self._exec_plan = PhaseGroup(label=self.label)

        # For recording execution history and checkpoints
        self._history: list[ExperimentRun] = []
        self._checkpoints: dict[str, Any] = {}
        # TODO: should checkpoint be a path or object?

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

    @property
    def history(self) -> list[ExperimentRun]:
        """All completed experiment runs in chronological order."""
        return list(self._history)

    @property
    def last_run(self) -> ExperimentRun | None:
        """Most recent ExperimentRun."""
        return self._history[-1] if self._history else None

    # ================================================
    # Execution
    # ================================================
    # Private helpers
    def _execute_training(
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

        Description:
            The provided TrainPhase will be executed regardless of whether it
            is registered to this Experiment (`execution_plan`).
            **This will mutate the experiment state, but history will not be
            recorded.**

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
        return res

    def _execute_evaluation(
        self,
        phase: EvalPhase,
        *,
        show_eval_progress: bool = False,
        persist_progress: bool = IN_NOTEBOOK,
    ) -> EvalResults:
        """
        Executes an evaluation phase on this experiment.

        Description:
            The provided EvalPhase will be executed regardless of whether it
            is registered to this Experiment (`execution_plan`).
            **This will mutate the experiment state, but history will not be
            recorded.**

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

        return res

    def _execute_phase_with_meta(
        self,
        phase: TrainPhase | EvalPhase,
        **kwargs,
    ) -> tuple[PhaseResults, PhaseExecutionMeta]:
        """
        Wraps phase execution with meta data.

        The phase is executed, with results and meta data returned.
        **This will mutate the experiment state, but history will not be
        recorded.**
        """
        phase_start = datetime.now()

        # Run phase (modifies experiment state but does not update history)
        if isinstance(phase, TrainPhase):
            train_keys = {
                "show_sampler_progress",
                "show_training_progress",
                "persist_progress",
                "persist_epoch_progress",
            }
            phase_res: TrainResults = self._execute_training(
                phase,
                **{k: v for k, v in kwargs.items() if k in train_keys},
            )
        elif isinstance(phase, EvalPhase):
            eval_keys = {"show_eval_progress", "persist_progress"}
            phase_res: EvalResults = self._execute_evaluation(
                phase,
                **{k: v for k, v in kwargs.items() if k in eval_keys},
            )
        else:
            msg = f"Expected type of TrainPhase or EvalPhase. Received: {type(phase)}."
            raise TypeError(msg)

        # Create meta for run
        phase_end = datetime.now()
        phase_meta = PhaseExecutionMeta(
            label=phase.label,
            started_at=phase_start,
            ended_at=phase_end,
            status="completed",
        )

        return phase_res, phase_meta

    def _execute_group_with_meta(
        self,
        group: PhaseGroup,
        **kwargs,
    ) -> tuple[PhaseGroupResults, PhaseGroupExecutionMeta]:
        """
        Wraps group execution with meta data.

        The group is executed, with results and meta data returned.
        **This will mutate the experiment state, but history will not be
        recorded.**
        """
        if not isinstance(group, PhaseGroup):
            msg = f"Expected type of PhaseGroup. Received: {type(group)}."
            raise TypeError(msg)

        # Construct result containers
        group_results = PhaseGroupResults(label=group.label)
        group_meta = PhaseGroupExecutionMeta(
            label=group.label,
            started_at=datetime.now(),
            ended_at=None,
        )

        # Run all items in group
        for element in group.all:
            if isinstance(element, ExperimentPhase):
                # Run phase with meta tracking
                phase_res, phase_meta = self._execute_phase_with_meta(
                    phase=element,
                    **kwargs,
                )

                # Record phase results
                group_results.add_result(phase_res)
                # Record phase meta
                group_meta.add_child(phase_meta)

            elif isinstance(element, PhaseGroup):
                # Run group with meta tracking
                sub_res, sub_meta = self._execute_group_with_meta(
                    group=element,
                    **kwargs,
                )

                # Record group results
                group_results.add_result(sub_res)
                # Record group meta
                group_meta.add_child(sub_meta)

            else:
                msg = (
                    "Unsupported group element. Expected ExperimentPhase "
                    f"or PhaseGroup. Received: {type(element)}."
                )
                raise TypeError(msg)

        # Update group meta
        group_meta.ended_at = datetime.now()

        return group_results, group_meta

    # Run API
    @overload
    def run_phase(
        self,
        phase: TrainPhase,
        *,
        show_sampler_progress: bool = True,
        show_training_progress: bool = True,
        persist_progress: bool = IN_NOTEBOOK,
        persist_epoch_progress: bool = IN_NOTEBOOK,
    ) -> TrainResults: ...
    @overload
    def run_phase(
        self,
        phase: EvalPhase,
        *,
        show_eval_progress: bool = False,
        persist_progress: bool = IN_NOTEBOOK,
    ) -> EvalResults: ...
    def run_phase(
        self,
        phase: ExperimentPhase,
        **kwargs,
    ) -> PhaseResults:
        """
        Executes a given phase.

        Description:
            The provided ExperimentPhase will be executed regardless
            of whether it is registered to this Experiment (`execution_plan`),
            and its outputs will be recorded under `history`.
            **This will mutate the experiment state**. To run a phase without
            mutating the experiment state, use `preview_phase(...)`.

        Args:
            phase (ExperimentPhase):
                The phase to run.
            **kwargs:
                Display flags forwarded to the phase-specific run method.

        Returns:
            PhaseResults: Phase results.

        """
        # Initiallize run attributes
        started_at = datetime.now()
        status = "completed"

        # Run phase and record phase-level meta data
        try:
            res, meta = self._execute_phase_with_meta(
                phase=phase,
                **kwargs,
            )
        except Exception:
            status = "failed"
            raise
        finally:
            ended_at = datetime.now()

        # Construct experiment
        run = ExperimentRun(
            label=phase.label,
            started_at=started_at,
            ended_at=ended_at,
            status=status,
            results=res,
            execution_meta=meta,
        )

        # Update internal history
        self._history.append(run)

        # Directly return phase results
        return res

    def run_group(
        self,
        group: PhaseGroup,
        **kwargs,
    ) -> PhaseGroupResults:
        """
        Execute all phases in a PhaseGroup.

        Description:
            The provided PhaseGroup will be executed regardless
            of whether it is registered to this Experiment (`execution_plan`),
            and its outputs will be recorded under `history`.
            **This will mutate the experiment state**. To run a group without
            mutating the experiment state, use `preview_group(...)`.

        Args:
            group (PhaseGroup):
                The PhaseGroup to execute.
            **kwargs:
                Display flags forwarded to each phase's run method.

        Returns:
            PhaseGroupResults:
                Results of the executed phase group.

        """
        # Initiallize run attributes
        started_at = datetime.now()
        status = "completed"

        # Run group and record phase-level meta data
        try:
            res, meta = self._execute_group_with_meta(
                group=group,
                **kwargs,
            )
        except Exception:
            status = "failed"
            raise
        finally:
            ended_at = datetime.now()

        # Construct experiment
        run = ExperimentRun(
            label=group.label,
            started_at=started_at,
            ended_at=ended_at,
            status=status,
            results=res,
            execution_meta=meta,
        )

        # Update internal history
        self._history.append(run)

        # Directly return group results
        return res

    def run(self, **kwargs) -> PhaseGroupResults:
        """
        Run the registered execution plan.

        Description:
            All phases and phase groups added to this experiment
            will be executed in the order they were added.
            Execution history can be viewed via the `history` attribute.

        Args:
            **kwargs:
                Additional arguments to be passed to each executed phase.

        Returns:
            PhaseGroupResults:
                Results of all executed phases.

        """
        return self.run_group(group=self._exec_plan, **kwargs)

    # Preview API
    @overload
    def preview_phase(
        self,
        phase: TrainPhase,
        *,
        show_sampler_progress: bool = True,
        show_training_progress: bool = True,
        persist_progress: bool = IN_NOTEBOOK,
        persist_epoch_progress: bool = IN_NOTEBOOK,
    ) -> TrainResults: ...
    @overload
    def preview_phase(
        self,
        phase: EvalPhase,
        *,
        show_eval_progress: bool = False,
        persist_progress: bool = IN_NOTEBOOK,
    ) -> EvalResults: ...
    def preview_phase(
        self,
        phase: ExperimentPhase,
        **kwargs,
    ) -> PhaseResults:
        """
        Executes a given phase without mutating the Experiment state.

        Description:
            The provided ExperimentPhase will be executed on the current
            experiment state. Any state changes are reverted after the phase
            is executed. Execution is not recorded in `history`.
            To run a phase with history tracking, use `run_phase(...)`.

        Args:
            phase (ExperimentPhase):
                The phase to run.
            **kwargs:
                Display flags forwarded to the phase-specific run method.

        Returns:
            PhaseResults: Phase results.

        """
        # Get initial state
        state = self.get_state()

        # Execute phase
        res, _ = self._execute_phase_with_meta(
            phase=phase,
            **kwargs,
        )

        # Restore experiment state
        self.set_state(state=state)

        return res

    def preview_group(
        self,
        group: PhaseGroup,
        **kwargs,
    ) -> PhaseGroupResults:
        """
        Executes a given phase group without mutating the Experiment state.

        Description:
            The provided PhaseGroup will be executed on the current
            experiment state. Any state changes are reverted after the group
            is executed. Execution is not recorded in `history`.
            To run a group with history tracking, use `run_group(...)`.

        Args:
            group (PhaseGroup):
                The phase group to run.
            **kwargs:
                Display flags forwarded to the phase-specific run method.

        Returns:
            PhaseGroupResults: Phase group results.

        """
        # Get initial state
        state = self.get_state()

        # Execute group
        res, _ = self._execute_group_with_meta(
            group=group,
            **kwargs,
        )

        # Restore experiment state
        self.set_state(state=state)

        return res

    # ================================================
    # Configurable
    # ================================================
    def get_config(self) -> dict[str, Any]:
        """
        Retrieve the configuration details for this experiment.

        This does not contain state information of the underlying model graph.
        """
        return {
            "label": self.label,
            "registration_policy": self._ctx._policy.value,
            "execution_plan": self._exec_plan.get_config(),
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Experiment:
        """
        Reconstructs an Experiment from configuration details.

        This does not restore state information.
        """
        active_ctx = ExperimentContext.get_active()
        exp = cls(
            label=config["label"],
            registration_policy=config.get("registration_policy"),
            ctx=active_ctx,
        )

        # Rebuild execution plan
        exec_plan_cfg = config.get("execution_plan")
        if exec_plan_cfg is not None:
            exp._exec_plan = PhaseGroup.from_config(exec_plan_cfg)
        return exp

    # ================================================
    # Stateful
    # ================================================
    def get_state(self) -> dict[str, Any]:
        return {
            "ctx": self.ctx.get_state(),
            "history": deepcopy(self._history),
            "checkpoints": deepcopy(self._checkpoints),
        }

    def set_state(self, state: dict[str, Any]) -> None:
        # Restore context state
        self._ctx.set_state(state["ctx"])

        # Restore history
        self._history = state.get("history", [])

        # Restore recorded checkpoints
        self._checkpoints = state.get("checkpoints", {})
