from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from modularml.context.experiment_context import ExperimentContext, RegistrationPolicy
from modularml.core.experiment.eval_phase import EvalPhase
from modularml.core.experiment.phase import ExperimentPhase, InputBinding
from modularml.core.experiment.train_phase import BatchSchedulingPolicy, TrainPhase

if TYPE_CHECKING:
    from modularml.core.topology.graph_node import GraphNode
    from modularml.core.topology.model_graph import ModelGraph
    from modularml.core.training.applied_loss import AppliedLoss


@dataclass
class ExperimentRunState:
    last_completed_phase: int = -1  # index into self._phases
    checkpoint_paths: dict[int, str] | None = None


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

        # Phase registry (ordered)
        self._phases: list[ExperimentPhase] = []
        self._run_state = ExperimentRunState()

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
    def phases(self) -> list[ExperimentPhase]:
        """All phases in execution order."""
        return list(self._phases)

    @property
    def train_phases(self) -> list[TrainPhase]:
        """All training phases in execution order."""
        return [p for p in self._phases if isinstance(p, TrainPhase)]

    @property
    def eval_phases(self) -> list[EvalPhase]:
        """All evaluation phases in execution order."""
        return [p for p in self._phases if isinstance(p, EvalPhase)]

    # ================================================
    # Phase Definition API
    # ================================================
    def add_phase(self, phase: ExperimentPhase) -> Experiment:
        """
        Register a phase with this experiment.

        Description:
            Registers a new phase to be run under this experiment.
            Phases are executed in the order they are added.

        Args:
            phase (ExperimentPhase):
                A fully constructed TrainPhase or EvalPhase.

        Returns:
            Experiment: self

        """
        if not isinstance(phase, ExperimentPhase):
            msg = f"Expected ExperimentPhase, got {type(phase)}."
            raise TypeError(msg)

        # Enforce unique phase labels
        if phase.label in {p.label for p in self._phases}:
            msg = f"Phase label '{phase.label}' already exists in Experiment."
            raise ValueError(msg)

        self._phases.append(phase)
        return self

    def add_train_phase(
        self,
        label: str,
        *,
        input_sources: list[InputBinding],
        losses: list[AppliedLoss],
        n_epochs: int = 1,
        active_nodes: list[GraphNode] | None = None,
        batch_schedule: BatchSchedulingPolicy | str = BatchSchedulingPolicy.ZIP_STRICT,
        show_sampler_progress: bool = True,
        show_training_progress: bool = True,
    ) -> Experiment:
        """
        Constructs and registers a new training phase.

        Args:
            label (str):
                A label to assign to this training phase. Must be unique to already
                registered phases.

            input_sources (list[InputBinding]):
                Input bindings for each head node in ModelGraph.

            losses (list[AppliedLoss]):
                A list of losses to be applied during this training pahse.

            n_epochs (int):
                Number of epochs to perform.

            active_nodes (list[str  |  GraphNode] | None, optional):
                A list of active GraphNodes in this phase of the experiment. Nodes can
                be listed via their ID, label, or with the actual node instance. If
                None, all nodes comprising the ModelGraph are used. Defaults to None.

            batch_schedule (str | BatchSchedulingPolicy, optional):
                Defines how batches from multiple samplers are scheduled during training.
                This is only relevant if more than one sampler is defined in
                `input_sources`.

                Let samplers `S1` and `S2` produce: `S1 = [b1, b2, b3]` and  `S2 = [c1, c2]`

                The outputs of each policy is given below:

                ```python
                "zip_strict": (b1, c1), (b2, c2)
                "zip_cycle": (b1, c1), (b2, c2), (b3, c1)
                "alternate_strict": b1, c1, b2, c2
                "alternate_cycle": b1, c1, b2, c2, b3, c1
                ```

                See also :class:`BatchSchedulingPolicy`.

            show_sampler_progress (bool, optional):
                Whether to show a progress bar for sampler batching. Defaults to True.

            show_training_progress (bool, optional):
                Whether to show a progress bar for training execution. Defaults to True.

        Returns:
            Experiment: self

        """
        phase = TrainPhase(
            label=label,
            input_sources=input_sources,
            losses=losses,
            n_epochs=n_epochs,
            active_nodes=active_nodes,
            batch_schedule=batch_schedule,
            show_sampler_progress=show_sampler_progress,
            show_training_progress=show_training_progress,
        )
        return self.add_phase(phase=phase)

    def add_eval_phase(
        self,
        label: str,
        input_sources: list[InputBinding],
        active_nodes: list[GraphNode] | None = None,
    ) -> Experiment:
        """
        Constructs and registers a new evaluation phase.

        Args:
            label (str):
                A label to assign to this evaluation phase. Must be unique to already
                registered phases.

            input_sources (list[InputBinding]):
                Input bindings for each head node in ModelGraph.

            active_nodes (list[GraphNode] | None, optional):
                A list of active GraphNodes in this phase of the experiment. Nodes can
                be listed via their ID, label, or with the actual node instance. If
                None, all nodes comprising the ModelGraph are used. Defaults to None.

        Returns:
            Experiment: self

        """
        phase = EvalPhase(
            label=label,
            input_sources=input_sources,
            active_nodes=active_nodes,
        )
        return self.add_phase(phase=phase)

    def remove_phase(self, phase: str | int) -> Experiment:
        """
        Remove a phase from the experiment.

        Description:
            Removes a registered phase by index or label. If checkpoints exist for
            the removed phase or any later phases, those checkpoints are deleted
            and the experiment run state is rewound accordingly.

        Args:
            phase (str | int):
                Phase label or index to remove.

        Returns:
            Experiment: self

        """
        # Resolve phase index
        idx = self._resolve_phase_index(phase)

        # Remove checkpoints at / after idx
        if self._run_state.checkpoint_paths is not None:
            to_delete = [
                p_idx for p_idx in self._run_state.checkpoint_paths if p_idx >= idx
            ]
            for p_idx in to_delete:
                ckpt_path = Path(self._run_state.checkpoint_paths[p_idx])
                if ckpt_path.exists():
                    ckpt_path.unlink()
                del self._run_state.checkpoint_paths[p_idx]

            # Reset last_completed
            last_idx = sorted(self._run_state.checkpoint_paths)[-1]
            self._run_state.last_completed_phase = last_idx

            # Reload from last checkpoint before phase being removed
            self._load_latest_checkpoint()

        # Remove phase
        _ = self._phases.pop(idx)

        return self

    # ================================================
    # Phase Execution API
    # ================================================
    def _load_latest_checkpoint(self):
        """Reload last saved checkpoint."""
        if self.model_graph is None:
            raise RuntimeError("Checkpoint load failed. ModelGraph is not defined.")

        # Pick the latest completed phase
        if self._run_state.checkpoint_paths is None:
            raise RuntimeError("No checkpoints available to resume from.")
        completed = sorted(self._run_state.checkpoint_paths.keys())
        last_phase_idx = completed[-1]

        # Reload checkpoint
        self.model_graph.load_checkpoint(
            self._run_state.checkpoint_paths[last_phase_idx],
        )

    def _save_checkpoint(
        self,
        checkpoint_dir: str | Path,
        phase_idx: int,
    ) -> Path:
        """
        Save checkpoint to path.

        Args:
            checkpoint_dir (str | Path):
                Directory to save checkpoint to.

            phase_idx (int):
                The index of the phase being saved.

        Returns:
            Path: The final filepath of the saved checkpoint.

        """
        if self.model_graph is None:
            raise RuntimeError("Checkpoint save failed. ModelGraph is not defined.")

        # Create directory
        dir_save = Path(checkpoint_dir)
        dir_save.mkdir(exist_ok=True, parents=True)

        # File extension will be enforced during ModelGraph.save_checkpoint
        filepath = dir_save / f"{self.label}_phase{phase_idx}"
        save_path = self.model_graph.save_checkpoint(filepath)

        # Store path for later reload
        self._run_state.checkpoint_paths[phase_idx] = str(save_path)

        return save_path

    def _resolve_phase_index(self, phase: str | int) -> int:
        if isinstance(phase, int):
            if phase < 0 or phase >= len(self._phases):
                msg = f"Phase index out of range: {phase}"
                raise IndexError(msg)
            return phase

        if isinstance(phase, str):
            for i, p in enumerate(self._phases):
                if p.label == phase:
                    return i
            msg = f"No phase found with label '{phase}'"
            raise ValueError(msg)

        msg = f"Invalid phase identifier: {type(phase)}"
        raise TypeError(msg)

    def _run_phase(self, phase: ExperimentPhase):
        raise NotImplementedError

    def run(
        self,
        *,
        from_phase: str | int | None = None,
        until_phase: str | int | None = None,
        resume: bool = False,
        checkpoint_dir: str | None = None,
    ):
        """
        Executes registered phases.

        Args:
            from_phase (str | int, optional):
                The phase to start execution from. Can specify either the index of the
                phase in all registered phases, or the phase name. If None, execution
                is started from the first registered phase. Defaults to None.

            until_phase (str | int, optional):
                The last phase to execute. Can specify either the index of the
                phase in all registered phases, or the phase name. If None, execution
                is run through the end of all registered phases. Defaults to None.

            resume (bool, optional):
                Whether to resume from the last saved checkpoint or restart.
                Defaults to False.

            checkpoint_dir (str, optional):
                If provided, checkpoints will be saved every time a full phase is
                executed. Defaults to None.

        """
        if not self._phases:
            raise RuntimeError("Experiment has no phases to run.")

        # Resolve phase bounds
        if resume:
            self._load_latest_checkpoint()
            start_idx = self._run_state.last_completed_phase + 1
        else:
            start_idx = (
                self._resolve_phase_index(from_phase) if from_phase is not None else 0
            )
        end_idx = (
            self._resolve_phase_index(until_phase)
            if until_phase is not None
            else len(self._phases) - 1
        )
        if start_idx > end_idx:
            msg = f"Invalid execution range: start={start_idx}, end={end_idx}"
            raise ValueError(msg)

        # Execute phases
        for idx in range(start_idx, end_idx + 1):
            phase = self._phases[idx]

            self._run_phase(phase=phase)

            self._run_state.last_completed_phase = idx

            if checkpoint_dir is not None:
                ckpt_path = self._save_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    phase_idx=idx,
                )
                self._run_state.checkpoint_paths[idx] = ckpt_path
