"""Core Experiment orchestration and execution utilities."""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, overload

from modularml.core.experiment.callbacks.experiment_callback import (
    ExperimentCallback,
)
from modularml.core.experiment.checkpointing import (
    EXPERIMENT_HOOKS,
    EXPERIMENT_NAME_TEMPLATE,
    EXPERIMENT_PLACEHOLDERS,
    Checkpointing,
)
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
from modularml.core.io.checkpoint import Checkpoint
from modularml.utils.environment.environment import IN_NOTEBOOK
from modularml.utils.logging.logger import get_logger
from modularml.utils.logging.warnings import warn

if TYPE_CHECKING:
    from modularml.core.experiment.results.phase_results import PhaseResults
    from modularml.core.topology.model_graph import ModelGraph

logger = get_logger(name="Experiment")


class Experiment:
    """High-level container coordinating phases, callbacks, and checkpoints."""

    def __init__(
        self,
        label: str,
        registration_policy: RegistrationPolicy | str | None = None,
        ctx: ExperimentContext | None = None,
        checkpointing: Checkpointing | None = None,
        callbacks: list[ExperimentCallback] | None = None,
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

            checkpointing (Checkpointing | None, optional):
                An optional Checkpointing configuration for automatically saving
                the full experiment state to disk at execution lifecycle hooks
                (e.g. `phase_end`, `group_end`). Must use `mode="disk"`
                and `save_on` hooks from: `phase_start`, `phase_end`,
                `group_start`, `group_end`. Defaults to None.

            callbacks (list[ExperimentCallback] | None, optional):
                An optional list of experiment-level callbacks to run during
                `Experiment.run()` execution at phase/group boundaries.
                Defaults to None.

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

        # For recording execution history
        self._history: list[ExperimentRun] = []

        # For checkpointing model graph state
        self._checkpoints: dict[str, Path] = {}
        self._checkpoint_dir: Path | None = None

        # Experiment-level checkpointing
        self._exp_checkpointing: Checkpointing | None = None
        self.set_checkpointing(checkpointing)

        # Experiment-level callbacks
        self._exp_callbacks: list[ExperimentCallback] = list(callbacks or [])
        self._exp_callbacks.sort(key=lambda cb: cb._exec_order)

        # Bool flags for guarding
        # True while executing inside a callback
        self._in_callback: bool = False
        # True disables all checkpointin (experiment-level and TrainPhase-level)
        self._checkpointing_disabled: bool = False

    # ================================================
    # Constructors
    # ================================================
    @classmethod
    def from_active_context(
        cls,
        label: str,
        registration_policy: RegistrationPolicy | str | None = None,
        checkpointing: Checkpointing | None = None,
        callbacks: list[ExperimentCallback] | None = None,
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

            checkpointing (Checkpointing | None, optional):
                An optional Checkpointing configuration for automatically saving
                the full experiment state to disk at execution lifecycle hooks
                (e.g. `phase_end`, `group_end`). Must use `mode="disk"`
                and `save_on` hooks from: `phase_start`, `phase_end`,
                `group_start`, `group_end`. Defaults to None.

            callbacks (list[ExperimentCallback] | None, optional):
                An optional list of experiment-level callbacks to run during
                `Experiment.run()` execution at phase/group boundaries.
                Defaults to None.

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
            checkpointing=checkpointing,
            callbacks=callbacks,
        )

    # ================================================
    # Properties
    # ================================================
    @property
    def ctx(self) -> ExperimentContext:
        """Gets the context associated with this Experiment."""
        return self._ctx

    @property
    def model_graph(self) -> ModelGraph | None:
        """Gets the ModelGraph associated with this Experiment."""
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

    @property
    def checkpointing(self) -> Checkpointing | None:
        """The experiment-level Checkpointing configuration, or None."""
        return self._exp_checkpointing

    @property
    def available_checkpoints(self) -> dict[str, Path]:
        """All available disk checkpoints (from both TrainPhase and Experiment)."""
        return dict(self._checkpoints)

    @property
    def exp_callbacks(self) -> list[ExperimentCallback]:
        """Experiment-level callbacks in execution order."""
        return list(self._exp_callbacks)

    # ================================================
    # Experiment Callback Management
    # ================================================
    def add_callback(self, callback: ExperimentCallback) -> None:
        """
        Register an experiment-level callback.

        Args:
            callback (ExperimentCallback):
                The callback to add.

        """
        if not isinstance(callback, ExperimentCallback):
            msg = f"Expected ExperimentCallback, got {type(callback)}."
            raise TypeError(msg)
        self._exp_callbacks.append(callback)
        self._exp_callbacks.sort(key=lambda cb: cb._exec_order)

    @contextmanager
    def disable_checkpointing(self):
        """
        Context manager that disables all checkpointing while active.

        Description:
            Suppresses both experiment-level checkpointing and any
            TrainPhase-level checkpointing that occurs within the block.
            The previous checkpointing configuration is restored on exit,
            even if an exception is raised.

        Examples:
        ```python
            with experiment.disable_checkpointing():
                experiment.run_phase(training_phase)
        ```

        """
        prev = self._checkpointing_disabled
        self._checkpointing_disabled = True
        try:
            yield
        finally:
            self._checkpointing_disabled = prev

    # ================================================
    # Checkpointing
    # ================================================
    def set_checkpointing(self, checkpointing: Checkpointing | None) -> None:
        """
        Attach or replace the Checkpointing configuration for this experiment.

        Validates that the mode is `"disk"`, that all `save_on` hooks are
        valid for an Experiment, and that the `name_template` only uses
        allowed placeholders. If no `name_template` is set, the experiment
        default is applied.

        Args:
            checkpointing (Checkpointing | None):
                The Checkpointing configuration, or None to disable.

        """
        if checkpointing is None:
            self._exp_checkpointing = None
            return

        # Experiment only supports disk mode
        if checkpointing.mode != "disk":
            msg = (
                "Experiment-level checkpointing only supports mode='disk'. "
                "In-memory checkpointing of the full experiment state is "
                "not supported due to memory overhead."
            )
            raise ValueError(msg)

        # Validate hooks
        invalid = set(checkpointing.save_on) - EXPERIMENT_HOOKS
        if invalid:
            msg = (
                f"Invalid save_on hooks for Experiment: {sorted(invalid)}. "
                f"Valid hooks: {sorted(EXPERIMENT_HOOKS)}."
            )
            raise ValueError(msg)

        # Apply default template if not set
        if checkpointing.name_template is None:
            checkpointing.name_template = EXPERIMENT_NAME_TEMPLATE

        # Validate placeholders
        Checkpointing.validate_placeholders(
            checkpointing.name_template,
            EXPERIMENT_PLACEHOLDERS,
            context_name="Experiment",
        )

        self._exp_checkpointing = checkpointing

        # Eagerly set experiment checkpoint directory from the config
        if checkpointing.directory is not None and self._checkpoint_dir is None:
            self.set_checkpoint_dir(checkpointing.directory, create=True)

    def set_checkpoint_dir(self, path: Path, *, create: bool = True):
        """
        Set directory used for storing experiment checkpoints.

        Args:
            path (Path):
                Directory path.
            create (bool, optional):
                Whether to create directory if it does not exist.

        """
        path = Path(path)
        if create:
            path.mkdir(parents=True, exist_ok=True)
        if not path.is_dir():
            msg = f"No directory exists at '{path!r}'."
            raise FileExistsError(msg)

        # Warn if directory already contains checkpoint files
        existing = list(path.glob("*.ckpt.mml"))
        if existing:
            warn(
                f"Checkpoint directory '{path}' already contains "
                f"{len(existing)} checkpoint file(s). Existing files will "
                f"only be overwritten if an exact name match occurs and "
                f"overwrite=True.",
                stacklevel=2,
            )

        self._checkpoint_dir = path

    def save_checkpoint(
        self,
        name: str,
        *,
        overwrite: bool = False,
        meta: dict[str, Any] | None = None,
    ) -> Path:
        """
        Save full experiment state to a Checkpoint file.

        Creates a :class:`Checkpoint` container with the full experiment
        state and serializes it to disk.

        Args:
            name (str):
                Unique name to assign to this checkpoint.
            overwrite (bool, optional):
                Whether to overwrite existing checkpoints with this name.
                Defaults to False.
            meta (dict[str, Any], optional):
                Additional meta data to attach to the checkpoint.

        Returns:
            Path: The saved checkpoint file path.

        """
        from modularml.core.io.serialization_policy import SerializationPolicy
        from modularml.core.io.serializer import serializer

        if self._checkpoint_dir is None:
            msg = (
                "Checkpoint directory not set. Call `set_checkpoint_dir()` "
                "or set `directory` on the Checkpointing config."
            )
            raise RuntimeError(msg)

        if name in self._checkpoints and not overwrite:
            msg = f"Checkpoint '{name}' already exists."
            raise ValueError(msg)

        filepath = self._checkpoint_dir / name

        ckpt = Checkpoint()
        ckpt.add_entry(key="experiment", obj=self)

        if meta is not None:
            for k, v in meta.items():
                ckpt.add_meta(k, v)

        save_path = serializer.save(
            ckpt,
            filepath,
            policy=SerializationPolicy.BUILTIN,
            overwrite=overwrite,
        )
        self._checkpoints[name] = Path(save_path)
        return Path(save_path)

    def restore_checkpoint(self, name_or_path: str | Path) -> None:
        """
        Restore state from a previously saved checkpoint.

        Description:
            Accepts either a checkpoint name (a key from
            :attr:`available_checkpoints`) or an explicit file path. The
            checkpoint type is detected automatically:

            - **Experiment checkpoint** (contains an ``"experiment"`` entry):
              restores the full experiment state via :meth:`set_state`.
            - **ModelGraph checkpoint** (contains a ``"modelgraph"`` entry):
              restores the model graph state via
              :meth:`ModelGraph.restore_checkpoint`.

        Args:
            name_or_path (str | Path):
                Either a checkpoint name registered in
                :attr:`available_checkpoints`, or the file path to a
                ``.ckpt.mml`` checkpoint file.

        Raises:
            ValueError: If ``name_or_path`` is not a registered name and
                does not point to an existing file.
            TypeError: If the loaded checkpoint contains neither an
                ``"experiment"`` nor a ``"modelgraph"`` entry.

        """
        from modularml.core.io.serializer import _enforce_file_suffix, serializer

        # Resolve to filepath
        name_or_path_str = str(name_or_path)
        if name_or_path_str in self._checkpoints:
            filepath = self._checkpoints[name_or_path_str]
        else:
            filepath = Path(name_or_path)

        filepath = Path(filepath)
        if filepath.suffix == "":
            filepath = _enforce_file_suffix(path=filepath, cls=Checkpoint)

        if not filepath.exists():
            msg = (
                f"No checkpoint named '{name_or_path}' exists and no file "
                f"found at '{filepath}'. "
                f"Available checkpoints: {list(self._checkpoints.keys())}."
            )
            raise ValueError(msg)

        ckpt: Checkpoint = serializer.load(filepath)

        # Auto-detect checkpoint type and restore
        if "experiment" in ckpt.entries:
            self.set_state(ckpt.entries["experiment"].entry_state)
        elif "modelgraph" in ckpt.entries:
            self.model_graph.restore_checkpoint(filepath)
        else:
            msg = (
                f"Checkpoint at '{filepath}' does not contain a recognized "
                f"entry. Expected 'experiment' or 'modelgraph' key, "
                f"found: {list(ckpt.entries.keys())}."
            )
            raise TypeError(msg)

    def _save_experiment_checkpoint(self, label: str) -> None:
        """
        Save the full experiment to disk using the checkpointing config.

        Args:
            label (str):
                The phase or group label to use as the checkpoint key.

        """
        ckpt = self._exp_checkpointing
        name = ckpt.format_name(label=label)

        # Ensure checkpoint directory
        if self._checkpoint_dir is None:
            if ckpt.directory is not None:
                self.set_checkpoint_dir(ckpt.directory, create=True)
            else:
                msg = "Cannot save experiment checkpoint: no checkpoint directory set."
                raise RuntimeError(msg)

        save_path = self.save_checkpoint(
            name=name,
            overwrite=ckpt.overwrite,
        )
        ckpt.record_disk(key=label, path=save_path)

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
        val_loss_metric: str = "val_loss",
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
            val_loss_metric (str, optional):
                The name of a recorded ValidationLossMetrics to show in the progress
                bar. Results must be tracked, and `val_loss_metric` must be an existing
                loss metric. Otherwise, no val_loss field will be shown in the progress
                bar. Defaults to `"val_loss"`.

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
            val_loss_metric=val_loss_metric,
        ):
            self.model_graph.train_step(
                ctx=ctx,
                losses=phase.losses,
                active_nodes=phase.active_nodes,
            )
            res.add_execution_context(ctx=ctx)

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
        # ------------------------------------------------
        # Propagate checkpoint directory to TrainPhase if needed
        # ------------------------------------------------
        if (
            isinstance(phase, TrainPhase)
            and phase.checkpointing is not None
            and phase.checkpointing.mode == "disk"
            and phase.checkpointing.directory is None
        ):
            exp_dir = self._checkpoint_dir
            if exp_dir is None and self._exp_checkpointing is not None:
                exp_dir = self._exp_checkpointing.directory
            if exp_dir is not None:
                phase_dir = exp_dir / phase.label
                phase_dir.mkdir(parents=True, exist_ok=True)
                phase.checkpointing._directory = phase_dir

        # Skip callbacks and checkpointing when inside a callback
        run_hooks = not self._in_callback
        run_ckpt = run_hooks and not self._checkpointing_disabled

        # ------------------------------------------------
        # on_phase_start
        # - Run experiment callback
        # - Run experiment checkpointing
        # ------------------------------------------------
        if run_hooks:
            self._in_callback = True
            try:
                for cb in self._exp_callbacks:
                    cb._on_phase_start(experiment=self, phase=phase)
            finally:
                self._in_callback = False

        if (
            run_ckpt
            and (self._exp_checkpointing is not None)
            and (self._exp_checkpointing.should_save("phase_start"))
        ):
            self._save_experiment_checkpoint(label=phase.label)

        # ------------------------------------------------
        # run phase
        # - modifies experiment state but does not update history
        # ------------------------------------------------
        phase_start = datetime.now()
        if isinstance(phase, TrainPhase):
            train_keys = {
                "show_sampler_progress",
                "show_training_progress",
                "persist_progress",
                "persist_epoch_progress",
                "val_loss_metric",
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

        # ------------------------------------------------
        # on_phase_end
        # - Run experiment callback
        # - Run experiment checkpointing
        # ------------------------------------------------
        if run_hooks:
            self._in_callback = True
            try:
                for cb in self._exp_callbacks:
                    cb.on_phase_end(
                        experiment=self,
                        phase=phase,
                        results=phase_res,
                    )
            finally:
                self._in_callback = False

        if (
            run_ckpt
            and (self._exp_checkpointing is not None)
            and self._exp_checkpointing.should_save("phase_end")
        ):
            self._save_experiment_checkpoint(label=phase.label)

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

        # Skip callbacks and checkpointing when inside a callback
        run_hooks = not self._in_callback
        run_ckpt = run_hooks and not self._checkpointing_disabled

        # ------------------------------------------------
        # on_group_start
        # - Run experiment callback
        # - Run experiment checkpointing
        # ------------------------------------------------
        if run_hooks:
            self._in_callback = True
            try:
                for cb in self._exp_callbacks:
                    cb.on_group_start(experiment=self, group=group)
            finally:
                self._in_callback = False

        if (
            run_ckpt
            and (self._exp_checkpointing is not None)
            and self._exp_checkpointing.should_save("group_start")
        ):
            self._save_experiment_checkpoint(label=group.label)

        # ------------------------------------------------
        # run phase group
        # - construct result container
        # - run each phase in order
        # ------------------------------------------------
        group_results = PhaseGroupResults(label=group.label)
        group_meta = PhaseGroupExecutionMeta(
            label=group.label,
            started_at=datetime.now(),
            ended_at=None,
        )
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

        # ------------------------------------------------
        # on_group_end
        # - Run experiment callback
        # - Run experiment checkpointing
        # ------------------------------------------------
        if run_hooks:
            self._in_callback = True
            try:
                for cb in self._exp_callbacks:
                    cb.on_group_end(
                        experiment=self,
                        group=group,
                        results=group_results,
                    )
            finally:
                self._in_callback = False

        if (
            run_ckpt
            and (self._exp_checkpointing is not None)
            and self._exp_checkpointing.should_save("group_end")
        ):
            self._save_experiment_checkpoint(label=group.label)

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
        val_loss_metric: str = "val_loss",
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
        Execute a single phase and record the results.

        Description:
            The provided :class:`ExperimentPhase` runs regardless of whether it
            is registered on :attr:`execution_plan`, and its outputs are stored
            under :attr:`history`. This mutates experiment state. To run a phase
            without mutating state, use :meth:`preview_phase`.

        Args:
            phase (ExperimentPhase):
                Phase to run.
            **kwargs (Any):
                Display flags forwarded to the phase-specific run method.

        Returns:
            PhaseResults: Results produced by the executed phase.

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
        # ------------------------------------------------
        # on_experiment_start
        # - Run experiment callback
        # - Run experiment checkpointing
        # ------------------------------------------------
        for cb in self._exp_callbacks:
            cb.on_experiment_start(experiment=self)
        if (
            self._exp_checkpointing is not None
        ) and self._exp_checkpointing.should_save("experiment_start"):
            self._save_experiment_checkpoint(label="START")

        # ------------------------------------------------
        # run all phases
        # - callback/checkpointing logic handled internally
        # ------------------------------------------------
        try:
            res = self.run_group(group=self._exec_plan, **kwargs)
        except BaseException as exc:
            self._in_callback = True
            try:
                for cb in self._exp_callbacks:
                    cb._on_exception(
                        experiment=self,
                        phase=None,
                        exception=exc,
                    )
            finally:
                self._in_callback = False
            raise

        # ------------------------------------------------
        # on_experiment_end
        # - Run experiment callback
        # - Run experiment checkpointing
        # ------------------------------------------------
        for cb in self._exp_callbacks:
            cb.on_experiment_end(experiment=self)
        if (
            self._exp_checkpointing is not None
        ) and self._exp_checkpointing.should_save("experiment_end"):
            self._save_experiment_checkpoint(label="END")

        # Return results
        return res

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
        val_loss_metric: str = "val_loss",
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
        Execute a phase without mutating the Experiment state.

        Description:
            The provided :class:`ExperimentPhase` runs against the current
            experiment state and any changes are reverted afterward. Execution
            is not recorded in :attr:`history`. Use :meth:`run_phase` to persist
            results.

        Args:
            phase (ExperimentPhase):
                Phase to run.
            **kwargs (Any):
                Display flags forwarded to the phase-specific run method.

        Returns:
            PhaseResults: Results produced by the previewed phase.

        """
        # Get initial state
        state = self.get_state()

        # Execute phase with checkpointing disabled
        with self.disable_checkpointing():
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

        # Execute group with checkpointing disabled
        with self.disable_checkpointing():
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

        Args:
            config (dict[str, Any]):
                Configuration payload returned by :meth:`get_config`.

        Returns:
            Experiment: Newly constructed experiment bound to the active context.

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
        """Return a deep copy of mutable experiment state."""
        return {
            "ctx": self.ctx.get_state(),
            "history": deepcopy(self._history),
            "checkpoints": deepcopy(self._checkpoints),
        }

    def set_state(self, state: dict[str, Any]) -> None:
        """
        Restore experiment state from :meth:`get_state` output.

        Args:
            state (dict[str, Any]): Serialized snapshot captured by :meth:`get_state`.

        """
        # Restore context state
        self._ctx.set_state(state["ctx"])

        # Restore history
        self._history = state.get("history", [])

        # Restore recorded checkpoints
        self._checkpoints = state.get("checkpoints", {})

    # ================================================
    # Serialization
    # ================================================
    def save(self, filepath: Path, *, overwrite: bool = False) -> Path:
        """
        Serializes this experiment to the specified filepath.

        Args:
            filepath (Path):
                File location to save to. Note that the suffix may be overwritten
                to enforce the ModularML file extension schema.
            overwrite (bool, optional):
                Whether to overwrite any existing file at the save location.
                Defaults to False.

        Returns:
            Path: The actual filepath at which the experiment was saved.

        """
        from modularml.core.io.serialization_policy import SerializationPolicy
        from modularml.core.io.serializer import serializer

        return serializer.save(
            self,
            filepath,
            policy=SerializationPolicy.BUILTIN,
            overwrite=overwrite,
        )

    @classmethod
    def load(
        cls,
        filepath: Path,
        *,
        checkpoint_dir: Path | None = None,
        allow_packaged_code: bool = False,
        overwrite: bool = False,
    ) -> Experiment:
        """
        Load an Experiment from file.

        Args:
            filepath (Path):
                File location of a previously saved Experiment.
            checkpoint_dir (Path | None, optional):
                Directory to extract saved checkpoints into. If the
                serialized experiment contains checkpoint artifacts and
                this is None, the checkpoints will not be restored and
                a warning will be emitted. Defaults to None.
            allow_packaged_code : bool
                Whether bundled code execution is allowed.
            overwrite (bool):
                Whether to replace any colliding node registrations in ExperimentContext
                If False, new IDs are assigned to the reloaded nodes comprising the
                graph. Otherwise, any collision are overwritten with the saved nodes.
                Defaults to False.
                It is recommended to only reload an Experiment into a new/empty
                `ExperimentContext`.

        Returns:
            Experiment: The reloaded Experiment.

        """
        from modularml.core.io.serializer import _enforce_file_suffix, serializer

        # Append proper suffix only if no suffix is given
        if Path(filepath).suffix == "":
            filepath = _enforce_file_suffix(path=filepath, cls=cls)

        return serializer.load(
            filepath,
            allow_packaged_code=allow_packaged_code,
            overwrite=overwrite,
            extras={"checkpoint_dir": checkpoint_dir}
            if checkpoint_dir is not None
            else None,
        )
