from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

from modularml.core.experiment.callback_result import CallbackResult, PayloadResult
from modularml.utils.nn.training import preserve_frozen_state

if TYPE_CHECKING:
    from modularml.core.data.execution_context import ExecutionContext
    from modularml.core.experiment.experiment import Experiment
    from modularml.core.experiment.phases.phase import ExperimentPhase
    from modularml.core.experiment.results.phase_result import PhaseResults


class Callback(ABC):
    """
    Base class for ExperimentPhase callbacks with lifecycle hooks.

    Description:
        Callbacks provide a structured way to inject behavior at common lifecycle
        points during a phase execution (phase/epoch/batch boundaries and exception
        handling).

        Subclasses should override only the public hooks (`on_*`) methods they need and
        implement :meth:`get_config` / :meth:`from_config` for serialization. Do not
        overwrite the private (`_on_*`) methods.

    """

    version: ClassVar[str] = "1.0"

    def __init__(self, label: str | None = None) -> None:
        """
        Initialize a new callback instance.

        Args:
            label (str | None):
                Stable identifier for this callback within a phase results container.
                If None, defaults to the callback class name.

        """
        self._phase: ExperimentPhase | None = None
        self._label = label or self.__class__.__qualname__

    # ================================================
    # Properties
    # ================================================
    @property
    def label(self) -> str:
        """Stable identifier of this callback instance."""
        return self._label

    # ================================================
    # Binding
    # ================================================
    def attach(self, phase: ExperimentPhase):
        """
        Attach this callback to an ExperimentPhase.

        Args:
            phase (ExperimentPhase):
                The phase this callback is attached to.

        """
        self._phase = phase

    @property
    def phase(self) -> ExperimentPhase | None:
        """Return the currently attached phase (if any)."""
        return self._phase

    # ================================================
    # Internal Lifecycle Hooks
    # ================================================
    # Phase lifecycle
    def _on_phase_start(
        self,
        *,
        experiment: Experiment,
        phase: ExperimentPhase,
        results: PhaseResults | None,
    ) -> None:
        """
        Internal hook that wraps callback outputs.

        Description:
            Callback-implemented hooks (eg, `on_phase_start`) can return any object.
            If not None, it is wrapped into a CallbackResult and stored in `result`.

        Args:
            experiment (Experiment):
                The Experiment on which the phase is attached.
                The model graph can be accessed via `experiment.model_graph`.
            phase (ExperimentPhase):
                The current phase.
            results (PhaseResults):
                A result tracker to append callback output to.

        """
        # Preserve trainable state of model graph
        with preserve_frozen_state(experiment.model_graph):
            raw_output: Any = self.on_phase_start(
                experiment=experiment,
                phase=phase,
            )

        if raw_output is None or results is None:
            return

        if isinstance(raw_output, CallbackResult):
            cb_res = raw_output
        else:
            cb_res = PayloadResult(payload=raw_output)
        cb_res.bind_scope(
            callback_label=self.label,
            phase_label=phase.label,
            epoch_idx=None,
            batch_idx=None,
            edge="start",
        )

        results.add_callback_result(cb_res=cb_res)

    def _on_phase_end(
        self,
        *,
        experiment: Experiment,
        phase: ExperimentPhase,
        results: PhaseResults | None,
    ) -> None:
        """
        Internal hook that wraps callback outputs.

        Description:
            Callback-implemented hooks (eg, `on_phase_end`) can return any object.
            If not None, it is wrapped into a CallbackResult and stored in `result`.

        Args:
            experiment (Experiment):
                The Experiment on which the phase is attached.
                The model graph can be accessed via `experiment.model_graph`.
            phase (ExperimentPhase):
                The current phase.
            results (PhaseResults):
                A result tracker to append callback output to.

        """
        # Preserve trainable state of model graph
        with preserve_frozen_state(experiment.model_graph):
            raw_output: Any = self.on_phase_end(
                experiment=experiment,
                phase=phase,
            )
        if raw_output is None or results is None:
            return

        if isinstance(raw_output, CallbackResult):
            cb_res = raw_output
        else:
            cb_res = PayloadResult(payload=raw_output)
        cb_res.bind_scope(
            callback_label=self.label,
            phase_label=phase.label,
            epoch_idx=None,
            batch_idx=None,
            edge="end",
        )

        results.add_callback_result(cb_res=cb_res)

    # Epoch lifecycle
    def _on_epoch_start(
        self,
        *,
        experiment: Experiment,
        phase: ExperimentPhase,
        exec_ctx: ExecutionContext,
        results: PhaseResults | None,
    ) -> None:
        """
        Internal hook that wraps callback outputs.

        Description:
            Callback-implemented hooks (eg, `on_epoch_start`) can return any object.
            If not None, it is wrapped into a CallbackResult and stored in `result`.

        Args:
            experiment (Experiment):
                The Experiment on which the phase is attached.
                The model graph can be accessed via `experiment.model_graph`.
            phase (ExperimentPhase):
                The current phase.
            results (PhaseResults):
                A result tracker to append callback output to.
            exec_ctx (ExecutionContext):
                The first execution context of the epoch.

        """
        # Preserve trainable state of model graph
        with preserve_frozen_state(experiment.model_graph):
            raw_output: Any = self.on_epoch_start(
                experiment=experiment,
                phase=phase,
                exec_ctx=exec_ctx,
            )
        if raw_output is None or results is None:
            return

        if isinstance(raw_output, CallbackResult):
            cb_res = raw_output
        else:
            cb_res = PayloadResult(payload=raw_output)
        cb_res.bind_scope(
            callback_label=self.label,
            phase_label=phase.label,
            epoch_idx=exec_ctx.epoch_idx,
            batch_idx=None,
            edge="start",
        )

        results.add_callback_result(cb_res=cb_res)

    def _on_epoch_end(
        self,
        *,
        experiment: Experiment,
        phase: ExperimentPhase,
        exec_ctx: ExecutionContext,
        results: PhaseResults | None,
    ) -> None:
        """
        Internal hook that wraps callback outputs.

        Description:
            Callback-implemented hooks (eg, `on_epoch_end`) can return any object.
            If not None, it is wrapped into a CallbackResult and stored in `result`.

        Args:
            experiment (Experiment):
                The Experiment on which the phase is attached.
                The model graph can be accessed via `experiment.model_graph`.
            phase (ExperimentPhase):
                The current phase.
            results (PhaseResults):
                A result tracker to append callback output to.
            exec_ctx (ExecutionContext):
                The first execution context of the epoch.

        """
        # Preserve trainable state of model graph
        with preserve_frozen_state(experiment.model_graph):
            raw_output: Any = self.on_epoch_end(
                experiment=experiment,
                phase=phase,
                exec_ctx=exec_ctx,
            )
        if raw_output is None or results is None:
            return

        if isinstance(raw_output, CallbackResult):
            cb_res = raw_output
        else:
            cb_res = PayloadResult(payload=raw_output)
        cb_res.bind_scope(
            callback_label=self.label,
            phase_label=phase.label,
            epoch_idx=exec_ctx.epoch_idx,
            batch_idx=None,
            edge="end",
        )

        results.add_callback_result(cb_res=cb_res)

    # Batch lifecycle
    def _on_batch_start(
        self,
        *,
        experiment: Experiment,
        phase: ExperimentPhase,
        exec_ctx: ExecutionContext,
        results: PhaseResults | None,
    ) -> None:
        """
        Internal hook that wraps callback outputs.

        Description:
            Callback-implemented hooks (eg, `on_batch_start`) can return any object.
            If not None, it is wrapped into a CallbackResult and stored in `result`.

        Args:
            experiment (Experiment):
                The Experiment on which the phase is attached.
                The model graph can be accessed via `experiment.model_graph`.
            phase (ExperimentPhase):
                The current phase.
            results (PhaseResults):
                A result tracker to append callback output to.
            exec_ctx (ExecutionContext):
                The first execution context of the epoch.

        """
        # Preserve trainable state of model graph
        with preserve_frozen_state(experiment.model_graph):
            raw_output: Any = self.on_batch_start(
                experiment=experiment,
                phase=phase,
                exec_ctx=exec_ctx,
            )
        if raw_output is None or results is None:
            return

        if isinstance(raw_output, CallbackResult):
            cb_res = raw_output
        else:
            cb_res = PayloadResult(payload=raw_output)
        cb_res.bind_scope(
            callback_label=self.label,
            phase_label=phase.label,
            epoch_idx=exec_ctx.epoch_idx,
            batch_idx=exec_ctx.batch_idx,
            edge="start",
        )

        results.add_callback_result(cb_res=cb_res)

    def _on_batch_end(
        self,
        *,
        experiment: Experiment,
        phase: ExperimentPhase,
        exec_ctx: ExecutionContext,
        results: PhaseResults | None,
    ) -> None:
        """
        Internal hook that wraps callback outputs.

        Description:
            Callback-implemented hooks (eg, `on_batch_end`) can return any object.
            If not None, it is wrapped into a CallbackResult and stored in `result`.

        Args:
            experiment (Experiment):
                The Experiment on which the phase is attached.
                The model graph can be accessed via `experiment.model_graph`.
            phase (ExperimentPhase):
                The current phase.
            results (PhaseResults):
                A result tracker to append callback output to.
            exec_ctx (ExecutionContext):
                The first execution context of the epoch.

        """
        # Preserve trainable state of model graph
        with preserve_frozen_state(experiment.model_graph):
            raw_output: Any = self.on_batch_end(
                experiment=experiment,
                phase=phase,
                exec_ctx=exec_ctx,
            )
        if raw_output is None or results is None:
            return

        if isinstance(raw_output, CallbackResult):
            cb_res = raw_output
        else:
            cb_res = PayloadResult(payload=raw_output)
        cb_res.bind_scope(
            callback_label=self.label,
            phase_label=phase.label,
            epoch_idx=exec_ctx.epoch_idx,
            batch_idx=exec_ctx.batch_idx,
            edge="end",
        )

        results.add_callback_result(cb_res=cb_res)

    # Error handling
    def _on_exception(
        self,
        *,
        experiment: Experiment,
        phase: ExperimentPhase,
        exec_ctx: ExecutionContext,
        exception: BaseException,
        results: PhaseResults | None,
    ) -> None:
        """
        Internal hook that wraps callback outputs.

        Description:
            Callback-implemented hooks (eg, `on_exception`) can return any object.
            If not None, it is wrapped into a CallbackResult and stored in `result`.

        Args:
            experiment (Experiment):
                The Experiment on which the phase is attached.
                The model graph can be accessed via `experiment.model_graph`.
            phase (ExperimentPhase):
                The current phase.
            exec_ctx (ExecutionContext):
                The most recent execution context (if available).
            exception (BaseException):
                The exception that was raised.
            results (PhaseResults):
                A result tracker to append callback output to.

        """
        # Preserve trainable state of model graph
        with preserve_frozen_state(experiment.model_graph):
            raw_output: Any = self.on_exception(
                experiment=experiment,
                phase=phase,
                exec_ctx=exec_ctx,
                exception=exception,
            )
        if raw_output is None or results is None:
            return

        if isinstance(raw_output, CallbackResult):
            cb_res = raw_output
        else:
            cb_res = PayloadResult(payload=raw_output)
        cb_res.bind_scope(
            callback_label=self.label,
            phase_label=phase.label,
            epoch_idx=exec_ctx.epoch_idx,
            batch_idx=exec_ctx.batch_idx,
            edge=None,
        )

        results.add_callback_result(cb_res=cb_res)

    # ================================================
    # Public Lifecycle Hooks
    # ================================================
    # Phase lifecycle
    def on_phase_start(
        self,
        *,
        experiment: Experiment,  # noqa: ARG002
        phase: ExperimentPhase,  # noqa: ARG002
    ) -> Any:
        """
        Run once at the start of a phase.

        Args:
            experiment (Experiment):
                The Experiment on which the phase is attached.
                The model graph can be accessed via `experiment.model_graph`.
            phase (ExperimentPhase):
                The current phase.

        Returns:
            CallbackResult | None:
                Optional callback results produced at phase start.

        """
        return None

    def on_phase_end(
        self,
        *,
        experiment: Experiment,  # noqa: ARG002
        phase: ExperimentPhase,  # noqa: ARG002
    ) -> CallbackResult | None:
        """
        Run once at the end of a phase.

        Args:
            experiment (Experiment):
                The Experiment on which the phase is attached.
                The model graph can be accessed via `experiment.model_graph`.
            phase (ExperimentPhase):
                The current phase.

        Returns:
            CallbackResult | None:
                Optional callback results produced at phase end.

        """
        return None

    # Epoch lifecycle
    def on_epoch_start(
        self,
        *,
        experiment: Experiment,  # noqa: ARG002
        phase: ExperimentPhase,  # noqa: ARG002
        exec_ctx: ExecutionContext,  # noqa: ARG002
    ) -> CallbackResult | None:
        """
        Run once at the start of an epoch within a phase.

        Args:
            experiment (Experiment):
                The Experiment on which the phase is attached.
                The model graph can be accessed via `experiment.model_graph`.
            phase (ExperimentPhase):
                The current phase.
            exec_ctx (ExecutionContext):
                The first execution context of the epoch.

        Returns:
            CallbackResult | None:
                Optional callback results produced at epoch start.

        """
        return None

    def on_epoch_end(
        self,
        *,
        experiment: Experiment,  # noqa: ARG002
        phase: ExperimentPhase,  # noqa: ARG002
        exec_ctx: ExecutionContext,  # noqa: ARG002
    ) -> CallbackResult | None:
        """
        Run once at the end of an epoch within a phase.

        Args:
            experiment (Experiment):
                The Experiment on which the phase is attached.
                The model graph can be accessed via `experiment.model_graph`.
            phase (ExperimentPhase):
                The current phase.
            exec_ctx (ExecutionContext):
                The last execution context of the epoch.

        Returns:
            CallbackResult | None:
                Optional callback results produced at epoch end.

        """
        return None

    # Batch lifecycle
    def on_batch_start(
        self,
        *,
        experiment: Experiment,  # noqa: ARG002
        phase: ExperimentPhase,  # noqa: ARG002
        exec_ctx: ExecutionContext,  # noqa: ARG002
    ) -> CallbackResult | None:
        """
        Run once at the start of a batch within an epoch.

        Args:
            experiment (Experiment):
                The Experiment on which the phase is attached.
                The model graph can be accessed via `experiment.model_graph`.
            phase (ExperimentPhase):
                The current phase.
            exec_ctx (ExecutionContext):
                The execution context of this batch.

        Returns:
            CallbackResult | None:
                Optional callback results produced at batch start.

        """
        return None

    def on_batch_end(
        self,
        *,
        experiment: Experiment,  # noqa: ARG002
        phase: ExperimentPhase,  # noqa: ARG002
        exec_ctx: ExecutionContext,  # noqa: ARG002
    ) -> CallbackResult | None:
        """
        Run once at the end of a batch within an epoch.

        Args:
            experiment (Experiment):
                The Experiment on which the phase is attached.
                The model graph can be accessed via `experiment.model_graph`.
            phase (ExperimentPhase):
                The current phase.
            exec_ctx (ExecutionContext):
                The execution context of this batch.

        Returns:
            CallbackResult | None:
                Optional callback results produced at batch end.

        """
        return None

    # Error handling
    def on_exception(
        self,
        *,
        experiment: Experiment,  # noqa: ARG002
        phase: ExperimentPhase,  # noqa: ARG002
        exec_ctx: ExecutionContext,  # noqa: ARG002
        exception: BaseException,  # noqa: ARG002
    ) -> CallbackResult | None:
        """
        Run when an exception occurs during phase execution.

        Args:
            experiment (Experiment):
                The Experiment on which the phase is attached.
                The model graph can be accessed via `experiment.model_graph`.
            phase (ExperimentPhase):
                The current phase.
            exec_ctx (ExecutionContext):
                The most recent execution context (if available).
            exception (BaseException):
                The exception that was raised.

        Returns:
            CallbackResult | None:
                Optional callback results produced during exception handling.

        """
        return None

    # ================================================
    # Configurable
    # ================================================
    @abstractmethod
    def get_config(self) -> dict[str, Any]:
        """
        Return configuration details required to reconstruct this callback.

        Returns:
            dict[str, Any]:
                Configuration used to reconstruct the callback.
                Keys must be strings.

        """
        ...

    @classmethod
    def from_config(cls, config: dict) -> Callback:
        """
        Construct a callback from a configuration dictionary.

        Args:
            config (dict[str, Any]):
                Configuration details. Keys must be strings.

        Returns:
            Callback: Reconstructed callback.

        """
        cb_cls_name = config.get("callback_type")
        if cb_cls_name == "Evaluation":
            from modularml.callbacks.evaluation import Evaluation

            return Evaluation.from_config(config=config)

        msg = (
            f"Unsupported callback class for parent class construction: {cb_cls_name}."
        )
        raise ValueError(msg)
