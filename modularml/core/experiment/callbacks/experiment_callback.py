from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from modularml.utils.nn.training import preserve_frozen_state

if TYPE_CHECKING:
    from modularml.core.experiment.callbacks.callback_result import CallbackResult
    from modularml.core.experiment.experiment import Experiment
    from modularml.core.experiment.phases.phase import ExperimentPhase
    from modularml.core.experiment.phases.phase_group import PhaseGroup


class ExperimentCallback(ABC):
    """
    Base class for experiment-level lifecycle callbacks.

    Description:
        These callbacks respond to `Experiment.run()` execution events
        at phase and group boundaries. They are distinct from phase-level
        :class:`Callback` instances, which respond to training events
        (epoch/batch boundaries).

        Subclasses should override the hooks they need and implement
        :meth:`get_config` / :meth:`from_config` for serialization.

    Examples:
    ```python
        class MyExperimentLogger(ExperimentCallback):
            def on_phase_end(self, *, experiment, phase, results):
                print(f"Phase '{phase.label}' completed.")

            def get_config(self):
                return {"callback_type": "MyExperimentLogger"}
    ```

    """

    def __init__(
        self,
        *,
        label: str | None = None,
        execution_order: int = 0,
    ) -> None:
        """
        Initialize a new experiment callback.

        Args:
            label (str | None, optional):
                Stable identifier for this callback. If None, defaults to
                the class name.
            execution_order (int, optional):
                Used for ordering when multiple callbacks are registered.
                Higher values execute later. Defaults to 0.

        """
        self._label = label or self.__class__.__qualname__
        self._exec_order = int(execution_order)

    # ================================================
    # Properties
    # ================================================
    @property
    def label(self) -> str:
        """Stable identifier of this callback instance."""
        return self._label

    # ================================================
    # Internal Lifecycle Hooks
    # ================================================
    # Phase lifecycle
    def _on_phase_start(
        self,
        *,
        experiment: Experiment,
        phase: ExperimentPhase,
    ) -> None:
        """
        Run before a phase is executed within `Experiment.run()`.

        Args:
            experiment (Experiment):
                The running Experiment instance.
            phase (ExperimentPhase):
                The phase about to be executed.

        """
        # Preserve trainable state of model graph
        with preserve_frozen_state(experiment.model_graph):
            _ = self.on_phase_start(
                experiment=experiment,
                phase=phase,
            )

        raise NotImplementedError

    def _on_phase_end(
        self,
        *,
        experiment: Experiment,
        phase: ExperimentPhase,
    ) -> None:
        """
        Run after a phase is executed within `Experiment.run()`.

        Args:
            experiment (Experiment):
                The running Experiment instance.
            phase (ExperimentPhase):
                The phase that was just executed.

        """
        # Preserve trainable state of model graph
        with preserve_frozen_state(experiment.model_graph):
            _ = self.on_phase_end(
                experiment=experiment,
                phase=phase,
            )

        raise NotImplementedError

    # Group lifecycle
    def _on_group_start(
        self,
        *,
        experiment: Experiment,
        phase_group: PhaseGroup,
    ) -> None:
        """
        Run before a phase gorup is executed within `Experiment.run()`.

        Args:
            experiment (Experiment):
                The running Experiment instance.
            phase_group (PhaseGroup):
                The phase group about to be executed.

        """
        # Preserve trainable state of model graph
        with preserve_frozen_state(experiment.model_graph):
            _ = self.on_group_start(
                experiment=experiment,
                phase_group=phase_group,
            )

        raise NotImplementedError

    def _on_group_end(
        self,
        *,
        experiment: Experiment,
        phase_group: PhaseGroup,
    ) -> None:
        """
        Run after a phase group is executed within `Experiment.run()`.

        Args:
            experiment (Experiment):
                The running Experiment instance.
            phase_group (PhaseGroup):
                The phase group that was just executed.

        """
        # Preserve trainable state of model graph
        with preserve_frozen_state(experiment.model_graph):
            _ = self.on_group_end(
                experiment=experiment,
                phase_group=phase_group,
            )

        raise NotImplementedError

    # Experiment lifecycle
    def _on_experiment_start(
        self,
        *,
        experiment: Experiment,
    ) -> None:
        """
        Run before `Experiment.run()` is started.

        Args:
            experiment (Experiment):
                The running Experiment instance.

        """
        # Preserve trainable state of model graph
        with preserve_frozen_state(experiment.model_graph):
            _ = self.on_experiment_start(
                experiment=experiment,
            )
        raise NotImplementedError

    def _on_experiment_end(
        self,
        *,
        experiment: Experiment,
    ) -> None:
        """
        Run after `Experiment.run()` completes.

        Args:
            experiment (Experiment):
                The running Experiment instance.

        """
        # Preserve trainable state of model graph
        with preserve_frozen_state(experiment.model_graph):
            _ = self.on_experiment_end(
                experiment=experiment,
            )
        raise NotImplementedError

    # Error handling
    def _on_exception(
        self,
        *,
        experiment: Experiment,
        phase: ExperimentPhase | None = None,
        exception: BaseException,
    ) -> None:
        """
        Runs if any exception occurs.

        Args:
            experiment (Experiment):
                The running Experiment instance.
            phase (ExperimentPhase):
                The phase being executed when the exception occurs.
                None if error ocurred outside of active phase.
            exception (BaseException):
                The exception that was called.

        """
        # Preserve trainable state of model graph
        with preserve_frozen_state(experiment.model_graph):
            _ = self.on_exception(
                experiment=experiment,
            )
        raise NotImplementedError

    # ================================================
    # Public Lifecycle Hooks
    # ================================================
    # Phase lifecycle
    def on_phase_start(
        self,
        *,
        experiment: Experiment,
        phase: ExperimentPhase,
    ) -> CallbackResult | None:
        """
        Run before a phase is executed within `Experiment.run()`.

        Args:
            experiment (Experiment):
                The running Experiment instance.
            phase (ExperimentPhase):
                The phase about to be executed.

        Returns:
            CallbackResult | None:
                Optional callback results produced.

        """
        return None

    def on_phase_end(
        self,
        *,
        experiment: Experiment,
        phase: ExperimentPhase,
    ) -> CallbackResult | None:
        """
        Run after a phase is executed within `Experiment.run()`.

        Args:
            experiment (Experiment):
                The running Experiment instance.
            phase (ExperimentPhase):
                The phase that was just executed.

        Returns:
            CallbackResult | None:
                Optional callback results produced.

        """
        return None

    # Group lifecycle
    def on_group_start(
        self,
        *,
        experiment: Experiment,
        phase_group: PhaseGroup,
    ) -> CallbackResult | None:
        """
        Run before a phase gorup is executed within `Experiment.run()`.

        Args:
            experiment (Experiment):
                The running Experiment instance.
            phase_group (PhaseGroup):
                The phase group about to be executed.

        Returns:
            CallbackResult | None:
                Optional callback results produced.

        """
        return None

    def on_group_end(
        self,
        *,
        experiment: Experiment,
        phase_group: PhaseGroup,
    ) -> CallbackResult | None:
        """
        Run after a phase group is executed within `Experiment.run()`.

        Args:
            experiment (Experiment):
                The running Experiment instance.
            phase_group (PhaseGroup):
                The phase group that was just executed.

        Returns:
            CallbackResult | None:
                Optional callback results produced.

        """
        return None

    # Experiment lifecycle
    def on_experiment_start(
        self,
        *,
        experiment: Experiment,
    ) -> CallbackResult | None:
        """
        Run before `Experiment.run()` is started.

        Args:
            experiment (Experiment):
                The running Experiment instance.

        Returns:
            CallbackResult | None:
                Optional callback results produced.

        """
        return None

    def on_experiment_end(
        self,
        *,
        experiment: Experiment,
    ) -> CallbackResult | None:
        """
        Run after `Experiment.run()` completes.

        Args:
            experiment (Experiment):
                The running Experiment instance.

        Returns:
            CallbackResult | None:
                Optional callback results produced.

        """
        return None

    # Error handling
    def on_exception(
        self,
        *,
        experiment: Experiment,
        phase: ExperimentPhase | None = None,
        exception: BaseException,
    ) -> CallbackResult | None:
        """
        Runs if any exception occurs.

        Args:
            experiment (Experiment):
                The running Experiment instance.
            phase (ExperimentPhase):
                The phase being executed when the exception occurs.
                None if error ocurred outside of active phase.
            exception (BaseException):
                The exception that was called.

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

        """
        ...

    @classmethod
    def from_config(cls, config: dict) -> ExperimentCallback:
        """
        Construct an experiment callback from a configuration dictionary.

        Args:
            config (dict[str, Any]):
                Configuration details.

        Returns:
            ExperimentCallback: Reconstructed callback.

        """
        cb_cls_name = config.get("callback_type")

        msg = f"Unsupported experiment callback class: {cb_cls_name}."
        raise ValueError(msg)
