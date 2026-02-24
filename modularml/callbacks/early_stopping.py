from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from modularml.core.experiment.callbacks.callback import Callback
from modularml.core.experiment.callbacks.callback_result import CallbackResult
from modularml.core.experiment.phases.train_phase import TrainPhase
from modularml.utils.logging.logger import get_logger

if TYPE_CHECKING:
    from modularml.core.data.execution_context import ExecutionContext
    from modularml.core.experiment.experiment import Experiment
    from modularml.core.experiment.phases.phase import ExperimentPhase
    from modularml.core.experiment.results.phase_results import PhaseResults

logger = get_logger(name="EarlyStopping")


@dataclass
class EarlyStoppingResult(CallbackResult):
    """
    Result emitted by EarlyStopping on phase end.

    Attributes:
        best_epoch (int | None):
            The epoch index that achieved the best monitored metric value,
            or None if no metric was ever observed.
        best_value (float | None):
            The best observed metric value.
        stopped_epoch (int | None):
            The epoch at which training was stopped, or None if training
            completed without early stopping.
        restored (bool):
            Whether the model state was restored to the best epoch.

    """

    kind: ClassVar[str] = "early_stopping"

    best_epoch: int | None = None
    best_value: float | None = None
    stopped_epoch: int | None = None
    restored: bool = False


class EarlyStopping(Callback):
    """
    Stop training when a monitored metric stops improving.

    This callback monitors a named metric from the MetricStore and stops
    training if no improvement is observed for a given number of epochs
    (patience). Improvement is determined by the `mode` parameter:
    "min" expects the metric to decrease, "max" expects it to increase.

    When triggered, EarlyStopping calls `phase.request_stop()` which
    sets a flag that the training loop checks at the end of each epoch.

    When `restore_best=True`, the model state is restored to the epoch
    with the best monitored metric value at the end of the phase. If the
    phase has a `Checkpointing` callback configured, its saved states are
    used for restoration. Otherwise, EarlyStopping manages its own
    in-memory state snapshots as a fallback.

    Example:
        >>> # Stop if validation loss doesn't improve for 5 epochs
        >>> phase.add_callback(  # doctest: +SKIP
        ...     EarlyStopping(monitor="val_loss", patience=5)
        ... )

        >>> # Stop and restore model to best epoch
        >>> phase.add_callback(  # doctest: +SKIP
        ...     EarlyStopping(
        ...         monitor="val_loss",
        ...         patience=5,
        ...         restore_best=True,
        ...     )
        ... )

    """

    def __init__(
        self,
        *,
        monitor: str = "val_loss",
        mode: Literal["min", "max"] = "min",
        patience: int = 5,
        min_delta: float = 0.0,
        restore_best: bool = False,
        reducer: Literal["mean", "sum", "last", "first"] = "last",
        label: str | None = None,
        execution_order: int = 1,
    ) -> None:
        """
        Initialize an EarlyStopping callback.

        Args:
            monitor (str, optional):
                Name of the metric to monitor. Must match a metric name that
                is logged into the MetricStore during training (e.g. by a
                EvalLossMetric or custom MetricCallback).
                Defaults to "val_loss".

            mode (Literal["min", "max"], optional):
                Whether the monitored metric should be minimized or maximized.
                Defaults to "min".

            patience (int, optional):
                Number of epochs with no improvement after which training
                will be stopped. Defaults to 5.

            min_delta (float, optional):
                Minimum change in the monitored metric to qualify as an
                improvement. Defaults to 0.0.

            restore_best (bool, optional):
                Whether to restore the model state to the epoch with the best
                monitored metric value at the end of the phase. If the phase
                has a Checkpointing callback, its saved states are used.
                Otherwise, in-memory snapshots are managed automatically.
                Defaults to False.

            reducer (str, optional):
                If the `monitor` metric is produced more than one per epoch, `reducer`
                defines how to aggregate all values in that epoch. Typically, `monitor`
                is produced at most once per epoch and this argument is not used.

            label (str | None, optional):
                Stable identifier for this callback. Defaults to "EarlyStopping".

            execution_order (int, optional):
                Used for execution ordering or multiple callbacks, where higher values
                are executed later than lower values. This value should be greater than
                the callback that produces the `monitor` metric. Unless you manually
                modified the other callback execution orders, a value of 1 is fine.

        """
        super().__init__(
            label=label or "EarlyStopping",
            execution_order=execution_order,
        )
        self._monitor = monitor
        self._mode = mode
        self._patience = patience
        self._min_delta = min_delta
        self._restore_best = restore_best
        self._reducer = reducer

        self._best_value: float | None = None
        self._best_epoch: int | None = None
        self._wait: int = 0
        self._stopped_epoch: int | None = None

        # In-memory state for restore_best (used when no Checkpointing)
        self._best_state: dict[str, Any] | None = None

    # ================================================
    # Properties
    # ================================================
    @property
    def monitor(self) -> str:
        """The name of the monitored metric."""
        return self._monitor

    @property
    def best_value(self) -> float | None:
        """The best observed metric value so far."""
        return self._best_value

    @property
    def best_epoch(self) -> int | None:
        """The epoch index that achieved the best metric value, or None."""
        return self._best_epoch

    @property
    def stopped_epoch(self) -> int | None:
        """The epoch at which training was stopped, or None if not triggered."""
        return self._stopped_epoch

    @property
    def restore_best(self) -> bool:
        """Whether model state restoration to the best epoch is enabled."""
        return self._restore_best

    # ================================================
    # Lifecycle Hooks
    # ================================================
    def on_phase_start(
        self,
        *,
        experiment: Experiment,
        phase: ExperimentPhase,
        results: PhaseResults | None = None,
    ) -> None:
        """Reset internal state at the start of each phase."""
        self._best_value = None
        self._best_epoch = None
        self._wait = 0
        self._stopped_epoch = None
        self._best_state = None

    def on_epoch_end(
        self,
        *,
        experiment: Experiment,
        phase: ExperimentPhase,
        exec_ctx: ExecutionContext,
        results: PhaseResults | None = None,
    ) -> CallbackResult | None:
        """Check monitored metric and request stop if patience is exceeded."""
        if results is None:
            return None

        # Get metric values for this epoch
        metric_value = self._get_metric_value(
            results=results,
            epoch_idx=exec_ctx.epoch_idx,
        )

        # If no metric was recorded, treat as no improvement
        if metric_value is None:
            self._wait += 1
            if self._wait >= self._patience:
                self._stopped_epoch = exec_ctx.epoch_idx
                phase.request_stop()
            return None

        # Check improvement
        improved = self._check_improvement(metric_value)
        if improved:
            self._best_value = metric_value
            self._best_epoch = exec_ctx.epoch_idx
            self._wait = 0

            # Snapshot state for restore_best (in-memory fallback)
            if self._restore_best and not self._has_phase_checkpointing(phase):
                self._best_state = experiment.model_graph.get_state()
        else:
            self._wait += 1
            if self._wait >= self._patience:
                self._stopped_epoch = exec_ctx.epoch_idx
                phase.request_stop()

        return None

    def on_phase_end(
        self,
        *,
        experiment: Experiment,
        phase: ExperimentPhase,
        results: PhaseResults | None = None,
    ) -> EarlyStoppingResult | None:
        """Restore best model state (if enabled) and return summary result."""
        restored = False

        if self._restore_best and self._best_epoch is not None:
            restored = self._restore_best_state(
                experiment=experiment,
                phase=phase,
            )

        return EarlyStoppingResult(
            best_epoch=self._best_epoch,
            best_value=self._best_value,
            stopped_epoch=self._stopped_epoch,
            restored=restored,
        )

    # ================================================
    # Internal
    # ================================================
    def _get_metric_value(
        self,
        *,
        results: PhaseResults,
        epoch_idx: int,
    ) -> float | None:
        """Extract the monitored metric value for a given epoch, or None."""
        if self._monitor not in results.metric_names():
            return None
        metric_series = results.metrics().where(
            name=self._monitor,
            epoch=epoch_idx,
        )
        if len(metric_series.values()) == 0:
            return None

        # Reducer on "batch", if needed
        metric_series = metric_series.collapse(axis="batch", reducer=self._reducer)
        entry = metric_series.one()
        return entry.value

    def _check_improvement(self, value: float) -> bool:
        """Check whether the new value is an improvement over the best."""
        if self._best_value is None:
            return True
        if self._mode == "min":
            return value < (self._best_value - self._min_delta)
        return value > (self._best_value + self._min_delta)

    def _has_phase_checkpointing(self, phase: ExperimentPhase) -> bool:
        """Check whether the phase has a Checkpointing callback configured."""
        return isinstance(phase, TrainPhase) and (phase.checkpointing is not None)

    def _restore_best_state(
        self,
        *,
        experiment: Experiment,
        phase: TrainPhase,
    ) -> bool:
        """
        Restore model state to the best epoch.

        Tries phase Checkpointing first, falls back to in-memory snapshot.

        Returns:
            bool: Whether restoration was successful.

        """
        # Try phase Checkpointing attr
        if self._has_phase_checkpointing(phase):
            ckpt = phase.checkpointing
            if ckpt.has_key(self._best_epoch):
                if ckpt.mode == "memory":
                    state = ckpt.get_state(self._best_epoch)
                    experiment.model_graph.set_state(state)
                else:
                    path = ckpt.get_path(self._best_epoch)
                    experiment.model_graph.restore_checkpoint(path)
                msg = (
                    f"Restored model to best epoch {self._best_epoch} "
                    f"(via Checkpointing, {self._monitor}={self._best_value})."
                )
                logger.info(msg=msg, stacklevel=2)
                return True

        # Fallback to in-memory snapshot
        if self._best_state is not None:
            experiment.model_graph.set_state(self._best_state)
            msg = (
                f"Restored model to best epoch {self._best_epoch} "
                f"(in-memory, {self._monitor}={self._best_value})."
            )
            logger.info(msg=msg, stacklevel=2)
            return True

        msg = (
            f"restore_best=True but no saved state found for epoch "
            f"{self._best_epoch}. Model was not restored."
        )
        logger.warning(msg=msg, stacklevel=2)
        return False

    # ================================================
    # Configurable
    # ================================================
    def get_config(self) -> dict[str, Any]:
        """Return configuration details required to reconstruct this callback."""
        return {
            "callback_type": self.__class__.__qualname__,
            "monitor": self._monitor,
            "mode": self._mode,
            "patience": self._patience,
            "min_delta": self._min_delta,
            "restore_best": self._restore_best,
            "reducer": self._reducer,
            "label": self.label,
        }

    @classmethod
    def from_config(cls, config: dict) -> EarlyStopping:
        """Construct from config data."""
        return cls(
            monitor=config.get("monitor", "val_loss"),
            mode=config.get("mode", "min"),
            patience=config.get("patience", 5),
            min_delta=config.get("min_delta", 0.0),
            restore_best=config.get("restore_best", False),
            reducer=config.get("reducer", "last"),
            label=config.get("label"),
        )
