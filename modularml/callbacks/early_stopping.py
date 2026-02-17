from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from modularml.core.experiment.callbacks.callback import Callback

if TYPE_CHECKING:
    from modularml.core.data.execution_context import ExecutionContext
    from modularml.core.experiment.callbacks.callback_result import CallbackResult
    from modularml.core.experiment.experiment import Experiment
    from modularml.core.experiment.phases.phase import ExperimentPhase
    from modularml.core.experiment.results.phase_results import PhaseResults


class EarlyStopping(Callback):
    """
    Stop training when a monitored metric stops improving.

    Description:
        This callback monitors a named metric from the MetricStore and stops
        training if no improvement is observed for a given number of epochs
        (patience). Improvement is determined by the `mode` parameter:
        "min" expects the metric to decrease, "max" expects it to increase.

        When triggered, EarlyStopping calls `phase.request_stop()` which
        sets a flag that the training loop checks at the end of each epoch.

    Examples:
    ```python
        # Stop if validation loss doesn't improve for 5 epochs
        phase.add_callback(EarlyStopping(monitor="val_loss", patience=5))
    ```

    """

    def __init__(
        self,
        *,
        monitor: str = "val_loss",
        mode: Literal["min", "max"] = "min",
        patience: int = 5,
        min_delta: float = 0.0,
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
        self._reducer = reducer

        self._best_value: float | None = None
        self._wait: int = 0
        self._stopped_epoch: int | None = None

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
    def stopped_epoch(self) -> int | None:
        """The epoch at which training was stopped, or None if not triggered."""
        return self._stopped_epoch

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
        self._wait = 0
        self._stopped_epoch = None

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
        if self._monitor not in results.metric_names():
            return None
        metric_series = results.metrics().where(
            name=self._monitor,
            epoch=exec_ctx.epoch_idx,
        )
        if len(metric_series.values()) == 0:
            return None

        # Reducer on "batch", if needed
        metric_series = metric_series.collapse(axis="batch", reducer=self._reducer)
        entry = metric_series.one()

        # Check improvement
        improved = self._check_improvement(entry.value)
        if improved:
            self._best_value = entry.value
            self._wait = 0
        else:
            self._wait += 1
            if self._wait >= self._patience:
                self._stopped_epoch = exec_ctx.epoch_idx
                phase.request_stop()

        return None

    # ================================================
    # Internal
    # ================================================
    def _check_improvement(self, value: float) -> bool:
        """Check whether the new value is an improvement over the best."""
        if self._best_value is None:
            return True
        if self._mode == "min":
            return value < (self._best_value - self._min_delta)
        return value > (self._best_value + self._min_delta)

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
            reducer=config.get("reducer", "last"),
            label=config.get("label"),
        )
