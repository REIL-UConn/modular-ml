from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

from modularml.core.experiment.callbacks.callback import Callback
from modularml.core.experiment.callbacks.callback_result import CallbackResult

if TYPE_CHECKING:
    from modularml.callbacks.evaluation import EvaluationCallbackResult
    from modularml.core.data.execution_context import ExecutionContext

RESERVED_METRIC_NAMES = ["train_loss"]


@dataclass
class MetricResult(CallbackResult):
    """
    A callback result carrying a named scalar metric value.

    Description:
        When a MetricCallback (or any callback) returns a MetricResult,
        the framework automatically logs the value into the phase's MetricStore
        in addition to storing it in the normal callback results.

    """

    kind: ClassVar[str] = "metric"

    metric_name: str = ""
    metric_value: float = 0.0


class MetricCallback(Callback):
    """
    Base class for callbacks that produce named scalar metric values.

    Description:
        A MetricCallback is a Callback that produces named scalar values which
        are automatically logged into the phase's MetricStore. This enables
        consumer callbacks (e.g. EarlyStopping) to reference metrics by name.

        Subclasses should override the lifecycle hooks they need (typically
        `on_epoch_end`) and return a `MetricResult` when a value is
        available, or `None` otherwise.

    """

    execution_order: ClassVar[int] = 0

    def __init__(
        self,
        *,
        name: str,
    ) -> None:
        """
        Initialize a MetricCallback.

        Args:
            name (str):
                The metric name that will be used in the MetricStore
                (e.g. "val_loss", "val_r2").

        """
        if name in RESERVED_METRIC_NAMES:
            msg = f"'{name}' is a reserved metric name. Use another name."
            raise ValueError(msg)

        super().__init__(label=name)
        self._metric_name = name

    @property
    def metric_name(self) -> str:
        """The name under which this metric is logged in the MetricStore."""
        return self._metric_name

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
            "name": self._metric_name,
        }

    @classmethod
    def from_config(cls, config: dict) -> MetricCallback:
        """
        Construct an MetricCallback callback from config data.

        Args:
            config (dict[str, Any]):
                Configuration details. Keys must be strings.

        Returns:
            MetricCallback: Reconstructed callback.

        """
        if config.get("callback_type") != cls.__qualname__:
            msg = f"Invalid config data for {cls.__qualname__} callback."
            raise ValueError(msg)

        return cls(
            name=config["name"],
        )


class EvaluationMetric(MetricCallback):
    """
    A metric that extracts its value from an Evaluation callback's results.

    Description:
        EvaluationMetric is a MetricCallback that is owned by an Evaluation
        callback. When the EvaluationCallback executes (e.g., at epoch end),
        it passes its `EvaluationCallbackResult` directly to each attached
        metric's :meth:`extract` method.

        This avoids the need for metrics to search through `results._callbacks`
        by label.

    Example:
        Custom metric subclasses can be defined via:

        >>> class ValidationR2(EvaluationMetric):  # doctest: +SKIP
        ...     def __init__(self, node_id):
        ...         super().__init__(name="val_r2", mode="max")
        ...         self._node_id = node_id
        ...
        ...     def extract(self, *, eval_result, exec_ctx=None):
        ...         preds = eval_result.get_node_outputs(node=self._node_id, fmt="np")
        ...         targets = eval_result.get_node_targets(node=self._node_id, fmt="np")
        ...         r2 = r2_score(targets, preds)
        ...         return MetricResult(
        ...             metric_name=self.metric_name,
        ...             metric_value=float(r2),
        ...         )
        ...
        ...     def get_config(self):
        ...         cfg = super().get_config()
        ...         cfg["node_id"] = self._node_id
        ...         return cfg

    """

    @abstractmethod
    def extract(
        self,
        *,
        eval_result: EvaluationCallbackResult,
        exec_ctx: ExecutionContext | None = None,
    ) -> MetricResult | None:
        """
        Extract a scalar metric from an Evaluation callback's results.

        Args:
            eval_result (EvaluationCallbackResult):
                The result produced by the parent Evaluation callback.
            exec_ctx (ExecutionContext | None, optional):
                The execution context at the time of evaluation. May be None
                when the evaluation is triggered at phase start.

        Returns:
            MetricResult | None:
                A metric result to log, or None if the metric cannot be
                computed from this evaluation.

        """
        ...
