from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from modularml.callbacks.metric import EvaluationMetric, MetricResult
from modularml.core.training.applied_loss import AppliedLoss

if TYPE_CHECKING:
    from modularml.callbacks.evaluation import EvaluationCallbackResult
    from modularml.core.data.execution_context import ExecutionContext


class EvalLossMetric(EvaluationMetric):
    """
    Extracts a scalar loss from an Evaluation callback and logs it as a metric.

    Description:
        This built-in EvaluationMetric reads the results of its parent
        Evaluation callback and extracts the aggregated loss value for a
        specific node. The extracted value is logged as a named metric
        (default: "val_loss") into the MetricStore.

    Examples:
        ```python
        mse_loss = AppliedLoss(...)
        val_metric = EvalLossMetric(loss=mse_loss, name="val_loss")
        eval_cb = Evaluation(eval_phase=eval_phase, metrics=[val_metric])
        phase.add_callback(eval_cb)
        ```

    """

    def __init__(
        self,
        *,
        loss: AppliedLoss,
        reducer: Literal["sum", "mean"] = "mean",
        name: str = "val_loss",
    ) -> None:
        """
        Initialize a ValidationLossMetric.

        Args:
            loss (AppliedLoss):
                An applied loss to track. Will be appended to the parent
                EvalPhase in Evaluation, if not already present.
            reducer (Literal["sum", "mean"], optional):
                How to aggregate per-batch losses from the evaluation into
                a single scalar. Defaults to "mean".
            name (str, optional):
                The metric name to log under. Defaults to "val_loss".
            mode (Literal["min", "max"], optional):
                Whether lower or higher values are better. Defaults to "min".

        """
        super().__init__(name=name)

        # Validate loss
        if not isinstance(loss, AppliedLoss):
            msg = f"Expected type of AppliedLoss. Received: {type(loss)}."
            raise TypeError(msg)
        self._loss = loss

        # Validate reducer
        red_methods = ["sum", "mean"]
        if reducer not in red_methods:
            msg = f"Expected one of {red_methods}. Received: {reducer}."
            raise ValueError(msg)
        self._reducer = reducer

    def extract(
        self,
        *,
        eval_result: EvaluationCallbackResult,
        exec_ctx: ExecutionContext | None = None,
    ) -> MetricResult | None:
        """Extract aggregated loss from the Evaluation result."""
        # Check that we have actual EvalResults to use
        eval_res = eval_result.eval_results
        if (eval_res is None) or (not eval_res._execution):
            return None

        # Get loss records for the defined AppliedLoss
        lds = eval_res.losses(node=self._loss.node_id).where(label=self._loss.label)
        if len(lds.values()) == 0:
            return None

        # Aggregate over all batches in this epoch
        lds = lds.collapse(axis="batch", reducer=self._reducer)

        # Remove epoch & label axes
        lds = lds.squeeze()
        if len(lds.axes) != 0:
            msg = (
                f"Failed to collapse LossDataSeries. Expected no axis labels "
                f"remain. Remaining axes: {lds.axes}."
            )
            raise RuntimeError(msg)

        # Return aux loss (as float)
        lr = lds.one()
        return MetricResult(
            metric_name=self.metric_name,
            metric_value=lr.to_float().auxiliary,
        )

    # ================================================
    # Configurable
    # ================================================
    def get_config(self) -> dict[str, Any]:
        """Return configuration details required to reconstruct this callback."""
        cfg = super().get_config()
        cfg.update(
            {
                "loss": self._loss.get_config(),
                "reducer": self._reducer,
            },
        )
        return cfg

    @classmethod
    def from_config(cls, config: dict) -> EvalLossMetric:
        """Construct from config data."""
        return cls(
            loss=AppliedLoss.from_config(config["loss"]),
            reducer=config.get("reducer", "mean"),
            name=config.get("name", "val_loss"),
        )
