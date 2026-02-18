from modularml.utils.registries import CaseInsensitiveRegistry

# Import callback modules
from .early_stopping import EarlyStopping
from .evaluation import Evaluation
from .eval_loss_metric import EvalLossMetric
from .metric import EvaluationMetric, MetricCallback

__all__ = [
    "EarlyStopping",
    "EvalLossMetric",
    "Evaluation",
    "EvaluationMetric",
    "MetricCallback",
]

# ================================================
# Create registry for Callback subclasses
# ================================================
callback_registry = CaseInsensitiveRegistry()


def callback_naming_fn(x):
    return x.__qualname__


# Register modularml callbacks
mml_callbacks: list[type] = [
    EarlyStopping,
    Evaluation,
    EvalLossMetric,
    EvaluationMetric,
    MetricCallback,
]
for t in mml_callbacks:
    callback_registry.register(callback_naming_fn(t), t)
