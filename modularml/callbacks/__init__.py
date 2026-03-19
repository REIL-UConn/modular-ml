from modularml.utils.registries import CaseInsensitiveRegistry

# Import callback modules
from .artifact_result import ArtifactResult
from .early_stopping import EarlyStopping
from .evaluation import Evaluation
from .eval_loss_metric import EvalLossMetric
from .metric import EvaluationMetric, MetricCallback, MetricResult

__all__ = [
    "ArtifactResult",
    "EarlyStopping",
    "EvalLossMetric",
    "Evaluation",
    "EvaluationMetric",
    "MetricCallback",
    "MetricResult",
]

# ================================================
# Create registry for Callback subclasses
# ================================================
callback_registry = CaseInsensitiveRegistry()


def callback_naming_fn(x):
    return x.__qualname__


# Register modularml callbacks
mml_callbacks: list[type] = [
    ArtifactResult,
    EarlyStopping,
    Evaluation,
    EvalLossMetric,
    EvaluationMetric,
    MetricCallback,
    MetricResult,
]
for t in mml_callbacks:
    callback_registry.register(callback_naming_fn(t), t)
