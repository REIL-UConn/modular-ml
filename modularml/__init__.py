from modularml.api import (
    AppliedLoss,
    BaseModel,
    CVBinding,
    Checkpointing,
    ConcatNode,
    CrossValidation,
    EarlyStopping,
    EvalLossMetric,
    EvalPhase,
    EvalResults,
    Experiment,
    FeatureSet,
    FeatureSetView,
    FitPhase,
    FitResults,
    InputBinding,
    Loss,
    ModelGraph,
    ModelNode,
    Optimizer,
    Scaler,
    SimilarityCondition,
    TensorflowBaseModel,
    TorchBaseModel,
    TrainPhase,
    TrainResults,
    supported_scalers,
)


from modularml.core.experiment.experiment_context import ExperimentContext

from modularml.registry import register_all

register_all()

# Create a default, empty context immediately
DEFAULT_EXPERIMENT_CONTEXT = ExperimentContext()

# Make it the active context
ExperimentContext._set_active(DEFAULT_EXPERIMENT_CONTEXT)
