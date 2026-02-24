from modularml.api import (
    AppliedLoss,
    BaseModel,
    Checkpointing,
    ConcatNode,
    CrossValidation,
    CVBinding,
    EarlyStopping,
    EvalLossMetric,
    EvalPhase,
    EvalResults,
    Experiment,
    ExperimentContext,
    FeatureSet,
    FeatureSetView,
    FitPhase,
    FitResults,
    InputBinding,
    Loss,
    ModelGraph,
    ModelNode,
    Optimizer,
    PhaseGroup,
    PhaseGroupResults,
    ResultRecording,
    Scaler,
    SimilarityCondition,
    TensorflowBaseModel,
    TorchBaseModel,
    TrainPhase,
    TrainResults,
    supported_scalers,
)
from modularml.registry import register_all

register_all()

# Create a default, empty context immediately
DEFAULT_EXPERIMENT_CONTEXT = ExperimentContext()

# Make it the active context
ExperimentContext._set_active(DEFAULT_EXPERIMENT_CONTEXT)
