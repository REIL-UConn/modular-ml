from modularml.api import (
    Accelerator,
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
    RegistrationPolicy,
    ResultRecording,
    ResultsConfig,
    Scaler,
    SimilarityCondition,
    TensorflowBaseModel,
    TorchBaseModel,
    TrainPhase,
    TrainResults,
    scaler_registry,
    supported_scalers,
)
from modularml.registry import register_all

register_all()

# Create a default, empty context immediately
DEFAULT_EXPERIMENT_CONTEXT = ExperimentContext()

# Make it the active context
ExperimentContext._set_active(DEFAULT_EXPERIMENT_CONTEXT)
