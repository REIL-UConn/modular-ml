from contextvars import ContextVar
from modularml.core.experiment.experiment_context import ExperimentContext
from modularml.core.data.featureset import FeatureSet
from modularml.core.data.featureset_view import FeatureSetView

from modularml.core.sampling.similiarity_condition import SimilarityCondition

from modularml.registry import register_all

register_all()

# Create a default, empty context immediately
DEFAULT_EXPERIMENT_CONTEXT = ExperimentContext()

# Make it the active context
ExperimentContext._set_active(DEFAULT_EXPERIMENT_CONTEXT)
