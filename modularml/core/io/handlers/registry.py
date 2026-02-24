"""Registry of IO handlers for different artifact types."""

from modularml.core.io.handlers.handler import HandlerRegistry

from modularml.core.data.featureset import FeatureSet
from modularml.core.io.handlers.featureset_handler import FeatureSetHandler

from modularml.core.io.handlers.model_graph_handler import ModelGraphHandler
from modularml.core.topology.model_graph import ModelGraph

from modularml.core.io.checkpoint import Checkpoint
from modularml.core.io.handlers.checkpoint_handler import CheckpointHandler

from modularml.core.experiment.experiment import Experiment
from modularml.core.io.handlers.experiment_handler import ExperimentHandler


handler_registry = HandlerRegistry()
handler_registry.register(cls=FeatureSet, handler=FeatureSetHandler())
handler_registry.register(cls=ModelGraph, handler=ModelGraphHandler())
handler_registry.register(cls=Checkpoint, handler=CheckpointHandler())
handler_registry.register(cls=Experiment, handler=ExperimentHandler())
