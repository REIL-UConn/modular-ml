from modularml.core.io.handlers.handler import HandlerRegistry

from modularml.core.data.featureset import FeatureSet
from modularml.core.io.handlers.featureset_handler import FeatureSetHandler


from modularml.core.models.base_model import BaseModel
from modularml.core.io.handlers.model_handler import ModelHandler

from modularml.core.io.handlers.model_graph_handler import ModelGraphHandler
from modularml.core.topology.model_graph import ModelGraph


handler_registry = HandlerRegistry()
handler_registry.register(cls=FeatureSet, handler=FeatureSetHandler())
handler_registry.register(cls=BaseModel, handler=ModelHandler())
handler_registry.register(cls=ModelGraph, handler=ModelGraphHandler())
