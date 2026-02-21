"""Registration helpers for model serialization and kind metadata."""

from modularml.core.io.symbol_registry import symbol_registry
from modularml.core.io.conventions import SerializationKind, kind_registry

from modularml.models import model_naming_fn, model_registry

from .base_model import BaseModel
from .torch_base_model import TorchBaseModel
from .torch_wrapper import TorchModelWrapper


def register_builtin():
    """Register :class:`BaseModel` implementations with the symbol registry."""
    symbol_registry.register_builtin_class(
        key="BaseModel",
        cls=BaseModel,
    )
    symbol_registry.register_builtin_class(
        key="TorchBaseModel",
        cls=TorchBaseModel,
    )
    symbol_registry.register_builtin_class(
        key="TorchModelWrapper",
        cls=TorchModelWrapper,
    )

    # Register registries
    symbol_registry.register_builtin_registry(
        import_path="modularml.models.model_registry",
        registry=model_registry,
        naming_fn=model_naming_fn,
    )


def register_kinds():
    """Register serialization kinds for built-in model artifacts."""
    kind_registry.register(
        cls=BaseModel,
        kind=SerializationKind(name="BaseModel", kind="md"),
    )
