"""Built-in scaler registry wiring for ModularML."""

from modularml.utils.registries import CaseInsensitiveRegistry

# Import all scaler modules
from .absolute import Absolute
from .negate import Negate
from .per_sample_min_max import PerSampleMinMaxScaler
from .per_sample_zero import PerSampleZeroStart
from .segmented_scaler import SegmentedScaler

# Optionally import sklearn scalers
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    MaxAbsScaler,
    RobustScaler,
    Normalizer,
    QuantileTransformer,
    PowerTransformer,
)


__all__ = [  # noqa: RUF022
    # classes
    "Absolute",
    "Negate",
    "PerSampleMinMaxScaler",
    "PerSampleZeroStart",
    "SegmentedScaler",
    # sklearn
    "StandardScaler",
    "MinMaxScaler",
    "MaxAbsScaler",
    "RobustScaler",
    "Normalizer",
    "QuantileTransformer",
    "PowerTransformer",
]

# Create registry
scaler_registry = CaseInsensitiveRegistry()


def scaler_naming_fn(x):
    """
    Return the registry key for a scaler class.

    Args:
        x (type): Scaler class being registered.

    Returns:
        str: Qualname used to index the scaler in :data:`scaler_registry`.

    """
    return x.__qualname__


# Register sklearn scalers
sklearn_scalers: list[type] = [
    StandardScaler,
    MinMaxScaler,
    MaxAbsScaler,
    RobustScaler,
    Normalizer,
    QuantileTransformer,
    PowerTransformer,
]
for t in sklearn_scalers:
    scaler_registry.register(scaler_naming_fn(t), t)

# Register modularml scalers
mml_scalers: list[type] = [
    Absolute,
    Negate,
    PerSampleMinMaxScaler,
    PerSampleZeroStart,
    SegmentedScaler,
]
for t in mml_scalers:
    scaler_registry.register(scaler_naming_fn(t), t)
