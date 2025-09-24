from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    Normalizer,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)

from .absolute import Absolute
from .negate import Negate
from .per_sample_min_max import PerSampleMinMaxScaler
from .per_sample_zero import PerSampleZeroStart
from .segmented_scaler import SegmentedScaler

__all__ = [
    "Absolute",
    "MaxAbsScaler",
    "MinMaxScaler",
    "Negate",
    "Normalizer",
    "PerSampleMinMaxScaler",
    "PerSampleZeroStart",
    "PowerTransformer",
    "QuantileTransformer",
    "RobustScaler",
    "SegmentedScaler",
    "StandardScaler",
]


SCALER_REGISTRY = {
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
    "maxabs": MaxAbsScaler,
    "robust": RobustScaler,
    "normalize": Normalizer,
    "quantile": QuantileTransformer,
    "power": PowerTransformer,
    "sample_minmax": PerSampleMinMaxScaler,
    "sample_zero": PerSampleZeroStart,
    "negate": Negate,
    "absolute": Absolute,
    "segmented": SegmentedScaler,
}


from .feature_importance import FeatureImportance

