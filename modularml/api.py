# ================================================
# Experiment & Phases
# ================================================
from modularml.core.experiment.experiment import Experiment
from modularml.core.experiment.phases.phase_group import PhaseGroup
from modularml.core.experiment.phases.eval_phase import EvalPhase
from modularml.core.experiment.results.eval_results import EvalResults
from modularml.core.experiment.phases.train_phase import TrainPhase
from modularml.core.experiment.results.train_results import TrainResults
from modularml.core.experiment.phases.phase import InputBinding
from modularml.core.experiment.phases.fit_phase import FitPhase
from modularml.core.experiment.results.fit_results import FitResults
from modularml.core.experiment.checkpointing import Checkpointing


# ================================================
# Execution Strategies
# ================================================
from modularml.core.execution.cross_validation.cross_validation import CrossValidation
from modularml.core.execution.cross_validation.cv_binding import CVBinding


# ================================================
# Callbacks
# ================================================
# Expose some common classes / non-typically considered callback
from modularml.callbacks.early_stopping import EarlyStopping
from modularml.callbacks.eval_loss_metric import EvalLossMetric

"""
Other callbacks are accessed with:

```python
    from modularml.callbacks import ...
```
"""


# ================================================
# Modeling
# ================================================
from modularml.core.topology.model_graph import ModelGraph
from modularml.core.topology.model_node import ModelNode
from modularml.core.topology.merge_nodes.concat_node import ConcatNode

from modularml.core.training.optimizer import Optimizer
from modularml.core.training.applied_loss import AppliedLoss
from modularml.core.training.loss import Loss

from modularml.core.models.base_model import BaseModel
from modularml.core.models.torch_base_model import TorchBaseModel
from modularml.core.models.tensorflow_base_model import TensorflowBaseModel

"""
Built-in models and merge nodes are accessed with:

```python
    from modularml.models import ...
```
"""


# ================================================
# FeatureSets
# ================================================
from modularml.core.data.featureset import FeatureSet
from modularml.core.data.featureset_view import FeatureSetView


# ================================================
# Splitting
# ================================================
from modularml.core.sampling.similiarity_condition import SimilarityCondition

"""
All built-in splitters are accessed with:

```python
    from modularml.splitters import ...
```
"""


# ================================================
# Scaling
# ================================================
from modularml.core.transforms.scaler import Scaler

supported_scalers = Scaler.get_supported_scalers()
"""
All built-in transformed are accessed with:

```python
    from modularml.transforms import ...
```
"""


__all__ = [
    "AppliedLoss",
    "BaseModel",
    "CVBinding",
    "Checkpointing",
    "ConcatNode",
    "CrossValidation",
    "EarlyStopping",
    "EvalLossMetric",
    "EvalPhase",
    "EvalResults",
    "Experiment",
    "FeatureSet",
    "FeatureSetView",
    "FitPhase",
    "FitResults",
    "InputBinding",
    "Loss",
    "ModelGraph",
    "ModelNode",
    "Optimizer",
    "PhaseGroup",
    "Scaler",
    "SimilarityCondition",
    "TensorflowBaseModel",
    "TorchBaseModel",
    "TrainPhase",
    "TrainResults",
    "supported_scalers",
]
