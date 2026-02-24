API Reference
=============

.. currentmodule:: modularml.api

Runtime Context
---------------

.. autoclass:: ExperimentContext
   :members:
   :show-inheritance:


Experiment & Phases
-------------------

.. autoclass:: Experiment
   :members:
   :show-inheritance:

.. autoclass:: PhaseGroup
   :members:
   :show-inheritance:

.. autoclass:: PhaseGroupResults
   :members:
   :show-inheritance:

.. autoclass:: Checkpointing
   :members:
   :show-inheritance:

.. autoclass:: InputBinding
   :members:
   :show-inheritance:

.. autoclass:: ResultRecording
   :members:
   :show-inheritance:

.. autoclass:: FitPhase
   :members:
   :show-inheritance:

.. autoclass:: FitResults
   :members:
   :show-inheritance:

.. autoclass:: TrainPhase
   :members:
   :show-inheritance:

.. autoclass:: TrainResults
   :members:
   :show-inheritance:

.. autoclass:: EvalPhase
   :members:
   :show-inheritance:

.. autoclass:: EvalResults
   :members:
   :show-inheritance:


Execution Strategies
--------------------

.. autoclass:: CrossValidation
   :members:
   :show-inheritance:

.. autoclass:: CVBinding
   :members:
   :show-inheritance:


Callbacks
---------

.. autoclass:: EarlyStopping
   :members:
   :show-inheritance:

.. autoclass:: EvalLossMetric
   :members:
   :show-inheritance:


Modeling
--------

.. autoclass:: ModelGraph
   :members:
   :show-inheritance:

.. autoclass:: ModelNode
   :members:
   :show-inheritance:

.. autoclass:: ConcatNode
   :members:
   :show-inheritance:

.. autoclass:: Loss
   :members:
   :show-inheritance:

.. autoclass:: AppliedLoss
   :members:
   :show-inheritance:

.. autoclass:: Optimizer
   :members:
   :show-inheritance:

.. autoclass:: BaseModel
   :members:
   :show-inheritance:

.. autoclass:: TorchBaseModel
   :members:
   :show-inheritance:

.. autoclass:: TensorflowBaseModel
   :members:
   :show-inheritance:


FeatureSets
-----------

.. autoclass:: FeatureSet
   :members:
   :show-inheritance:

.. autoclass:: FeatureSetView
   :members:
   :show-inheritance:


Splitting
---------

.. autoclass:: SimilarityCondition
   :members:
   :show-inheritance:


Scaling
-------

.. autoclass:: Scaler
   :members:
   :show-inheritance:

.. autodata:: supported_scalers
