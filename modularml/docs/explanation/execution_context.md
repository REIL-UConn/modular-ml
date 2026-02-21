# Execution Context

Clarify what the execution context stores (resolved nodes, FeatureSet batches, outputs, intermediate artifacts) and how it powers :mod:`modularml.core.execution`.

## Outline

- Enumerate key APIs on :class:`~modularml.core.data.execution_context.ExecutionContext`.
- Show how contexts surface inside callbacks, losses, and trainers.
- Reference tutorials for `06_checkpointing` and `08_ensemble_modeling` where state handling is critical.
