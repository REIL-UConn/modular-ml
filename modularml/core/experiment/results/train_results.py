"""Results container for training phases."""

from __future__ import annotations

from dataclasses import dataclass

from modularml.core.experiment.results.phase_results import PhaseResults


@dataclass
class TrainResults(PhaseResults):
    """
    Results container for a training phase.

    Description:
        TrainResults wraps the outputs of a TrainPhase, which executes
        multiple epochs with multiple batches per epoch. This class provides:

        - Access to training data keyed by epoch and batches
        - Direct access to validation losses/tensors from evaluation callbacks
        - Loss aggregation per epoch

        Validation callbacks (kind="evaluation") are automatically detected
        and their results exposed through dedicated accessors.

    Attributes:
        label (str): Phase label.
        _execution (list[ExecutionContext]): Ordered execution contexts.
        _callbacks (list[CallbackResult]): Recorded callback outputs.
        _metrics (MetricStore): Stored scalar metrics.
        _series_cache (dict[tuple, Any]): Cache of memoized AxisSeries queries.

    """

    # ================================================
    # Representation
    # ================================================
    def __repr__(self):
        n_epochs = self.n_epochs if self._execution else 0
        return f"TrainResults(label='{self.label}', epochs={n_epochs})"

    # ================================================
    # Properties
    # ================================================
    @property
    def epoch_indices(self) -> list[int]:
        """
        Sorted list of recorded epoch indices.

        Returns:
            list[int]: Epoch indices in ascending order.

        """
        epoch_vals = self.execution_contexts().axis_values("epoch")
        return sorted(int(e) for e in epoch_vals)

    @property
    def n_epochs(self) -> int:
        """
        The number of epochs executed during training.

        Returns:
            int: Total number of recorded epochs.

        """
        return len(self.epoch_indices)
