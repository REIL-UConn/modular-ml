"""ArtifactResult callback result for tracking rich non-scalar artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

from modularml.core.experiment.callbacks.callback_result import CallbackResult


@dataclass
class ArtifactResult(CallbackResult):
    """
    A callback result carrying a named artifact (figure, table, array, etc.).

    When a callback returns an ``ArtifactResult``, the framework stores the
    artifact in the phase's :class:`ArtifactStore`. Unlike
    :class:`~modularml.callbacks.metric.MetricResult`, ``ArtifactResult``
    accepts any Python object: matplotlib figures, pandas DataFrames,
    numpy arrays, strings, etc.

    If the experiment is configured with an artifact directory via
    :class:`~modularml.core.experiment.results.results_config.ResultsConfig`,
    the artifact is serialized to disk automatically.

    Attributes:
        artifact_name (str):
            Stable name for this artifact (e.g. ``"val_scatter"``).
        artifact (Any):
            The artifact object to store.

    Example:
        ```python
        from modularml.callbacks.artifact_result import ArtifactResult


        class ScatterPlotCallback(Callback):
            def on_epoch_end(self, *, experiment, phase, exec_ctx, results=None):
                fig = make_scatter_plot(...)
                return ArtifactResult(
                    artifact_name="val_scatter",
                    artifact=fig,
                )
        ```

    """

    kind: ClassVar[str] = "artifact"

    artifact_name: str = ""
    artifact: Any = None
