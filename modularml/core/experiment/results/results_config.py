"""Configuration for experiment results storage locations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class ResultsConfig:
    """
    Controls where experiment result stores are saved on disk.

    By default all results are kept in memory. Set ``results_dir`` to a
    directory path to enable on-disk storage. Use the boolean flags to
    choose which of the three result stores are persisted to disk; the
    remaining stores are kept in memory but are still accessible via the
    normal results API.

    On disk, each phase produces the following layout::

        results_dir/                         # must be empty or non-existent
          {run_idx}_{phase_label}/
            execution_data/   # ExecutionContext pickle files (if save_execution)
              0.pkl, 1.pkl, ...
            metrics/          # MetricEntry pickle files     (if save_metrics)
              0.pkl, 1.pkl, ...
            artifacts/        # Artifact pickle files        (if save_artifacts)
              0.pkl, 1.pkl, ...
            callbacks/        # CallbackResult pickle files  (if save_execution)
              {callback_label}/
                0.pkl, 1.pkl, ...

    Attributes:
        results_dir (Path | None):
            Root directory for on-disk storage. ``None`` means in-memory
            (the default).
        save_execution (bool):
            Whether to persist
            :class:`~modularml.core.experiment.results.execution_store.ExecutionStore`
            entries to disk. Defaults to ``True``.
        save_metrics (bool):
            Whether to persist
            :class:`~modularml.core.experiment.results.metric_store.MetricStore`
            entries to disk. Scalar metrics are small and usually fine in
            memory; set to ``True`` to offload them. Defaults to ``False``.
        save_artifacts (bool):
            Whether to persist
            :class:`~modularml.core.experiment.results.artifact_store.ArtifactStore`
            entries to disk. Defaults to ``True``.

    Example:
        Save all result stores under a single directory:

        ```python
            ResultsConfig(results_dir=Path("./exp_results"))
        ```

        Save only artifacts and execution contexts, keep metrics in memory:

        ```python
            ResultsConfig(
                results_dir=Path("./exp_results"),
                save_metrics=False,
            )
        ```

        Save everything including scalar metrics:

        ```python
            ResultsConfig(
                results_dir=Path("./exp_results"),
                save_execution=True,
                save_metrics=True,
                save_artifacts=True,
            )
        ```

    """

    results_dir: Path | None = None
    save_execution: bool = True
    save_metrics: bool = False
    save_artifacts: bool = True

    def __post_init__(self) -> None:
        if self.results_dir is not None:
            p = Path(self.results_dir)
            if p.exists() and any(p.iterdir()):
                msg = (
                    f"`results_dir` must be empty or non-existent at the start of an "
                    f"experiment, but {p!r} already contains files."
                )
                raise ValueError(msg)

    def phase_dir(self, phase_suffix: str | Path) -> Path | None:
        """
        Return the on-disk directory for a given phase suffix.

        Args:
            phase_suffix (str | Path):
                Relative path component identifying the phase, e.g.
                ``"0_training"`` or ``Path("cv_group") / "fold_0"``.

        Returns:
            Path | None: Absolute phase directory, or ``None`` if
            ``results_dir`` is not set.

        """
        if self.results_dir is None:
            return None
        return Path(self.results_dir) / phase_suffix

    def get_config(self) -> dict:
        """Return configuration details required to reconstruct this object."""
        return {
            "results_dir": str(self.results_dir)
            if self.results_dir is not None
            else None,
            "save_execution": self.save_execution,
            "save_metrics": self.save_metrics,
            "save_artifacts": self.save_artifacts,
        }

    @classmethod
    def from_config(cls, config: dict) -> ResultsConfig:
        """
        Construct from config data.

        Args:
            config (dict): Serialized configuration produced by :meth:`get_config`.

        Returns:
            ResultsConfig: Reconstructed config.

        """
        results_dir = config.get("results_dir")
        return cls(
            results_dir=Path(results_dir) if results_dir is not None else None,
            save_execution=config.get("save_execution", True),
            save_metrics=config.get("save_metrics", False),
            save_artifacts=config.get("save_artifacts", True),
        )
