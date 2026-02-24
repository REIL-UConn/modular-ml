"""Cross-validation result containers and helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, TypeVar

from modularml.core.experiment.results.group_results import PhaseGroupResults
from modularml.core.experiment.results.train_results import TrainResults
from modularml.utils.data.multi_keyed_data import AxisSeries

if TYPE_CHECKING:
    from collections.abc import Callable

    from modularml.core.topology.graph_node import GraphNode
    from modularml.core.training.loss_record import LossCollection

T = TypeVar("T")


@dataclass
class CVResults(PhaseGroupResults):
    """
    Results container for cross-validation.

    Description:
        :class:`CVResults` extends :class:`PhaseGroupResults` to provide
        cross-fold querying. Each top-level entry is a fold's
        :class:`PhaseGroupResults` containing :class:`TrainResults`,
        :class:`EvalResults`, etc.

    Structure:
        ```
        CVResults(label='CV')
        ├── fold_0: PhaseGroupResults
        │     ├── train: TrainResults
        │     └── eval: EvalResults
        ├── fold_1: PhaseGroupResults
        ...
        ```

        The :meth:`collect` method applies an extractor to each fold and
        merges results into a single :class:`AxisSeries` with a `fold` axis,
        enabling cross-fold filtering and aggregation via the standard
        :class:`AxisSeries` API (:meth:`AxisSeries.where`,
        :meth:`AxisSeries.collapse`, :meth:`AxisSeries.at`).

    Example:
        Accessing CVResults after a CrossValidation run:

        >>> cv_results = cv.run()  # doctest: +SKIP
        >>> # Cross-fold epoch losses (convenience method)
        >>> losses = cv_results.epoch_losses(node="output")  # doctest: +SKIP
        >>> losses.where(epoch=3)  # all folds at epoch 3 # doctest: +SKIP
        >>> losses.collapse(  # doctest: +SKIP
        ...     "fold", reducer="mean"
        ... )  # mean across folds

        >>> # Generic collect
        >>> cv_results.collect(  # doctest: +SKIP
        ...     lambda fold: fold.get_eval_result("eval").aggregated_losses(
        ...         node="output"
        ...     )
        ... )

    """

    # ================================================
    # Representation
    # ================================================
    def __repr__(self):
        return f"CVResults(label='{self.label}', n_folds={self.n_folds})"

    # ================================================
    # Properties
    # ================================================
    @property
    def n_folds(self) -> int:
        """Number of folds."""
        return len(self._results)

    @property
    def fold_labels(self) -> list[str]:
        """Fold labels in execution order."""
        return list(self._results.keys())

    # ================================================
    # Fold Access
    # ================================================
    def get_fold(self, fold: int | str) -> PhaseGroupResults:
        """
        Get results for a specific fold.

        Args:
            fold (int | str):
                Fold index (int, converted to `fold_{i}`) or fold label (str).

        Returns:
            PhaseGroupResults: The results for the specified fold.

        Raises:
            KeyError: If no fold exists with the given label.
            TypeError: If the result is not a :class:`PhaseGroupResults`.

        """
        if isinstance(fold, int):
            fold = f"fold_{fold}"
        return self.get_group_result(fold)

    # ================================================
    # Cross-Fold Querying
    # ================================================
    def collect(
        self,
        extractor: Callable[[PhaseGroupResults], AxisSeries[T] | T],
    ) -> AxisSeries[T]:
        """
        Apply an extractor to each fold and merge into one :class:`AxisSeries`.

        Description:
            The extractor receives each fold's :class:`PhaseGroupResults`. The return
            value determines how results are keyed:

            - If the extractor returns an :class:`AxisSeries`, its axes are
              preserved and `fold` is prepended as the first axis.
              For example, an :class:`AxisSeries` of :class:`LossCollection`
              `(epoch,)` becomes keyed by `(fold, epoch)`.

            - If the extractor returns a **scalar value**, the result is
              an :class:`AxisSeries` keyed by `(fold,)` only.

        Args:
            extractor (Callable[[PhaseGroupResults], AxisSeries[T] | T]):
                Function that extracts data from a single fold's results.

        Returns:
            AxisSeries[T]:
                Merged results with `fold` as the first axis.

        """
        fold_results: dict[str, AxisSeries[T] | T] = {}
        for fold_label in self.fold_labels:
            fold = self.get_fold(fold_label)
            fold_results[fold_label] = extractor(fold)

        # Detect whether results are AxisSeries or scalar
        first = next(iter(fold_results.values()))

        if isinstance(first, AxisSeries):
            # Flatten: prepend fold axis to each entry's key
            merged_axes = ("fold", *first.axes)
            merged_data = {}
            for fold_label, series in fold_results.items():
                for key, value in series._data.items():
                    merged_data[(fold_label, *key)] = value
            return AxisSeries(axes=merged_axes, _data=merged_data)

        # Scalar: wrap in AxisSeries keyed by fold
        scalar_data = {(fold_label,): val for fold_label, val in fold_results.items()}
        return AxisSeries(axes=("fold",), _data=scalar_data)

    # ================================================
    # Convenience Methods
    # ================================================
    def _resolve_train_phase(self, phase: str | None) -> str:
        """
        Resolve a train phase label from the first fold.

        If `phase` is None, auto-detects the single :class:`TrainResults` in the
        first fold. Raises if ambiguous.

        """
        first_fold = self.get_fold(fold=self.fold_labels[0])
        if phase is not None:
            first_fold.get_train_result(phase)
            return phase

        # Auto-detect (only works if single train phase)
        train_labels = [
            lbl
            for lbl, res in first_fold.phase_results.items()
            if isinstance(res, TrainResults)
        ]

        if len(train_labels) == 0:
            msg = "No TrainResults found in fold results."
            raise ValueError(msg)
        if len(train_labels) > 1:
            msg = (
                f"Multiple TrainResults found: {train_labels}. "
                "Specify which one with the `phase` argument."
            )
            raise ValueError(msg)

        return train_labels[0]

    def epoch_losses(
        self,
        node: str | GraphNode,
        *,
        phase: str | None = None,
        reducer: Literal["sum", "mean"] = "mean",
    ) -> AxisSeries[LossCollection]:
        """
        Training losses per fold per epoch.

        Args:
            node (str | GraphNode):
                The node to retrieve losses for.
            phase (str | None, optional):
                Training phase label. If None and only one :class:`TrainResults`
                exists per fold, it is auto-detected. Defaults to None.
            reducer (Literal["sum", "mean"], optional):
                How to aggregate losses within each epoch.
                Defaults to "mean".

        Returns:
            AxisSeries[LossCollection]:
                Losses keyed by `(fold, epoch)`.

        """
        phase_label = self._resolve_train_phase(phase)
        return self.collect(
            lambda fold, _p=phase_label, _n=node, _r=reducer: (
                fold.get_train_result(_p).epoch_losses(node=_n, reducer=_r)
            ),
        )

    def validation_losses(
        self,
        node: str | GraphNode,
        *,
        phase: str | None = None,
        label: str | None = None,
        reducer: Literal["sum", "mean"] = "mean",
    ) -> AxisSeries[LossCollection]:
        """
        Validation losses per fold per epoch.

        Args:
            node (str | GraphNode):
                The node to retrieve validation losses for.
            phase (str | None, optional):
                Training phase label. Auto-detected if omitted.
            label (str | None, optional):
                Validation callback label. Required if multiple validation
                callbacks exist.
            reducer (Literal["sum", "mean"], optional):
                How to aggregate losses across validation batches.
                Defaults to "mean".

        Returns:
            AxisSeries[LossCollection]:
                Validation losses keyed by `(fold, epoch)`.

        """
        phase_label = self._resolve_train_phase(phase)
        return self.collect(
            lambda fold, _p=phase_label, _n=node, _l=label, _r=reducer: (
                fold.get_train_result(_p).validation_losses(
                    node=_n,
                    label=_l,
                    reducer=_r,
                )
            ),
        )
