"""Results tree mirroring :class:`PhaseGroup` execution structure."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from modularml.core.experiment.results.eval_results import EvalResults
from modularml.core.experiment.results.phase_results import PhaseResults
from modularml.core.experiment.results.train_results import TrainResults

if TYPE_CHECKING:
    from collections.abc import Iterator


@dataclass
class PhaseGroupResults:
    """
    Hierarchical results container matching the structure of a PhaseGroup.

    Description:
        PhaseGroupResults stores PhaseResults and nested PhaseGroupResults in
        the same hierarchy as the PhaseGroup that produced them. Results are
        stored in insertion order matching execution order.

        Provides convenience methods for:

        - Accessing results by phase or group label
        - Flattening nested results into a single-level mapping
        - Iterating over results in execution order

    Attributes:
        label (str): Phase-group label associated with this result tree.
        _results (dict[str, PhaseResults | PhaseGroupResults]):
            Ordered mapping of labels to phase or nested group results.

    Example:
        Accessing phase group results

        >>> # Access results by label
        >>> train_res = group_results.get_phase_result("train")  # doctest: +SKIP
        >>> eval_res = group_results.get_phase_result("eval")  # doctest: +SKIP
        >>> # Flatten nested structure
        >>> flat = group_results.flatten()  # doctest: +SKIP
        >>> for label, phase_res in flat.items():  # doctest: +SKIP
        ...     print(f"{label}: {phase_res!r}")
        >>> # Iterate in execution order
        >>> for label, result in group_results.items():  # doctest: +SKIP
        ...     print(label, type(result).__name__)

    """

    label: str

    _results: dict[str, PhaseResults | PhaseGroupResults] = field(
        default_factory=dict,
    )

    def __repr__(self):
        entries = ", ".join(
            f"'{k}': {type(v).__name__}" for k, v in self._results.items()
        )
        return f"PhaseGroupResults(label='{self.label}', results={{{entries}}})"

    # ================================================
    # Runtime Modifiers
    # ================================================
    def add_result(self, result: PhaseResults | PhaseGroupResults):
        """
        Record a phase or group result.

        Args:
            result (PhaseResults | PhaseGroupResults):
                The result to add. Must have a unique label within this group.

        Raises:
            TypeError:
                If `result` is not a PhaseResults or PhaseGroupResults.
            ValueError:
                If a result with the same label already exists.

        """
        if not isinstance(result, (PhaseResults, PhaseGroupResults)):
            msg = (
                f"Expected PhaseResults or PhaseGroupResults. Received: {type(result)}."
            )
            raise TypeError(msg)

        if result.label in self._results:
            msg = f"A result with label '{result.label}' already exists in this group."
            raise ValueError(msg)
        self._results[result.label] = result

    # ================================================
    # Properties
    # ================================================
    @property
    def labels(self) -> list[str]:
        """
        All top-level result labels in insertion order.

        Returns:
            list[str]: Ordered labels.

        """
        return list(self._results.keys())

    @property
    def phase_results(self) -> dict[str, PhaseResults]:
        """
        Only the top-level PhaseResults entries, keyed by label.

        Description:
            Returns only the PhaseResults (not nested PhaseGroupResults)
            at this level of the hierarchy. The returned dict does not
            encode execution order.

        """
        return {k: v for k, v in self._results.items() if isinstance(v, PhaseResults)}

    @property
    def group_results(self) -> dict[str, PhaseGroupResults]:
        """
        Only the top-level PhaseGroupResults entries, keyed by label.

        Description:
            Returns only the nested PhaseGroupResults (not PhaseResults)
            at this level of the hierarchy. The returned dict does not
            encode execution order.

        """
        return {
            k: v for k, v in self._results.items() if isinstance(v, PhaseGroupResults)
        }

    # ================================================
    # Accessors
    # ================================================
    def _resolve_single_phase(self, phase: str | None, req_cls: type) -> str:
        """
        Resolve a phase label for results of type `req_cls`.

        If `phase` is None, auto-detects the single phase in the top
        level of the group. Raises if ambiguous.

        """
        if phase is not None:
            if phase not in self._results:
                msg = f"No phase exists with label '{phase}'."
                raise ValueError(msg)
            return phase

        # Auto-detect (only works if single train phase)
        avail_lbls = [lbl for lbl, res in self.items() if isinstance(res, req_cls)]
        if len(avail_lbls) == 0:
            msg = f"No {req_cls.__qualname__} found in this group."
            raise ValueError(msg)
        if len(avail_lbls) > 1:
            msg = (
                f"Multiple {req_cls.__qualname__} found: {avail_lbls}. "
                "Specify which one with the `phase` argument."
            )
            raise ValueError(msg)

        return avail_lbls[0]

    def __getitem__(self, key: str) -> PhaseResults | PhaseGroupResults:
        """
        Retrieve a result by its label.

        Args:
            key (str):
                The label of the phase or group result.

        Returns:
            PhaseResults | PhaseGroupResults:
                The result for the given label.

        Raises:
            KeyError: If no result exists with the given label.

        """
        if key not in self._results:
            msg = f"No result exists with label '{key}'. Available: {self.labels}."
            raise KeyError(msg)
        return self._results[key]

    def __contains__(self, key: str) -> bool:
        """Check if a result exists with the given label."""
        return key in self._results

    def __len__(self) -> int:
        """Number of top-level results in this group."""
        return len(self._results)

    def items(self) -> Iterator[tuple[str, PhaseResults | PhaseGroupResults]]:
        """
        Iterate over label-result pairs in execution order.

        Returns:
            Iterator[tuple[str, PhaseResults | PhaseGroupResults]]:
                Iterator over label/result pairs.

        """
        yield from self._results.items()

    def get_phase_result(self, label: str) -> PhaseResults:
        """
        Retrieve a PhaseResults by its label.

        Args:
            label (str):
                The phase label to look up.

        Returns:
            PhaseResults: The results for the specified phase.

        Raises:
            KeyError: If no result exists with the given label.
            TypeError: If the result is not of type PhaseResults.

        """
        result = self[label]
        if not isinstance(result, PhaseResults):
            msg = (
                f"Result with label '{label}' is a "
                f"{type(result).__name__}, not PhaseResults."
            )
            raise TypeError(msg)
        return result

    def get_train_result(self, label: str | None = None) -> TrainResults:
        """
        Retrieve a TrainResults by its label.

        Args:
            label (str, optional):
                The training phase label to look up.
                Auto-detected if omitted. Defaults to None.

        Returns:
            TrainResults: The results for the specified phase.

        Raises:
            KeyError: If no result exists with the given label.
            TypeError: If the result is not of type TrainResults.

        """
        phase_lbl = self._resolve_single_phase(
            phase=label,
            req_cls=TrainResults,
        )
        result = self[phase_lbl]
        if not isinstance(result, TrainResults):
            msg = (
                f"Result with label '{label}' is a "
                f"{type(result).__name__}, not TrainResults."
            )
            raise TypeError(msg)
        return result

    def get_eval_result(self, label: str | None = None) -> EvalResults:
        """
        Retrieve a EvalResults by its label.

        Args:
            label (str):
                The evaluation phase label to look up.
                Auto-detected if omitted. Defaults to None.

        Returns:
            EvalResults: The results for the specified phase.

        Raises:
            KeyError: If no result exists with the given label.
            TypeError: If the result is not of type EvalResults.

        """
        phase_lbl = self._resolve_single_phase(
            phase=label,
            req_cls=EvalResults,
        )
        result = self[phase_lbl]
        if not isinstance(result, EvalResults):
            msg = (
                f"Result with label '{label}' is a "
                f"{type(result).__name__}, not EvalResults."
            )
            raise TypeError(msg)
        return result

    def get_group_result(self, label: str | None = None) -> PhaseGroupResults:
        """
        Retrieve a nested PhaseGroupResults by its label.

        Args:
            label (str):
                The group label to look up.
                Auto-detected if omitted. Defaults to None.

        Returns:
            PhaseGroupResults: The results for the specified group.

        Raises:
            KeyError: If no result exists with the given label.
            TypeError: If the result is a PhaseResults, not PhaseGroupResults.

        """
        phase_lbl = self._resolve_single_phase(
            phase=label,
            req_cls=PhaseGroupResults,
        )
        result = self[phase_lbl]
        if not isinstance(result, PhaseGroupResults):
            msg = (
                f"Result with label '{label}' is a "
                f"{type(result).__name__}, not PhaseGroupResults."
            )
            raise TypeError(msg)
        return result

    # ================================================
    # Flattening
    # ================================================
    def flatten(self) -> dict[str, PhaseResults]:
        """
        Flatten all nested groups into a single-level dict.

        Description:
            Recursively unravels the hierarchy of PhaseGroupResults into
            a flat mapping of phase labels to their PhaseResults. The
            returned dict preserves execution order.

            All phase labels must be unique across the entire hierarchy.
            If duplicate labels are found, a ValueError is raised.

        Returns:
            dict[str, PhaseResults]:
                A flat mapping of phase labels to results in execution order.

        Raises:
            ValueError:
                If duplicate phase labels exist across the hierarchy.

        Example:
            All internal PhaseResults can be flattened to a single list:

            >>> flat = group_results.flatten()  # doctest: +SKIP
            >>> for label, phase_res in flat.items():  # doctest: +SKIP
            ...     print(f"{label}: {type(phase_res).__name__}")

        """
        flat: dict[str, PhaseResults] = {}
        duplicates: list[str] = []

        self._collect_flat(into=flat, duplicates=duplicates)

        if duplicates:
            msg = (
                "Cannot flatten PhaseGroupResults; duplicate phase labels "
                f"found across the hierarchy: {duplicates}."
            )
            raise ValueError(msg)

        return flat

    def _collect_flat(
        self,
        *,
        into: dict[str, PhaseResults],
        duplicates: list[str],
    ) -> None:
        """Recursively collect PhaseResults into a flat dict."""
        for label, result in self._results.items():
            if isinstance(result, PhaseGroupResults):
                result._collect_flat(into=into, duplicates=duplicates)
            else:
                if label in into:
                    duplicates.append(label)
                into[label] = result
