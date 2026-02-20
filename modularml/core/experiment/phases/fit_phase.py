"""Fit-phase implementation for batch-fit scikit-learn models."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from modularml.core.data.batch_view import BatchView
from modularml.core.data.execution_context import ExecutionContext
from modularml.core.data.featureset_view import FeatureSetView
from modularml.core.data.schema_constants import ROLE_DEFAULT
from modularml.core.experiment.callbacks.callback import Callback
from modularml.core.experiment.experiment_context import ExperimentContext
from modularml.core.experiment.phases.phase import ExperimentPhase, InputBinding
from modularml.core.training.applied_loss import AppliedLoss

if TYPE_CHECKING:
    from collections.abc import Iterator

    from modularml.core.data.featureset import FeatureSet
    from modularml.core.experiment.results.fit_results import FitResults
    from modularml.core.references.featureset_reference import FeatureSetReference
    from modularml.core.topology.graph_node import GraphNode


class FitPhase(ExperimentPhase):
    """
    Phase that fits batch-fit model nodes on the complete dataset.

    Description:
        FitPhase is designed for scikit-learn models (and similar) that require
        all training data at once via `.fit(X, y)` rather than iterative
        mini-batch gradient updates.

        Unlike TrainPhase, FitPhase has no epochs or sampling. It yields a
        single ExecutionContext containing the entire dataset from the specified
        split(s).

        By default, fitted nodes are frozen after fitting so that downstream
        gradient-trained nodes can use their outputs without interference.

    """

    def __init__(
        self,
        label: str,
        *,
        input_sources: list[InputBinding],
        losses: list[AppliedLoss] | None = None,
        active_nodes: list[GraphNode] | None = None,
        freeze_after_fit: bool = True,
        callbacks: list[Callback] | None = None,
    ):
        """
        Initialize a new fit phase for the experiment.

        Notes:
            All `input_sources` must originate from the same upstream FeatureSet.
            If multiple FeatureSets need to be fitted, they must be done so in
            separate FitPhases.

        Args:
            label (str):
                A label to assign to this phase of the experiment. Used for logging.

            input_sources (list[InputBinding]):
                Input bindings for each head node in ModelGraph. All bindings must
                resolve to the same FeatureSet.

            losses (list[AppliedLoss], optional):
                A list of losses to compute after fitting (for metrics only).

            active_nodes (list[GraphNode] | None, optional):
                A list of GraphNodes to fit. Nodes can be listed via their ID,
                label, or with the actual node instance. If None, all nodes
                comprising the ModelGraph are used. Defaults to None.

            freeze_after_fit (bool, optional):
                Whether to freeze fitted nodes after `.fit()` completes.
                Defaults to True.

            callbacks (list[Callback] | None, optional):
                An optional list of Callbacks to run during phase execution.

        """
        super().__init__(
            label=label,
            input_sources=input_sources,
            losses=losses,
            active_nodes=active_nodes,
            callbacks=callbacks,
        )
        self.freeze_after_fit = freeze_after_fit
        self._inp_fsv: FeatureSetView | None = None
        self._validate_single_featureset()

    # ================================================
    # Convenience Constructors
    # ================================================
    @classmethod
    def from_split(
        cls,
        label: str,
        *,
        split: str,
        losses: list[AppliedLoss] | None = None,
        active_nodes: list[GraphNode] | None = None,
        freeze_after_fit: bool = True,
        callbacks: list[Callback] | None = None,
    ) -> FitPhase:
        """
        Initialize a new fit phase for a given FeatureSet split.

        Notes:
            All active head nodes must input from the defined split.

        Args:
            label (str):
                A label to assign to this phase of the experiment. Used for logging.

            split (str):
                The FeatureSet split name to fit on (e.g., "train").

            losses (list[AppliedLoss], optional):
                A list of losses to compute after fitting (for metrics only).

            active_nodes (list[GraphNode] | None, optional):
                A list of GraphNodes to fit. Nodes can be listed via their ID,
                label, or with the actual node instance. If None, all nodes
                comprising the ModelGraph are used. Defaults to None.

            freeze_after_fit (bool, optional):
                Whether to freeze fitted nodes after `.fit()` completes.
                Defaults to True.

            callbacks (list[Callback] | None, optional):
                An optional list of Callbacks to run during phase execution.

        """
        input_sources = cls._build_input_sources_from_split(
            split=split,
            sampler=None,
            active_nodes=active_nodes,
        )
        return cls(
            label=label,
            input_sources=input_sources,
            losses=losses,
            active_nodes=active_nodes,
            freeze_after_fit=freeze_after_fit,
            callbacks=callbacks,
        )

    # ================================================
    # Validation
    # ================================================
    def _validate_single_featureset(self):
        """Ensure all input sources originate from same FeatureSet (and split)."""
        fs_node_ids = {binding.upstream_ref.node_id for binding in self.input_sources}
        if len(fs_node_ids) > 1:
            fs_lbls = {
                binding.upstream_ref.node_label for binding in self.input_sources
            }
            msg = (
                "All `input_sources` of a FitPhase must resolve to a single upstream "
                f"FeatureSet. Detected multiple: {fs_lbls}."
            )
            raise ValueError(msg)

        fs_splits: set[str | None] = {binding.split for binding in self.input_sources}
        if len(fs_splits) > 1:
            msg = (
                "All `input_sources` of a FitPhase must resolve to the same split of "
                f"the same FeatureSet. Detected multiple splits: {fs_splits}."
            )
            raise ValueError(msg)

        # Convert this FeatureSet + split to a view
        fs: FeatureSet = ExperimentContext.get_active().get_node(
            node_id=next(iter(fs_node_ids)),
            enforce_type="FeatureSet",
        )
        split = next(iter(fs_splits))
        if split is None:
            self._inp_fsv = fs.to_view()
        else:
            self._inp_fsv = fs.get_split(split_name=split)

    # ================================================
    # Representation
    # ================================================
    def __repr__(self):
        return f"FitPhase(label='{self.label}')"

    # ================================================
    # Execution
    # ================================================
    def iter_execution(
        self,
        *,
        results: FitResults | None = None,
    ) -> Iterator[ExecutionContext]:
        """
        Iterate over execution steps for this fit phase.

        Description:
            Generates a single ExecutionContext containing the entire dataset
            from the specified split. No batching or epochs are used.

        Args:
            results (FitResults | None, optional):
                Optional container in which results will be registered.

        Yields:
            ExecutionContext:
                A single execution context containing all data, suitable for
                `ModelGraph.fit_step(ctx)`.

        """
        # Validate input view
        if not isinstance(self._inp_fsv, FeatureSetView):
            msg = f"Failed to resolve input view for FitPhase '{self.label}'."
            raise TypeError(msg)

        n = len(self._inp_fsv)
        if n == 0:
            msg = f"FitPhase '{self.label}' has no samples in view '{self._inp_fsv!r}'."
            raise RuntimeError(msg)

        # ------------------------------------------------
        # Callbacks: on_phase_start
        # ------------------------------------------------
        exp_ctx = ExperimentContext.get_active()
        experiment = exp_ctx.get_experiment()

        experiment._in_callback = True
        try:
            for cb in self.callbacks:
                cb._on_phase_start(
                    experiment=experiment,
                    phase=self,
                    results=results,
                )
        finally:
            experiment._in_callback = False

        try:
            # Create a single BatchView over the entire dataset
            bv = BatchView(
                source=self._inp_fsv.source,
                role_indices={ROLE_DEFAULT: self._inp_fsv.indices},
            )
            inputs: dict[tuple[str, FeatureSetReference], BatchView] = {
                (binding.node_id, binding.upstream_ref): bv
                for binding in self.input_sources
            }
            exec_ctx = ExecutionContext(
                phase_label=self.label,
                epoch_idx=0,
                batch_idx=0,
                inputs=inputs,
            )

            # ------------------------------------------------
            # Callbacks: on_batch_start
            # ------------------------------------------------
            experiment._in_callback = True
            try:
                for cb in self.callbacks:
                    cb._on_batch_start(
                        experiment=experiment,
                        phase=self,
                        exec_ctx=exec_ctx,
                        results=results,
                    )
            finally:
                experiment._in_callback = False

            yield exec_ctx

            # ------------------------------------------------
            # Callbacks: on_batch_end
            # ------------------------------------------------
            experiment._in_callback = True
            try:
                for cb in self.callbacks:
                    cb._on_batch_end(
                        experiment=experiment,
                        phase=self,
                        exec_ctx=exec_ctx,
                        results=results,
                    )
            finally:
                experiment._in_callback = False

            # ------------------------------------------------
            # Callbacks: on_phase_end
            # ------------------------------------------------
            for cb in self.callbacks:
                cb._on_phase_end(
                    experiment=experiment,
                    phase=self,
                    results=results,
                )

        except BaseException as exc:
            experiment._in_callback = True
            try:
                for cb in self.callbacks:
                    cb._on_exception(
                        experiment=experiment,
                        phase=self,
                        exec_ctx=None,
                        exception=exc,
                        results=results,
                    )
            finally:
                experiment._in_callback = False
            raise

    # ================================================
    # Configurable
    # ================================================
    def get_config(self) -> dict[str, Any]:
        """
        Return configuration details required to reconstruct this phase.

        Returns:
            dict[str, Any]:
                Configuration used to reconstruct the phase.

        """
        cfg = super().get_config()
        cfg.update(
            {
                "phase_type": "FitPhase",
                "freeze_after_fit": self.freeze_after_fit,
            },
        )
        return cfg

    @classmethod
    def from_config(cls, config: dict) -> FitPhase:
        """
        Construct a FitPhase from a configuration dictionary.

        Args:
            config (dict[str, Any]):
                Configuration details. Keys must be strings.

        Returns:
            FitPhase: Reconstructed phase.

        """
        if "phase_type" not in config:
            raise ValueError("FitPhase config must include `phase_type`")
        if config["phase_type"] != "FitPhase":
            msg = (
                "Invalid config for FitPhase. Received config for: "
                f"{config['phase_type']}"
            )
            raise ValueError(msg)

        losses = None
        if config["losses"] is not None:
            losses = [AppliedLoss.from_config(cfg) for cfg in config["losses"]]
        return cls(
            label=config["label"],
            input_sources=[
                InputBinding.from_config(cfg) for cfg in config["input_sources"]
            ],
            losses=losses,
            active_nodes=config["active_nodes"],
            freeze_after_fit=config.get("freeze_after_fit", True),
            callbacks=[Callback.from_config(cfg) for cfg in config["callbacks"]],
        )
