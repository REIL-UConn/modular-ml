"""Evaluation-phase implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from modularml.core.data.batch_view import BatchView
from modularml.core.data.execution_context import ExecutionContext
from modularml.core.data.featureset_view import FeatureSetView
from modularml.core.data.schema_constants import ROLE_DEFAULT
from modularml.core.experiment.callbacks.callback import Callback
from modularml.core.experiment.experiment_context import ExperimentContext
from modularml.core.experiment.phases.phase import ExperimentPhase, InputBinding
from modularml.core.training.applied_loss import AppliedLoss
from modularml.utils.environment.environment import IN_NOTEBOOK
from modularml.utils.progress_bars.progress_task import ProgressTask

if TYPE_CHECKING:
    from collections.abc import Iterator

    from modularml.core.data.featureset import FeatureSet
    from modularml.core.experiment.results.eval_results import EvalResults
    from modularml.core.references.featureset_reference import FeatureSetReference
    from modularml.core.topology.graph_node import GraphNode


class EvalPhase(ExperimentPhase):
    """Phase that evaluates model outputs on a fixed FeatureSet split."""

    def __init__(
        self,
        label: str,
        *,
        input_sources: list[InputBinding],
        losses: list[AppliedLoss] | None = None,
        active_nodes: list[GraphNode] | None = None,
        batch_size: int | None = None,
        callbacks: list[Callback] | None = None,
    ):
        """
        Initiallizes a new evaluation phase for the experiment.

        Notes:
            All `input_sources` must originate from the same upstream FeatureSet.
            If multiple FeatureSets need to be evaluated, they must be done so in
            separate EvalPhases.

        Args:
            label (str):
                A label to assign to this phase of the experiment. Used for logging.

            input_sources (list[InputBinding]):
                Input bindings for each head node in ModelGraph. All bindings must
                resolve to the same FeatureSet.

            losses (list[AppliedLoss], optional):
                A list of losses to be applied during this evaluation phase.

            active_nodes (list[GraphNode] | None, optional):
                A list of GraphNodes to run a forward phase on. Nodes can
                be listed via their ID, label, or with the actual node instance. If
                None, all nodes comprising the ModelGraph are used. Defaults to None.

            batch_size (int, optional):
                If defined, limits the number of samples to using during a single
                forward pass. Otherwise, all samples are passed at once.

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
        self.batch_size = batch_size
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
        batch_size: int | None = None,
        callbacks: list[Callback] | None = None,
    ) -> EvalPhase:
        """
        Initiallizes a new evaluation phase for a given FeatureSet split.

        Notes:
            All active head nodes must input from the defined split.

        Args:
            label (str):
                A label to assign to this phase of the experiment. Used for logging.

            split (str):
                The FeatureSet split name to evaluate.

            losses (list[AppliedLoss], optional):
                A list of losses to be applied during this evaluation phase.

            active_nodes (list[GraphNode] | None, optional):
                A list of GraphNodes to run a forward phase on. Nodes can
                be listed via their ID, label, or with the actual node instance. If
                None, all nodes comprising the ModelGraph are used. Defaults to None.

            batch_size (int, optional):
                If defined, limits the number of samples to using during a single
                forward pass. Otherwise, all samples are passed at once.

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
            batch_size=batch_size,
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
                "All `input_sources` of an EvalPhase must resolve to a single upstream "
                f"FeatureSet. Detected multiple: {fs_lbls}."
            )
            raise ValueError(msg)

        fs_splits: set[str | None] = {binding.split for binding in self.input_sources}
        if len(fs_splits) > 1:
            msg = (
                "All `input_sources` of an EvalPhase must resolve to the same split of "
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
        return f"EvalPhase(label='{self.label}')"

    # ================================================
    # Execution
    # ================================================
    def iter_execution(
        self,
        *,
        results: EvalResults | None = None,
        show_eval_progress: bool = False,
        persist_progress: bool = IN_NOTEBOOK,
    ) -> Iterator[ExecutionContext]:
        """
        Iterate over execution steps for this evaluation phase.

        Description:
            Generates ExecutionContext objects that cover the specified split of the
            upstream FeatureSet. The split is optionally chunked into batches based
            on `batch_size` to limit memory usage.

        Args:
            results (EvalResults | None, optional):
                Optional container in which results will be registered.

            show_eval_progress (bool, optional):
                Whether to show a progress bar for eval batches. Defaults to False.

            persist_progress (bool, optional):
                Whether to leave all eval progress bars shown after they complete.
                Defaults to `IN_NOTEBOOK` (True if working in a notebook, False if in
                a Python script).

        Yields:
            ExecutionContext:
                Execution contexts suitable for forward-only execution via
                `ModelGraph.eval_step(ctx)`.

        """
        # Validate input view
        if not isinstance(self._inp_fsv, FeatureSetView):
            msg = f"Failed to resolve input view for EvaluationPhase '{self.label}'."
            raise TypeError(msg)

        # Determine max number of samples in a execution context
        n = len(self._inp_fsv)
        if n == 0:
            msg = (
                f"EvalPhase '{self.label}' has no samples in view '{self._inp_fsv!r}'."
            )
            raise RuntimeError(msg)
        batch_size = self.batch_size if self.batch_size is not None else n
        n_batches = int(n // batch_size)
        if (n / batch_size) - n_batches > 0:
            n_batches += 1

        # ------------------------------------------------
        # Progress Bar: batches
        # ------------------------------------------------
        eval_ptask = ProgressTask(
            style="evaluation",
            description=f"Evaluating ['{self.label}']",
            total=n_batches,
            enabled=show_eval_progress,
            persist=persist_progress,
        )
        eval_ptask.start()

        # ------------------------------------------------
        # Callbacks: on_phase_start
        # ------------------------------------------------
        exp_ctx = ExperimentContext.get_active()
        experiment = exp_ctx.get_experiment()
        last_ctx: ExecutionContext | None = None

        try:
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

            # ------------------------------------------------
            # Iterate over all batches
            # ------------------------------------------------
            for i in range(n_batches):
                start = i * batch_size
                end = min((i + 1) * batch_size, n)
                fsv_batch = self._inp_fsv.take(np.arange(start, end, 1))

                bv = BatchView(
                    source=fsv_batch.source,
                    role_indices={ROLE_DEFAULT: fsv_batch.indices},
                )
                inputs: dict[tuple[str, FeatureSetReference], BatchView] = {
                    (binding.node_id, binding.upstream_ref): bv
                    for binding in self.input_sources
                }
                exec_ctx = ExecutionContext(
                    phase_label=self.label,
                    epoch_idx=0,
                    batch_idx=i,
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

                eval_ptask.tick(n=1)

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
                        exec_ctx=last_ctx,
                        exception=exc,
                        results=results,
                    )
            finally:
                experiment._in_callback = False
            raise

        # Finish progress bar
        eval_ptask.finish()

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
                "phase_type": "EvalPhase",
                "batch_size": self.batch_size,
            },
        )
        return cfg

    @classmethod
    def from_config(cls, config: dict) -> EvalPhase:
        """
        Construct a phase from a configuration dictionary.

        Args:
            config (dict[str, Any]):
                Configuration details. Keys must be strings.

        Returns:
            ExperimentPhase: Reconstructed phase.

        """
        if "phase_type" not in config:
            raise ValueError("EvalPhase config must include `phase_type`")
        if config["phase_type"] != "EvalPhase":
            msg = (
                "Invalid config for EvalPhase. Received config for: "
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
            batch_size=config["batch_size"],
            callbacks=[Callback.from_config(cfg) for cfg in config["callbacks"]],
        )
