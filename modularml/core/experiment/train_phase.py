from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np

from modularml.context.execution_context import ExecutionContext
from modularml.context.experiment_context import ExperimentContext
from modularml.core.data.batch_view import BatchView
from modularml.core.data.featureset_view import FeatureSetView
from modularml.core.experiment.phase import ExperimentPhase, InputBinding
from modularml.utils.environment.environment import IN_NOTEBOOK
from modularml.utils.progress_bars.progress_task import ProgressTask

if TYPE_CHECKING:
    from collections.abc import Iterator

    from numpy.typing import NDArray

    from modularml.core.data.sampled_view import SampledView
    from modularml.core.references.featureset_reference import FeatureSetReference
    from modularml.core.topology.graph_node import GraphNode
    from modularml.core.training.applied_loss import AppliedLoss


class BatchSchedulingPolicy(Enum):
    """
    Defines how batches from multiple samplers are scheduled during training.

    Let samplers produce the following batch sequences:

        S1 = [b1, b2, b3]
        S2 = [c1, c2]

    The available scheduling policies behave as follows:

    ZIP_STRICT:
        Lockstep iteration, stopping when the shortest sampler is exhausted.

        Output:
            (b1, c1), (b2, c2)

        Total steps:
            min(len(S1), len(S2))

    ZIP_CYCLE:
        Lockstep iteration until the longest sampler is exhausted.
        Shorter samplers cycle from the beginning as needed.

        Output:
            (b1, c1), (b2, c2), (b3, c1)

        Total steps:
            max(len(S1), len(S2))

    ALTERNATE_STRICT:
        Alternate one batch at a time from each sampler in round-robin order,
        stopping when any sampler is exhausted.

        Output:
            b1, c1, b2, c2

        Total steps:
            sum(len(Si)) until first sampler is exhausted

    ALTERNATE_CYCLE:
        Alternate one batch at a time from each sampler in round-robin order,
        cycling shorter samplers until the longest sampler is exhausted.

        Output:
            b1, c1, b2, c2, b3, c1

        Total steps:
            sum(max(len(Si)))

    Notes:
        - This policy controls batch ordering only.
        - No semantic alignment between samplers is performed.
        - If semantic alignment is required (e.g. contrastive pairs),
          it must be handled inside the sampler via roles.
        - Sequential training on different samplers should be expressed
          as multiple training phases, not via batch scheduling.

    """

    ZIP_STRICT = "zip_strict"
    ZIP_CYCLE = "zip_cycle"
    ALTERNATE_STRICT = "alternate_strict"
    ALTERNATE_CYCLE = "alternate_cycle"

    @classmethod
    def from_value(cls, value: str | BatchSchedulingPolicy):
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            try:
                return cls(value.lower())
            except ValueError:
                pass
        msg = (
            f"Invalid BatchSchedulingPolicy: {value}. "
            f"Expected one of {[p.value for p in cls]}"
        )
        raise ValueError(msg)


@dataclass(frozen=True)
class SamplerExecutionKey:
    featureset_id: str
    split: str | None
    sampler_cfg: Any | None


@dataclass
class SamplerExecution:
    sampler_id: int
    sampled: SampledView
    bindings: list[InputBinding]


class TrainPhase(ExperimentPhase):
    def __init__(
        self,
        label: str,
        *,
        input_sources: list[InputBinding],
        losses: list[AppliedLoss],
        n_epochs: int = 1,
        active_nodes: list[str | GraphNode] | None = None,
        batch_schedule: BatchSchedulingPolicy | str = BatchSchedulingPolicy.ZIP_STRICT,
        show_sampler_progress: bool = True,
        show_training_progress: bool = True,
    ):
        """
        Initiallizes a new training phase for the experiment.

        Args:
            label (str):
                A label to assign to this phase of the experiment. Used for logging.

            input_sources (list[InputBinding]):
                Input bindings for each head node in ModelGraph.

            losses (list[AppliedLoss]):
                A list of losses to be applied during this training pahse.

            n_epochs (int):
                Number of epochs to perform.

            active_nodes (list[str | GraphNode] | None, optional):
                A list of active GraphNodes in this phase of the experiment. Nodes can
                be listed via their ID, label, or with the actual node instance. If
                None, all nodes comprising the ModelGraph are used. Defaults to None.

            batch_schedule (str | BatchSchedulingPolicy, optional):
                Defines how batches from multiple samplers are scheduled during training.
                This is only relevant if more than one sampler is defined in
                `input_sources`.

                Let samplers `S1` and `S2` produce: `S1 = [b1, b2, b3]` and  `S2 = [c1, c2]`

                The outputs of each policy is given below:

                ```python
                "zip_strict": (b1, c1), (b2, c2)
                "zip_cycle": (b1, c1), (b2, c2), (b3, c1)
                "alternate_strict": b1, c1, b2, c2
                "alternate_cycle": b1, c1, b2, c2, b3, c1
                ```

                See also :class:`BatchSchedulingPolicy`.

            show_sampler_progress (bool, optional):
                Whether to show a progress bar for sampler batching. Defaults to True.

            show_training_progress (bool, optional):
                Whether to show a progress bar for training execution. Defaults to True.

        """
        super().__init__(
            label=label,
            input_sources=input_sources,
            active_nodes=active_nodes,
        )
        self.losses = losses
        self.batch_schedule = BatchSchedulingPolicy.from_value(batch_schedule)
        self.show_sampler_progress = show_sampler_progress
        self.show_training_progress = show_training_progress

        if n_epochs < 1:
            raise ValueError("n_epochs must be >= 1")
        self.n_epochs = n_epochs

        self._validate_samplers()
        self._validate_losses()

        # Integer IDs assigned to each binding
        # Each ID corresponds to a unique sampler (used for alternating schedulers)
        self._sampler_ids: NDArray[np.int_] = np.arange(
            len(self.input_sources),
            dtype=int,
        )

    # ================================================
    # Validation
    # ================================================
    def _validate_samplers(self):
        exp_ctx = ExperimentContext.get_active()

        # Ensure all binding of head nodes define a sampler and stream
        for binding in self.input_sources:
            if binding.sampler is None:
                node = exp_ctx.get_node(
                    node_id=binding.node_id,
                    enforce_type="GraphNode",
                )
                msg = (
                    "TrainPhase requires that samplers are defined for all input "
                    f"sources. Missing sampler for node: '{node.label}'. Use "
                    "`<node>.create_input_binding(...)` to create the input source."
                )
                raise ValueError(msg)

    def _validate_losses(self):
        from modularml.core.training.applied_loss import AppliedLoss

        # Validate loss type
        for ls in self.losses:
            if not isinstance(ls, AppliedLoss):
                msg = f"Loss entries must be of type AppliedLoss. Received: {type(ls)}."
                raise TypeError(msg)

    # ================================================
    # Representation
    # ================================================
    def __repr__(self):
        return f"TrainPhase(label='{self.label}')"

    # ================================================
    # Execution
    # ================================================
    def _masked_batch_like(self, bv: BatchView) -> BatchView:
        """Creates a new, fully masked BatchView with the same size as `bv`."""
        return BatchView(
            source=bv.source,
            role_indices=np.full(bv.n_samples, -1, dtype=int),
            role_indice_weights=None,
        )

    def _build_sampler_executions(self) -> list[SamplerExecution]:
        """
        Groups `self.input_sources` into SamplerExecution objects.

        Description:
            Multiple bindings may share the same effective sampling “execution”:
            same upstream FeatureSet, same split restriction, and same (hashable)
            sampler configuration. In that case, we should only materialize batches
            once, and reuse the resulting SampledView for all bindings in the group.

            Grouping is conservative:
                - If sampler config is missing or unhashable -> do not dedupe.
                - If sampler has no `get_config()` -> do not dedupe.

            A SamplerExecution stores:
                - sampler_id: a unique integer [0..N_unique-1]
                - sampled: the SampledView produced by executing that sampler once
                - bindings: all InputBindings that reuse this execution

        Returns:
            list[SamplerExecution]:
                All unique SamplerExecution groups, sorted by `sampler_id`.

        """
        # key -> sampler_id
        key_to_id: dict[SamplerExecutionKey, int] = {}

        # sampler_id -> list of bindings
        id_to_bindings: dict[int, list[InputBinding]] = defaultdict(list)

        # sampler_id -> sampled view
        id_to_sampled: dict[int, SampledView] = {}

        # Need a unique key for when we cannot safely dedupe
        fallback_counter = 0
        for binding in self.input_sources:
            # Build a dedupe key for this sampler config
            sampler_cfg = None
            if hasattr(binding.sampler, "get_config"):
                try:
                    sampler_cfg = binding.sampler.get_config()
                except Exception:  # noqa: BLE001
                    sampler_cfg = None
            if sampler_cfg is not None:
                try:
                    hash(sampler_cfg)
                except TypeError:
                    sampler_cfg = None

            if sampler_cfg is None:
                # No reliable config => do not dedupe this sampler execution
                fallback_counter += 1
                exec_key = SamplerExecutionKey(
                    featureset_id=binding.upstream_ref.node_id,
                    split=binding.split,
                    sampler_cfg=("__no_dedupe__", fallback_counter),
                )
            else:
                exec_key = SamplerExecutionKey(
                    featureset_id=binding.upstream_ref.node_id,
                    split=binding.split,
                    sampler_cfg=sampler_cfg,
                )

            # Assign sampler_id for this exec group
            if exec_key not in key_to_id:
                # New exec key -> create ID and materialize SampledView
                sampler_id = len(key_to_id)
                key_to_id[exec_key] = sampler_id

                # InputBinding.resolve_input_view restrict columns to that specific binding
                # Since samplers are column-agnostics, we need to clear the columns
                fsv = binding.resolve_input_view()
                sampler_src = FeatureSetView(
                    source=fsv.source,
                    indices=fsv.indices,
                    columns=fsv.source.get_all_keys(
                        include_domain_prefix=True,
                        include_rep_suffix=True,
                    ),
                    label=f"{fsv.label}_for_sampler",
                )
                # Materialize batches once
                binding.sampler.bind_sources(sources=[sampler_src])
                binding.sampler.materialize_batches(
                    show_progress=self.show_sampler_progress,
                )

                # Capture the sampled output
                id_to_sampled[sampler_id] = binding.sampler.sampled

            else:
                # Existing exec -> get sampler_id
                sampler_id = key_to_id[exec_key]

            # Add this binding to the exec group
            id_to_bindings[sampler_id].append(binding)

        # Build ordered executions (sort by sampler ID)
        execs: list[SamplerExecution] = []
        for sampler_id in sorted(id_to_sampled.keys()):
            execs.append(
                SamplerExecution(
                    sampler_id=sampler_id,
                    sampled=id_to_sampled[sampler_id],
                    bindings=id_to_bindings[sampler_id],
                ),
            )

        return execs

    def _iter_schedule(
        self,
        *,
        policy: BatchSchedulingPolicy,
        sampler_lengths: list[int],
    ) -> Iterator[dict[int, int]]:
        """
        Yield a per-step schedule mapping sampler_id -> batch_index.

        ZIP_*:
            Each yielded dict contains all sampler_ids (one batch per sampler per step)

        ALTERNATE_*:
            Each yielded dict contains exactly one sampler_id (the active sampler for
            that step). Non-active samplers must be masked by the caller.
        """
        n_samplers = len(sampler_lengths)

        if policy == BatchSchedulingPolicy.ZIP_STRICT:
            # Zipped batching, but stop when shortest would be exceeded
            n_steps = min(sampler_lengths)
            for i in range(n_steps):
                # Each sampler uses same batch idx
                yield dict.fromkeys(range(n_samplers), i)

        elif policy == BatchSchedulingPolicy.ZIP_CYCLE:
            # Zipped batching, but stop when largest would be exceeded
            n_steps = max(sampler_lengths)
            for i in range(n_steps):
                # Must loop batch idx if beyond length of this sampler
                yield {sid: i % sampler_lengths[sid] for sid in range(n_samplers)}

        elif policy == BatchSchedulingPolicy.ALTERNATE_STRICT:
            # Round-robin batching, but stop when the shortest would be exceeded
            n_rounds = min(sampler_lengths)
            for i in range(n_rounds):
                for sid in range(n_samplers):
                    yield {sid: i}

        elif policy == BatchSchedulingPolicy.ALTERNATE_CYCLE:
            # Round-robin batching, but stop when the largest would be exceeded
            n_rounds = max(sampler_lengths)
            for i in range(n_rounds):
                for sid in range(n_samplers):
                    # Must loop batch idx if beyond length of this sampler
                    yield {sid: i % sampler_lengths[sid]}

        else:
            msg = f"Unknown BatchSchedulingPolicy: {policy}"
            raise TypeError(msg)

    def iter_execution(self) -> Iterator[ExecutionContext]:
        """
        Iterate over all execution steps for this training phase.

        Description:
            This generator produces a sequence of :class:`ExecutionContext` objects
            representing the full training schedule of the phase, across all epochs
            and all batch steps within each epoch.

            The execution flow is:

            1. Group input bindings into unique sampler executions via
            `_build_sampler_executions()`. Samplers with identical configuration,
            FeatureSet, and split are executed only once and shared across bindings.

            2. For each sampler execution, obtain the number of materialized batches
            from its :class:`SampledView`.

            3. For each epoch:
                a. Generate a step-wise batch schedule using `_iter_schedule()`,
                according to `self.batch_schedule`.
                b. For each schedule step, construct the inputs for *all* head-node
                bindings:
                    - If a sampler is active in the current step, its corresponding
                        batch is selected.
                    - If a sampler is inactive (ALTERNATE policies), its inputs are
                        replaced with a fully masked :class:`BatchView`.

            4. Yield an :class:`ExecutionContext` containing:
                - phase label
                - epoch index
                - batch index (within the epoch)
                - resolved input BatchViews for all bindings
                - losses to be applied for this phase

        Notes:
            - ZIP scheduling policies always provide one batch per sampler per step.
            - ALTERNATE scheduling policies activate exactly one sampler per step;
            all others are masked.
            - No semantic alignment between samplers is performed here. Any required
            alignment (e.g., contrastive pairs, matched samples) must be handled
            inside the sampler itself via roles.
            - The yielded ExecutionContexts are intended to be consumed directly by
            the ModelGraph training loop (e.g., `ModelGraph.train_step(ctx)`).

        Yields:
            ExecutionContext:
                A fully specified execution context for a single batch step in a
                specific epoch of this training phase.

        """
        # Samplers may be repeated over input bindings
        # We group by unique samplers (same sampler cfg, same FeatureSet + split)
        sampler_execs = self._build_sampler_executions()
        sampler_lens = [se.sampled.num_batches for se in sampler_execs]
        if any(x == 0 for x in sampler_lens):
            msg = (
                "One or more samplers produced zero batches; cannot execute TrainPhase."
            )
            raise RuntimeError(msg)

        step_cnt = sum(
            1
            for _ in self._iter_schedule(
                policy=self.batch_schedule,
                sampler_lengths=sampler_lens,
            )
        )
        epoch_ptask = ProgressTask(
            style="training",
            description=f"Training ['{self.label}']",
            total=self.n_epochs,
            enabled=self.show_training_progress,
            persist=IN_NOTEBOOK,
        )
        epoch_ptask.start()

        # Iterate over all epochs
        for epoch_idx in range(self.n_epochs):
            # Create per-epoch progress bar
            per_epoch_ptask = ProgressTask(
                style="training_loss",
                description=f"Epoch {epoch_idx + 1}",
                total=step_cnt,
                enabled=self.show_training_progress,
                persist=(epoch_idx == self.n_epochs - 1) and IN_NOTEBOOK,
            )
            # per_epoch_ptask.fields["loss"] = 0

            # Determine scheduling from self.batch_schedule
            step_iter = self._iter_schedule(
                policy=self.batch_schedule,
                sampler_lengths=sampler_lens,
            )
            # For each schedule step, get BatchView for each input binding
            for step_idx, step_plan in enumerate(step_iter):
                # step_plan: {sampler_id: batch_idx_for_that_sampler}
                inputs: dict[tuple[str, FeatureSetReference], BatchView] = {}

                # For each sampler, decide whether it's active this step and select/mask
                for sid, se in enumerate(sampler_execs):
                    for binding in se.bindings:
                        # Get list of batches for a given binding
                        all_bvs = se.sampled.get_stream(name=binding.stream)
                        key = (binding.node_id, binding.upstream_ref)

                        # Use real batch view, if this sampler is active in this step
                        if sid in step_plan:
                            bv = all_bvs[step_plan[sid]]
                            inputs[key] = bv
                        # Otherwise, use a fully masked batch
                        else:
                            bv = all_bvs[0]
                            inputs[key] = self._masked_batch_like(bv=bv)

                ctx = ExecutionContext(
                    phase_label=self.label,
                    epoch_idx=epoch_idx,
                    batch_idx=step_idx,
                    inputs=inputs,
                )
                yield ctx

                total_loss = sum(v.total for v in ctx.losses.values())
                total_train = sum(v.trainable for v in ctx.losses.values())
                total_aux = sum(v.auxiliary for v in ctx.losses.values())
                per_epoch_ptask.tick(
                    n=1,
                    loss_total=total_loss,
                    loss_train=total_train,
                    loss_aux=total_aux,
                )

            per_epoch_ptask.finish()
            epoch_ptask.tick(n=1)
        epoch_ptask.finish()
