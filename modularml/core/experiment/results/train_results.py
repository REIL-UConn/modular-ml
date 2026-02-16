from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from modularml.callbacks.evaluation import EvaluationCallbackResult
from modularml.core.experiment.results.phase_results import PhaseResults
from modularml.utils.data.multi_keyed_data import AxisSeries

if TYPE_CHECKING:
    from modularml.core.experiment.results.eval_results import EvalResults
    from modularml.core.references.execution_reference import TensorLike
    from modularml.core.topology.graph_node import GraphNode
    from modularml.core.training.loss_record import LossCollection
    from modularml.utils.data.data_format import DataFormat


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
        validation_callback_labels (list[str] | None):
            Labels of attached validation callbacks. Set by TrainPhase
            during execution.

    Examples:
        ```python
        # Run training
        train_results = experiment.run_training(phase=train_phase)

        # Get training loss per epoch
        epoch_losses = train_results.epoch_losses(node="output_node")

        # Get validation losses (from Evaluation callbacks)
        val_losses = train_results.validation_losses(node="output_node")
        ```

    """

    validation_callback_labels: list[str] | None = None

    # ================================================
    # Representation
    # ================================================
    def __repr__(self):
        n_epochs = self.n_epochs if self._execution else 0
        val_labels = self.validation_callback_labels or []
        return (
            f"TrainResults(label='{self.label}', "
            f"epochs={n_epochs}, "
            f"validation_callbacks={val_labels})"
        )

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
        """The number of epochs executed during training."""
        return len(self.epoch_indices)

    # ================================================
    # Loss Aggregation
    # ================================================
    def epoch_losses(
        self,
        node: str | GraphNode,
        *,
        reducer: Literal["sum", "mean"] = "mean",
    ) -> AxisSeries[LossCollection]:
        """
        Retrieve training losses aggregated per epoch.

        Description:
            Collapses batch-level losses within each epoch using the specified
            reducer. Returns an AxisSeries keyed by epoch for easy iteration
            and plotting.

        Args:
            node (str | GraphNode):
                The node to retrieve losses for.
            reducer (Literal["sum", "mean"], optional):
                How to aggregate losses within each epoch.
                Defaults to "mean".

        Returns:
            AxisSeries[LossCollection]:
                Losses keyed by epoch. Use .values() to get a list or
                iterate with .items() for epoch -> LossCollection pairs.

        Examples:
            ```python
            epoch_losses = train_results.epoch_losses(node="output_node")

            # Plot training curve
            for epoch, loss in epoch_losses.items():
                print(f"Epoch {epoch}: {loss.total:.4f}")

            # Get loss for specific epoch
            loss_e5 = epoch_losses.at(epoch=5)
            ```

        """
        loss_series = self.losses(node=node)

        # Collapse batch axis (per epoch, per label)
        batch_collapsed = loss_series.collapse(axis="batch", reducer=reducer)

        # Collapse label axis (per epoch)
        return batch_collapsed.collapse(axis="label", reducer=reducer)

    def batch_losses(
        self,
        node: str | GraphNode,
        *,
        epoch: int | list[int] | None = None,
    ) -> AxisSeries[LossCollection]:
        """
        Retrieve raw training losses keyed by (epoch, batch).

        Description:
            Returns unaggregated losses for fine-grained analysis. Optionally
            filter to specific epoch(s). Labels are collapsed so each
            (epoch, batch) maps to a single LossCollection.

        Args:
            node (str | GraphNode):
                The node to retrieve losses for.
            epoch (int | list[int] | None, optional):
                Filter to specific epoch(s). If None, returns all epochs.

        Returns:
            AxisSeries[LossCollection]:
                Losses keyed by (epoch, batch).

        Examples:
            ```python
            # Get all batch losses
            all_losses = train_results.batch_losses(node="output")

            # Get batch losses for epoch 5 only
            e5_losses = train_results.batch_losses(node="output", epoch=5)
            ```

        """
        loss_series = self.losses(node=node)

        # Collapse label axis to get (epoch, batch) -> LossCollection
        label_collapsed = loss_series.collapse(axis="label", reducer="mean")

        if epoch is not None:
            return label_collapsed.where(epoch=epoch)

        return label_collapsed

    # ================================================
    # Validation Callback Methods
    # ================================================
    def validation_callbacks(
        self,
        *,
        label: str | None = None,
    ) -> AxisSeries[EvaluationCallbackResult]:
        """
        Retrieve validation callback results.

        Description:
            Filters callbacks by kind="evaluation" and returns the underlying
            EvaluationCallbackResult objects. These contain full EvalResults
            for each validation run.

        Args:
            label (str | None, optional):
                Filter to a specific callback label. If None and multiple
                validation callbacks exist, all are returned. If None and
                exactly one validation callback exists, it is used.

        Returns:
            AxisSeries[EvaluationCallbackResult]:
                Validation results keyed by epoch. Each value is the
                EvaluationCallbackResult containing the full EvalResults.

        Raises:
            ValueError:
                If no validation callbacks were attached.
            KeyError:
                If the specified label does not exist.

        Examples:
            ```python
            # Get all validation results
            val_callbacks = train_results.validation_callbacks()

            # Access underlying EvalResults
            for epoch, cb_result in val_callbacks.items():
                eval_results = cb_result.eval_results
                ...
            ```

        """
        if self.validation_callback_labels is None:
            msg = "No validation callbacks were attached to this phase."
            raise ValueError(msg)

        # Resolve label(s) to filter
        if label is None:
            labels = self.validation_callback_labels
        else:
            if label not in self.validation_callback_labels:
                msg = (
                    f"Validation callback '{label}' not found. "
                    f"Available: {self.validation_callback_labels}"
                )
                raise KeyError(msg)
            labels = [label]

        # Filter callbacks by kind="evaluation" and matching labels
        cb_series = self.callbacks()

        # Build new AxisSeries keyed by (epoch,)
        result_data: dict[tuple[int,], EvaluationCallbackResult] = {}

        for lbl in labels:
            try:
                cb_list = cb_series.at(label=lbl, kind="evaluation")
            except KeyError:
                continue

            for cb_result in cb_list:
                if not isinstance(cb_result, EvaluationCallbackResult):
                    continue
                # Key by epoch; use -1 for run_on_start (epoch=None)
                epoch_key = (
                    cb_result.epoch_idx if cb_result.epoch_idx is not None else -1
                )
                result_data[(epoch_key,)] = cb_result

        return AxisSeries(axes=("epoch",), _data=result_data)

    def validation_losses(
        self,
        node: str | GraphNode,
        *,
        label: str | None = None,
        reducer: Literal["sum", "mean"] = "mean",
    ) -> AxisSeries[LossCollection]:
        """
        Retrieve validation losses keyed by training epoch.

        Description:
            Extracts losses from validation callbacks (Evaluation) and returns
            them keyed by the training epoch at which validation was run.

            Each validation run produces an EvalResults with multiple batches.
            These are aggregated using the specified reducer.

        Args:
            node (str | GraphNode):
                The node to retrieve validation losses for.
            label (str | None, optional):
                Validation callback label. Required if multiple validation
                callbacks exist. Defaults to None (auto-detect single).
            reducer (Literal["sum", "mean"], optional):
                How to aggregate losses across validation batches.
                - "sum": Concatenates all individual loss records
                - "mean": Computes the mean loss value per label
                Defaults to "mean".

        Returns:
            AxisSeries[LossCollection]:
                Validation losses keyed by epoch. Use epoch=-1 for
                validation run at phase start (for `run_on_start=True`).

        Examples:
            ```python
            val_losses = train_results.validation_losses(node="output_node")

            # Plot validation curve
            for epoch, loss in val_losses.items():
                print(f"Epoch {epoch}: val_loss={loss.total:.4f}")

            # Compare training and validation
            train_losses = train_results.epoch_losses(node="output_node")
            for epoch in train_results.epoch_indices:
                train_l = train_losses.at(epoch=epoch)
                val_l = val_losses.at(epoch=epoch)
                print(f"E{epoch}: train={train_l.total:.4f}, val={val_l.total:.4f}")
            ```

        """
        val_cbs = self.validation_callbacks(label=label)

        result_data: dict[tuple[int,], LossCollection] = {}
        for epoch, cb_result in val_cbs.items():
            eval_results: EvalResults | None = cb_result.eval_results
            if eval_results is None:
                continue

            # Aggregate losses from EvalResults
            agg_loss = eval_results.aggregated_losses(
                node=node,
                reducer=reducer,
                by_label=False,
            )
            result_data[(epoch,)] = agg_loss

        return AxisSeries(axes=("epoch",), _data=result_data)

    def validation_tensors(
        self,
        node: str | GraphNode,
        domain: Literal["outputs", "targets", "tags", "sample_uuids"],
        *,
        label: str | None = None,
        role: str = "default",
        fmt: DataFormat | None = None,
        unscale: bool = False,
    ) -> AxisSeries[TensorLike]:
        """
        Retrieve validation tensors (outputs/targets) keyed by training epoch.

        Description:
            Extracts tensors from validation callbacks and returns them keyed
            by the training epoch at which validation was run. Tensors from
            each validation run are automatically concatenated across batches.

        Args:
            node (str | GraphNode):
                The node to retrieve validation tensors for.
            domain (Literal["outputs", "targets", "tags", "sample_uuids"]):
                Which tensor domain to retrieve.
            label (str | None, optional):
                Validation callback label. Defaults to None.
            role (str, optional):
                Data role for multi-role samplers. Defaults to "default".
            fmt (DataFormat | None, optional):
                Format to cast tensors to. Defaults to None.
            unscale (bool, optional):
                Whether to inverse scalers. Only supported if `node` refers to
                a tail node in the model graph. Defaults to False.

        Returns:
            AxisSeries[TensorLike]:
                Validation tensors keyed by epoch. Each tensor contains
                all samples from the validation dataset.

        Examples:
            ```python
            # Get validation outputs per epoch
            val_outputs = train_results.validation_tensors(
                node="output_node",
                domain="outputs",
            )

            # Compute custom metric per epoch
            val_targets = train_results.validation_tensors(
                node="output_node",
                domain="targets",
            )
            for epoch, outputs in val_outputs.items():
                targets = val_targets.at(epoch=epoch)
                mse = ((outputs - targets) ** 2).mean()
                print(f"Epoch {epoch}: MSE = {mse:.4f}")
            ```

        """
        val_cbs = self.validation_callbacks(label=label)

        result_data: dict[tuple[int,], TensorLike] = {}
        for epoch, cb_result in val_cbs.items():
            eval_results: EvalResults | None = cb_result.eval_results
            if eval_results is None:
                continue

            # Get stacked tensor from EvalResults
            tensor = eval_results.stacked_tensors(
                node=node,
                domain=domain,
                role=role,
                fmt=fmt,
                unscale=unscale,
            )
            result_data[(epoch,)] = tensor

        return AxisSeries(axes=("epoch",), _data=result_data)
