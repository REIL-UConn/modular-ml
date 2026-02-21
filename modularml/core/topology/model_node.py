"""Model node implementations within ModularML model graphs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, overload

from modularml.core.data.batch import Batch
from modularml.core.data.featureset import FeatureSet
from modularml.core.data.sample_data import RoleData, SampleData
from modularml.core.experiment.experiment_node import ExperimentNode
from modularml.core.models import wrap_model
from modularml.core.models.base_model import BaseModel
from modularml.core.references.experiment_reference import ExperimentNodeReference
from modularml.core.references.featureset_reference import FeatureSetReference
from modularml.core.topology.compute_node import ComputeNode, TForward
from modularml.core.training.loss_record import LossCollection, LossRecord
from modularml.core.training.optimizer import Optimizer
from modularml.utils.data.data_format import DataFormat, get_data_format_for_backend
from modularml.utils.environment.optional_imports import check_tensorflow, check_torch
from modularml.utils.errors.exceptions import (
    BackendMismatchError,
    BackendNotSupportedError,
    OptimizerNotSetError,
)
from modularml.utils.logging.warnings import catch_warnings, warn
from modularml.utils.nn.backend import Backend
from modularml.utils.representation.summary import safe_cast_to_summary_rows
from modularml.utils.topology.graph_search_utils import find_upstream_featuresets

if TYPE_CHECKING:
    from modularml.core.data.execution_context import ExecutionContext
    from modularml.core.training.applied_loss import AppliedLoss

tf = check_tensorflow()
torch = check_torch()


class ModelNode(ComputeNode):
    """
    Single learnable or static stage inside a :class:`ModelGraph`.

    Attributes:
        _model (BaseModel): Wrapped backend model implementation.
        _optimizer (Optimizer | None): Optional optimizer coordinating
            gradient steps.
        _freeze (bool): Flag indicating whether training is disabled.

    """

    def __init__(
        self,
        label: str,
        model: BaseModel | Any,
        upstream_ref: ExperimentNode | ExperimentNodeReference,
        optimizer: Optimizer | None = None,
        *,
        node_id: str | None = None,
        register: bool = True,
    ):
        """
        Initialize a ModelNode.

        Args:
            label (str):
                Unique name identifying this stage within the model graph.
            model (Union[BaseModel, Any]):
                A backend-specific model instance or config.
            upstream_ref (ExperimentReference):
                Reference to the upstream node.
            optimizer (Optional[Optimizer]):
                Optimizer to use during training (optional).
            node_id (str, optional):
                Used only for de-serialization.
            register (bool, optional):
                Used only for de-serialization.

        """
        ref = None
        if isinstance(upstream_ref, FeatureSet):
            dup_rep_warnings = False
            with catch_warnings() as cw:
                ref = upstream_ref.reference()
                if cw.match("Multiple representations selected"):
                    dup_rep_warnings = True
            if dup_rep_warnings:
                msg = (
                    "Setting a ModelNode `upstream_ref` with a FeatureSet will result in multiple "
                    "representations of the same column being combined into input/target tensors. "
                )
                hint = (
                    "Use `FeatureSet(...).reference(...)` is this is not intentional."
                )
                warn(msg, category=UserWarning, stacklevel=2, hints=hint)
        elif isinstance(upstream_ref, ExperimentNodeReference):
            ref = upstream_ref
        elif isinstance(upstream_ref, ExperimentNode):
            ref = upstream_ref.reference()
        else:
            msg = f"`upstream_ref` must be of type ExperimentReference or ExperimentNode. Received: {type(upstream_ref)}."
            raise TypeError(msg)

        super().__init__(
            label=label,
            upstream_refs=ref,
            node_id=node_id,
            register=register,
        )

        # Set model (cast to BaseModel if explicit subclass not provided)
        self._model: BaseModel = wrap_model(model)
        self._freeze = False  # make stage trainable as default

        # Error checking on optimizer (can be None)
        self._optimizer = optimizer
        self._check_valid_optimizer(required=False)

    @property
    def model(self) -> BaseModel:
        """
        Return the wrapped backend model instance.

        Returns:
            BaseModel: Backend-specific implementation.

        """
        return self._model

    @property
    def input_shape(self) -> tuple[int, ...]:
        """
        Return the model's input tensor shape.

        Returns:
            tuple[int, ...]: Expected feature tensor shape.

        """
        return self.model.input_shape

    # ================================================
    # ComputeNode Interface
    # ================================================
    @property
    def output_shape(self) -> tuple[int, ...]:
        """
        Return the model's output tensor shape.

        Returns:
            tuple[int, ...]: Output tensor shape.

        """
        return self.model.output_shape

    @property
    def max_upstream_refs(self) -> int:
        """
        Return the maximum number of allowed upstream references.

        Returns:
            int: Always 1 because :class:`ModelNode` has a single input.

        """
        return 1

    @property
    def is_built(self) -> bool:
        """
        Checks if the model has been built (i.e., instantiated with input/output shape).

        Returns:
            bool: True if built, False otherwise.

        """
        return self._model.is_built

    def _build_impl(
        self,
        *,
        input_shapes: dict[ExperimentNodeReference, tuple[int, ...]] | None = None,
        output_shape: tuple[int, ...] | None = None,
        force: bool = False,
        **kwargs,  # noqa: ARG002
    ):
        """
        Construct the wrapped model using upstream/downstream shapes.

        Args:
            input_shapes (dict[ExperimentNodeReference, tuple[int, ...]] | None):
                Shapes of upstream tensors; must contain a single entry.
            output_shape (tuple[int, ...] | None):
                Expected output shape used to validate decoder layers.
            force (bool):
                Whether to rebuild even if already built.
            **kwargs:
                Additional subclass parameters (unused).

        Raises:
            ValueError: If multiple inputs are provided.

        """
        if input_shapes is None:
            input_shape = None
        else:
            if len(input_shapes) != 1:
                msg = (
                    f"{self.__class__.__name__} expects exactly one input. "
                    f"Received {len(input_shapes)}."
                )
                raise ValueError(msg)
            input_shape = next(iter(input_shapes.values()))

        self.build_model(
            input_shape=input_shape,
            output_shape=output_shape,
            force=force,
        )

    def _build_optimizer(self, *, force: bool = False):
        """
        Construct the optimizer once the model weights exist.

        Args:
            force (bool): Whether to rebuild even if already built.

        Raises:
            ValueError: If optimizer or model state is unavailable.
            BackendNotSupportedError: If the backend is unknown.

        """
        if self._optimizer is None:
            raise ValueError("Optimizer is None. Cannot build.")
        if not self.is_built:
            raise ValueError("Optimzier cannot be built until model is built.")

        if self.backend == Backend.TORCH:
            self._optimizer.build(
                parameters=self._model.parameters(),
                backend=self.backend,
                force_rebuild=force,
            )
        elif self.backend == Backend.TENSORFLOW:
            self._optimizer.build(
                backend=self.backend,
                force_rebuild=force,
            )
        elif self.backend == Backend.SCIKIT:
            # Scikit-learn optimizers are typically fit internally
            pass
        else:
            raise BackendNotSupportedError(
                backend=self.backend,
                message="Unknown backend for optimizer building",
            )

    def build_model(
        self,
        input_shape: tuple[int, ...] | None = None,
        output_shape: tuple[int, ...] | None = None,
        *,
        force: bool = False,
    ):
        """
        Build the ModelNode by initializing the internal BaseModel and optimizer.

        Args:
            input_shape (tuple[int, ...] | None, optional):
                Input shape to construct this model with.
                Defaults to None.

            output_shape (tuple[int, ...] | None, optional):
                Output shape to construct this model with. If not provided, the
                BaseModel must be capable of inferring it internally or during
                construction. Defaults to None.

            force (bool, optional):
                If model is already instantiated it will not be re-instantiated unless
                `force=True`. Defaults to False.

        Notes:
            - For PyTorch and TensorFlow, optimizers are built after the model is initialized.
            - Scikit-learn models typically do not require external optimizers.
            - This method assumes that shape inference and merge logic (if needed) has already
              been resolved upstream by the ModelGraph.

        """
        # Build underlying BaseModel if not already built
        if (not self._model.is_built) or force:
            self._model.build(
                input_shape=input_shape,
                output_shape=output_shape,
                force=force,
            )

        # Build optimizer if defined
        if self._optimizer is not None:
            self._build_optimizer(force=force)

    @overload
    def forward_single(self, batch: Batch, **kwargs) -> Batch: ...

    @overload
    def forward_single(self, roles: RoleData, **kwargs) -> RoleData: ...

    @overload
    def forward_single(self, data: SampleData, **kwargs) -> SampleData: ...

    def forward_single(
        self,
        x: SampleData | RoleData | Batch,
        **kwargs,
    ) -> SampleData | RoleData | Batch:
        """
        Performs a forward pass through the model using SampleData.

        This method preserves raw tensor outputs to maintain backend autograd support.
        It returns a `SampleData` object keyed by output roles containing model predictions.

        Args:
            x (SampleData | RoleData | Batch): Input data to the model.
            **kwargs: Any additional keyword arguments to provide to BaseModel.forward

        Returns:
            SampleData | RoleData | Batch:
                Outputs from the model. Output type matches input.

        """
        # Ensure built
        if not self.is_built:
            # We can try to auto-build base on runtime upstream/downstream connections
            # If upstream_ref is a FeatureSet, we can take feature shapes
            in_shape = None
            if isinstance(self.upstream_ref, FeatureSetReference):
                # Get feature and target shapes (drops leading dim of n_samples)
                fsv = self.upstream_ref.resolve()
                in_shape = fsv.get_features(fmt=DataFormat.NUMPY).shape[1:]

            # If this is a tail node, and is downstream of only one FeatureSet, we
            # can infer the output shape to be the FeatureSet.targets shape
            out_shape = None
            ups_fs_refs = find_upstream_featuresets(node=self)
            ups_fs_ids = {ref.node_id for ref in ups_fs_refs}
            if len(ups_fs_ids) == 1:
                fsv = ups_fs_refs[0].resolve()
                t_shape = fsv.get_targets(fmt=DataFormat.NUMPY).shape[1:]
                out_shape = tuple(t_shape)

            try:
                self.build_model(
                    input_shape=in_shape,
                    output_shape=out_shape,
                )
            except Exception as e:
                msg = (
                    f"ModelNode '{self.label}' has not been built yet. "
                    "Call `build_model()` first."
                )
                raise RuntimeError(msg) from e

        def _forward_sample_data(d: SampleData) -> SampleData:
            """
            Run backend-forward pass for a single :class:`SampleData`.

            Args:
                d (SampleData): Input sample bundle.

            Returns:
                SampleData: Output bundle preserving metadata.

            """
            # Ensure SampleData is in expected backend (modified inplace)
            d.as_backend(self.backend)

            # Pass features through internal model
            out_features = self._model(d.features, **kwargs)

            # Targets, tags, and uuids pass through without modification
            return SampleData(
                features=out_features,
                targets=d.targets,
                tags=d.tags,
                sample_uuids=d.sample_uuids,
                kind="output",
            )

        if isinstance(x, SampleData):
            return _forward_sample_data(x)

        if isinstance(x, RoleData):
            out = {k: _forward_sample_data(v) for k, v in x.items()}
            return RoleData(data=out)

        if isinstance(x, Batch):
            out = RoleData(
                data={k: _forward_sample_data(v) for k, v in x.role_data.items()},
            )

            return Batch(
                batch_size=x.batch_size,
                role_data=out,
                shapes=out.shapes,
                role_weights=x.role_weights,
                role_masks=x.role_masks,
            )

        msg = f"Input must be of type SampleData or RoleData or Batch. Received: {type(x)}"
        raise TypeError(msg)

    def _forward_impl(
        self,
        *,
        inputs: dict[ExperimentNodeReference, TForward],
        **kwargs,
    ) -> TForward:
        """
        Delegate to :meth:`forward_single` after validating inputs.

        Args:
            inputs (dict[ExperimentNodeReference, TForward]): =
                Single upstream tensor keyed by its reference.
            **kwargs:
                Extra arguments forwarded to :meth:`forward_single`.

        Returns:
            TForward: Batch or sample data emitted by the model.

        Raises:
            ValueError: If more than one input is supplied.

        """
        if len(inputs) != 1:
            msg = (
                f"{self.__class__.__name__} expects exactly one input. "
                f"Received {len(inputs)}."
            )
            raise ValueError(msg)

        x = next(iter(inputs.values()))
        return self.forward_single(x, **kwargs)

    __call__ = forward_single

    # ================================================
    # Representation
    # ================================================
    def _summary_rows(self) -> list[tuple]:
        """
        Return tabular summary rows for logging output.

        Returns:
            list[tuple]: Key/value metadata about the node.

        """
        return [
            ("label", self.label),
            ("upstream_ref", safe_cast_to_summary_rows(self.upstream_ref)),
            (
                "downstream_refs",
                [safe_cast_to_summary_rows(r) for r in self._downstream_refs],
            ),
            (
                "input_shape",
                str(self.input_shape) if self.is_built else "NOT BUILT YET",
            ),
            (
                "output_shape",
                str(self.output_shape) if self.is_built else "NOT BUILT YET",
            ),
            ("model", safe_cast_to_summary_rows(self._model)),
            ("optimizer", safe_cast_to_summary_rows(self._optimizer)),
            ("backend", safe_cast_to_summary_rows(self.backend)),
            ("frozen", f"{'True' if self.is_frozen else 'False'}"),
        ]

    def __repr__(self):
        """
        Return developer-friendly representation for debugging.

        Returns:
            str: String showing labels, model, optimizer, and backend.

        """
        return (
            f"ModelNode(label='{self.label}', "
            f"upstream_refs={self._upstream_refs}, "
            f"downstream_refs={self._downstream_refs}, "
            f"model={self._model!r}, "
            f"optimizer={self._optimizer}, "
            f"backend={self.backend})"
        )

    def __str__(self):
        """
        Return human-readable identifier for logging.

        Returns:
            str: Node label formatted for readability.

        """
        return f"ModelNode('{self.label}')"

    # ================================================
    # Error Checking Methods
    # ================================================
    def _check_valid_optimizer(self, *, required: bool = True):
        """
        Verifies that the optimizer is compatible with the model's backend.

        Args:
            required (bool): Whether an optimizer is required. Default is True.

        Raises:
            OptimizerNotSetError: If required and optimizer is None.
            BackendMismatchError: If optimizer and model backends differ.

        """
        if self._optimizer is None and required:
            msg = f"Missing optimizer for ModelNode '{self.label}'."
            raise OptimizerNotSetError(message=msg)

        if self._optimizer is not None:
            if self._optimizer.backend is None:
                self._optimizer.backend = self.backend
            elif self._optimizer.backend != self.backend:
                raise BackendMismatchError(
                    expected=self.backend,
                    received=self._optimizer.backend,
                    message=f"Optimizer backend does not match model backend: {self._optimizer.backend} != {self.backend}",
                )

    def _validate_ctx(self, ctx: ExecutionContext):
        """
        Validates that the context contains needed input data for this node.

        Args:
            ctx (ExecutionContext):
                Execution context to validate losses on.

        Raises:
            ValueError: If any expected input or loss role is missing.

        """
        # If this node takes input from FeatureSet, ensure in ctx.inputs
        if isinstance(self.upstream_ref, FeatureSetReference):
            req_input_key = (self.node_id, self.upstream_ref)
            if req_input_key not in ctx.inputs:
                msg = (
                    f"ExecutionContext missing input data for ModelNode '{self.label}'."
                )
                raise ValueError(msg)

        # Otherwise, prior model outputs must be in ctx.outputs
        elif self.upstream_ref.node_id not in ctx.outputs:
            msg = f"ExecutionContext missing output data from upstream node '{self.upstream_ref.node_label}'."
            raise ValueError(msg)

    # ================================================
    # Trainable Protocol
    # ================================================
    @property
    def backend(self) -> Backend:
        """
        Returns the backend associated with the wrapped model.

        Returns:
            Backend: TORCH, TENSORFLOW, SCIKIT, ...

        """
        return self._model.backend

    @property
    def is_frozen(self) -> bool:
        """
        Indicates whether this stage is frozen (not trainable).

        Returns:
            bool: True if frozen, False if trainable.

        """
        return self._freeze

    def freeze(self):
        """Freezes this node (prevents training updates)."""
        self._freeze = True

        # Ensure trainable state
        if self.backend == Backend.TORCH:
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.eval()
        elif self.backend == Backend.TENSORFLOW:
            self.model.trainable = False

    def unfreeze(self):
        """Unfreezes this node (allows training updates)."""
        self._freeze = False

        # Ensure trainable state
        if self.backend == Backend.TORCH:
            for p in self.model.parameters():
                p.requires_grad = True
            self.model.train()
        elif self.backend == Backend.TENSORFLOW:
            self.model.trainable = True

    def _get_input_batch(
        self,
        ctx: ExecutionContext,
    ) -> Batch:
        """Retrieves Batch data for this ModelNode at the current execution step."""
        all_inp_data = self.get_input_data(
            inputs=ctx.inputs,
            outputs=ctx.outputs,
            fmt=get_data_format_for_backend(backend=self.backend),
        )
        return all_inp_data[self.upstream_ref]

    def _train_step_torch(
        self,
        ctx: ExecutionContext,
        losses: list[AppliedLoss],
    ):
        """
        Runs a training step using PyTorch: forward, loss, backward, optimizer.

        Args:
            ctx (ExecutionContext):
                Context (input/output data) for the given execution step.
            losses (list[AppliedLoss]):
                List of losses to be applied in this execution step.

        """
        # Set optimizer and train mode
        self._model.train()
        self._optimizer.zero_grad()

        loss_records: list[LossRecord] = []

        # Forward pass (ctx.execution modified inplace)
        out_batch: Batch = self.forward_single(self._get_input_batch(ctx=ctx))
        ctx.set_output(node_id=self.node_id, batch=out_batch)

        # Compute losses
        for loss in losses:
            weighted_raw_loss = loss.compute(ctx=ctx)
            lr = LossRecord(
                label=loss.label,
                node_id=loss.node_id,
                trainable=weighted_raw_loss,
            )
            loss_records.append(lr)

        # Backward + opt step
        lc = LossCollection(records=loss_records)
        lc.trainable.backward()
        self._optimizer.step()

        # Record loss collection
        ctx.add_losses(lc)

    def _train_step_tensorflow(
        self,
        ctx: ExecutionContext,
        losses: list[AppliedLoss],
    ):
        """
        Runs a training step using Tensorflow: forward, loss, backward, optimizer.

        Args:
            ctx (ExecutionContext):
                Context (input/output data) for the given execution step.
            losses (list[AppliedLoss]):
                List of losses to be applied in this execution step.

        """
        # Zero optimizer
        self._optimizer.zero_grad()

        loss_records: list[LossRecord] = []

        # Track gradients over forward passes & loss computation
        with tf.GradientTape() as tape:
            # Forward pass (ctx.execution modified inplace)
            out_batch: Batch = self.forward_single(
                self._get_input_batch(ctx=ctx),
                training=True,
            )
            ctx.set_output(node_id=self.node_id, batch=out_batch)

        # Compute losses
        for loss in losses:
            weighted_raw_loss = loss.compute(ctx=ctx)
            lr = LossRecord(
                label=loss.label,
                node_id=loss.node_id,
                trainable=weighted_raw_loss,
            )
            loss_records.append(lr)

        # Backward + opt step
        lc = LossCollection(records=loss_records)
        grads = tape.gradient(lc.trainable, self._model.trainable_variables)
        self._optimizer.step(grads=grads, variables=self._model.trainable_variables)

        # Record loss collection
        ctx.add_losses(lc)

    def _train_step_scikit(
        self,
        ctx: ExecutionContext,
        losses: list[AppliedLoss],
    ):
        """
        Runs a training step using scikit-learn's `partial_fit`.

        Only applicable to models that support incremental learning (e.g.,
        SGDRegressor, MLPRegressor). Batch-fit models should use `FitPhase`
        instead.

        Args:
            ctx (ExecutionContext):
                Context (input/output data) for the given execution step.
            losses (list[AppliedLoss]):
                List of losses to be applied in this execution step.

        """
        from modularml.core.models.scikit_wrapper import (
            ScikitModelWrapper,
            ScikitTrainingMode,
        )

        if (
            isinstance(self._model, ScikitModelWrapper)
            and self._model.resolved_training_mode != ScikitTrainingMode.PARTIAL_FIT
        ):
            msg = (
                f"ModelNode '{self.label}' wraps a batch-fit scikit model "
                f"({type(self._model.model).__name__}) that does not support "
                "incremental training. Use `fit_step` instead of `train_step`."
            )
            raise RuntimeError(msg)

        # Get input batch
        input_batch: Batch = self._get_input_batch(ctx=ctx)

        # Merge data from all roles, then partial fit on joint set
        joint_sd = SampleData.concat(
            *list(input_batch.role_data.values()),
            fmt=get_data_format_for_backend(self.backend),
        )
        # Perform incremental fit on this merged data
        self._model.partial_fit(
            joint_sd.features,
            joint_sd.targets,
        )

        # Forward pass to record outputs (equivalent to .predict())
        out_batch: Batch = self.forward_single(input_batch)
        ctx.set_output(node_id=self.node_id, batch=out_batch)

        # Compute losses (recorded as auxiliary since no gradient backprop)
        loss_records: list[LossRecord] = []
        if losses:
            for loss in losses:
                weighted_raw_loss = loss.compute(ctx=ctx)
                lr = LossRecord(
                    label=loss.label,
                    node_id=loss.node_id,
                    auxiliary=weighted_raw_loss,
                )
                loss_records.append(lr)

        lc = LossCollection(records=loss_records)
        ctx.add_losses(lc)

    def train_step(
        self,
        ctx: ExecutionContext,
        losses: list[AppliedLoss],
    ):
        """
        Performs a training step (forward, loss, backward, optimizer step) for this stage.

        Only callable if this stage has an optimizer and is not frozen. Otherwise, training
        must be delegated to `ModelGraph`.

        Args:
            ctx (ExecutionContext):
                Context (input/output data) for the given execution step.
            losses (list[AppliedLoss]):
                List of losses to be applied in this execution step.

        Raises:
            RuntimeError: If stage is frozen or optimizer is missing.

        """
        # If stage is frozen, raise error
        if self.is_frozen:
            msg = "Cannot train a frozen node. Either unfreeze or use `eval_step`."
            raise RuntimeError(msg)

        # Ensure input data exists for this node
        self._validate_ctx(ctx=ctx)
        # Ensure losses only include those applied to this node
        valid_losses = losses
        if losses is not None:
            valid_losses = [loss for loss in losses if loss.node_id == self.node_id]

        # Ensure optimizer is set and matches model backend
        self._check_valid_optimizer(required=True)

        if self.backend == Backend.TORCH:
            return self._train_step_torch(ctx=ctx, losses=valid_losses)

        if self.backend == Backend.TENSORFLOW:
            return self._train_step_tensorflow(ctx=ctx, losses=valid_losses)

        if self.backend == Backend.SCIKIT:
            return self._train_step_scikit(ctx=ctx, losses=valid_losses)

        msg = f"Unknown backend: {self.backend}"
        raise ValueError(msg)

    # ================================================
    # Evaluable Protocol
    # ================================================
    def _eval_step_torch(
        self,
        ctx: ExecutionContext,
        losses: list[AppliedLoss] | None = None,
    ):
        """
        Runs an evaluation step using PyTorch: forward + loss (no gradients).

        Args:
            ctx (ExecutionContext):
                Context (input/output data) for the given execution step.
            losses (list[AppliedLoss]):
                Optional list of losses to be applied in this execution step.

        """
        # Set eval mode
        self._model.eval()

        loss_records: list[LossRecord] = []

        # Forward pass (ctx.execution modified inplace)
        with torch.no_grad():
            out_batch: Batch = self.forward_single(self._get_input_batch(ctx=ctx))
            ctx.set_output(node_id=self.node_id, batch=out_batch)

            # Compute losses
            if losses is not None:
                for loss in losses:
                    weighted_raw_loss = loss.compute(ctx=ctx)
                    lr = LossRecord(
                        label=loss.label,
                        node_id=loss.node_id,
                        auxiliary=weighted_raw_loss,
                    )
                    loss_records.append(lr)

        # Record loss collection
        lc = LossCollection(records=loss_records)
        ctx.add_losses(lc)

    def _eval_step_tensorflow(
        self,
        ctx: ExecutionContext,
        losses: list[AppliedLoss] | None = None,
    ):
        """
        Runs an evaluation step using Tensorflow: forward + loss (no gradients).

        Args:
            ctx (ExecutionContext):
                Context (input/output data) for the given execution step.
            losses (list[AppliedLoss]):
                Optional list of losses to be applied in this execution step.

        """
        loss_records: list[LossRecord] = []

        # Forward pass (ctx.execution modified inplace)
        out_batch: Batch = self.forward_single(
            self._get_input_batch(ctx=ctx),
            training=False,
        )
        ctx.set_output(node_id=self.node_id, batch=out_batch)

        # Compute losses
        if losses is not None:
            for loss in losses:
                weighted_raw_loss = loss.compute(ctx=ctx)
                lr = LossRecord(
                    label=loss.label,
                    node_id=loss.node_id,
                    auxiliary=weighted_raw_loss,
                )
                loss_records.append(lr)

        # Record loss collection
        lc = LossCollection(records=loss_records)
        ctx.add_losses(lc)

    def _eval_step_scikit(
        self,
        ctx: ExecutionContext,
        losses: list[AppliedLoss] | None = None,
    ):
        """
        Runs an evaluation step for a scikit-learn model: forward pass + optional loss.

        Args:
            ctx (ExecutionContext):
                Context (input/output data) for the given execution step.
            losses (list[AppliedLoss]):
                Optional list of losses to be applied in this execution step.

        """
        # Forward pass
        out_batch: Batch = self.forward_single(self._get_input_batch(ctx=ctx))
        ctx.set_output(node_id=self.node_id, batch=out_batch)

        # Compute losses (auxiliary only)
        loss_records: list[LossRecord] = []
        if losses is not None:
            for loss in losses:
                weighted_raw_loss = loss.compute(ctx=ctx)
                lr = LossRecord(
                    label=loss.label,
                    node_id=loss.node_id,
                    auxiliary=weighted_raw_loss,
                )
                loss_records.append(lr)

        lc = LossCollection(records=loss_records)
        ctx.add_losses(lc)

    def eval_step(
        self,
        ctx: ExecutionContext,
        losses: list[AppliedLoss] | None = None,
    ):
        """
        Performs an evaluation step (forward pass and loss computation) for this stage.

        Only callable if this stage is frozen. No gradient tracking is performed.

        Args:
            ctx (ExecutionContext):
                Context (input/output data) for the given execution step.
            losses (list[AppliedLoss]):
                Optional list of losses to be applied in this execution step.

        Raises:
            RuntimeError: If stage is not frozen.

        """
        # If stage is not frozen, raise error
        if self.is_frozen:
            msg = "Cannot evaluate an unfrozen node. Either freeze or use `train_step`."
            raise RuntimeError(msg)

        # Ensure input data exists for this node
        self._validate_ctx(ctx=ctx)
        # Ensure losses only include those applied to this node
        valid_losses = losses
        if losses is not None:
            valid_losses = [loss for loss in losses if loss.node_id == self.node_id]

        if self.backend == Backend.TORCH:
            return self._eval_step_torch(ctx=ctx, losses=valid_losses)

        if self.backend == Backend.TENSORFLOW:
            return self._eval_step_tensorflow(ctx=ctx, losses=valid_losses)

        if self.backend == Backend.SCIKIT:
            return self._eval_step_scikit(ctx=ctx, losses=valid_losses)

        msg = f"Unknown backend: {self.backend}"
        raise ValueError(msg)

    # ================================================
    # Fittable Protocol
    # ================================================
    def fit_step(
        self,
        ctx: ExecutionContext,
        losses: list[AppliedLoss] | None = None,
    ):
        """
        Fits this node on complete data (for batch-fit scikit-learn models).

        Calls the underlying model's `.fit(X, y)` method using the full
        dataset provided in the execution context. After fitting, a forward
        pass is performed to record outputs for downstream nodes.

        Args:
            ctx (ExecutionContext):
                Context containing full-dataset inputs.
            losses (list[AppliedLoss] | None):
                Optional losses to compute after fitting (for metrics only).

        Raises:
            RuntimeError: If this node is frozen.

        """
        if self.is_frozen:
            msg = f"Cannot fit a frozen node '{self.label}'."
            raise RuntimeError(msg)
        if not hasattr(self, "fit"):
            msg = f"Node `{self.label}` does not implement a `.fit()` method."
            raise AttributeError(msg)

        self._validate_ctx(ctx=ctx)

        # Get input batch
        input_batch: Batch = self._get_input_batch(ctx=ctx)

        # Merge data from all roles, then fit on joint set
        joint_sd = SampleData.concat(
            *list(input_batch.role_data.values()),
            fmt=get_data_format_for_backend(self.backend),
        )
        # Perform incremental fit on this merged data
        self._model.fit(
            joint_sd.features,
            joint_sd.targets,
        )

        # Forward pass to record outputs for downstream nodes
        out_batch: Batch = self.forward_single(input_batch)
        ctx.set_output(node_id=self.node_id, batch=out_batch)

        # Optional loss computation (auxiliary only)
        if losses is not None:
            valid_losses = [loss for loss in losses if loss.node_id == self.node_id]
            loss_records: list[LossRecord] = []
            for loss in valid_losses:
                weighted_raw_loss = loss.compute(ctx=ctx)
                lr = LossRecord(
                    label=loss.label,
                    node_id=loss.node_id,
                    auxiliary=weighted_raw_loss,
                )
                loss_records.append(lr)
            ctx.add_losses(LossCollection(records=loss_records))

    # ================================================
    # Configurable
    # ================================================
    def get_config(self) -> dict[str, Any]:
        """
        Retrieve the configuration details of this ModelNode instance.

        This does not contain state information of the underlying model or optimizer.
        """
        cfg = super().get_config()
        cfg.update(
            {
                "model": self._model.get_config(),
                "optimizer": None
                if self._optimizer is None
                else self._optimizer.get_config(),
                "frozen": self._freeze,
                "graph_node_type": "ModelNode",
            },
        )
        return cfg

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        *,
        register: bool = True,
    ) -> ModelNode:
        """
        Reconstructs a ModelNode from configuration details.

        This does not restore state information of the underlying model or optimizer.
        """
        if "graph_node_type" not in config or config["graph_node_type"] != "ModelNode":
            raise ValueError("Invalid config data for ModelNode.")

        # Rebuild model (no weights)
        model = BaseModel.from_config(config["model"])

        # Rebuild optimizer
        optimizer = None
        optimizer_cfg = config.get("optimizer")
        if optimizer_cfg is not None:
            optimizer = Optimizer.from_config(optimizer_cfg)

        # Create ModelNode
        node = cls(
            label=config["label"],
            model=model,
            upstream_ref=config["upstream_refs"][0]
            if config["upstream_refs"]
            else None,
            optimizer=optimizer,
            node_id=config.get("node_id"),
            register=register,
        )

        # Restore downstream refs explicitly
        node.set_downstream_refs(config.get("downstream_refs", []))

        # Restore frozen flag
        node._freeze = config.get("frozen", False)

        return node

    # ================================================
    # Stateful
    # ================================================
    def get_state(self) -> dict[str, Any]:
        """
        Return serialized state for the node, model, and optimizer.

        Returns:
            dict[str, Any]: Snapshot captured for :meth:`set_state`.

        """
        state = {
            "super": super().get_state(),
            "model": self._model.get_state(),
            "optimizer": None
            if self._optimizer is None
            else self._optimizer.get_state(),
            "frozen": self._freeze,
        }
        return state

    def set_state(self, state: dict[str, Any]) -> None:
        """
        Restore runtime state from :meth:`get_state` output.

        Args:
            state (dict[str, Any]): Serialized node data.

        """
        # Set parent state first
        super().set_state(state["super"])

        # Model weights can always be restored
        self._model.set_state(state["model"])

        # Optimizer state may need to wait until build()
        if self._optimizer is not None and state.get("optimizer") is not None:
            self._optimizer.set_state(state["optimizer"])

        # Restore freeze state (must re-apply to sync backend parameters)
        if state.get("frozen", False):
            self.freeze()
        else:
            self.unfreeze()
