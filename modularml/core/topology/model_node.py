from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, overload

from modularml.core.data.batch import Batch
from modularml.core.data.featureset import FeatureSet
from modularml.core.data.featureset_view import FeatureSetView
from modularml.core.data.sample_data import RoleData, SampleData
from modularml.core.experiment.experiment_node import ExperimentNode
from modularml.core.models import wrap_model
from modularml.core.references.experiment_reference import ExperimentNodeReference
from modularml.core.topology.compute_node import ComputeNode
from modularml.core.topology.mixins.evaluable import EvaluableMixin
from modularml.core.topology.mixins.trainable import TrainableMixin
from modularml.core.topology.node_shapes import NodeShapes
from modularml.core.training.loss_record import LossCollection, LossRecord
from modularml.utils.environment.optional_imports import check_tensorflow, check_torch
from modularml.utils.errors.exceptions import BackendMismatchError, BackendNotSupportedError, OptimizerNotSetError
from modularml.utils.logging.warnings import catch_warnings, warn
from modularml.utils.nn.backend import Backend
from modularml.utils.representation.summary import safe_cast_to_summary_rows

if TYPE_CHECKING:
    from modularml.context.resolution_context import ResolutionContext
    from modularml.core.models.base_model import BaseModel
    from modularml.core.training.applied_loss import AppliedLoss
    from modularml.core.training.optimizer import Optimizer

tf = check_tensorflow()
torch = check_torch()


class ModelNode(ComputeNode, TrainableMixin, EvaluableMixin):
    """
    A ModelNode represents a single learnable or non-learnable transformation block in a ModelGraph.

    It wraps a backend-specific model (e.g., PyTorch, TensorFlow, or Scikit-learn) and optionally includes an Optimizer.
    A ModelNode receives data from a single input source (FeatureSet or another ModelNode), and produces an output
    which can be consumed by downstream stages or used directly for loss computation.

    If an optimizer is attached, `train_step()` and `eval_step()` can be called directly for this stage. Otherwise,
    training and evaluation should be managed by a parent ModelGraph that handles multiple stages.
    """

    def __init__(
        self,
        label: str,
        model: BaseModel | Any,
        upstream_ref: ExperimentNode | ExperimentNodeReference,
        optimizer: Optimizer | None = None,
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

        """
        ref = None
        if isinstance(upstream_ref, FeatureSet):
            dup_rep_warnings = False
            with catch_warnings() as cw:
                upstream_ref.reference()
                if cw.match("Multiple representations selected"):
                    dup_rep_warnings = True
            if dup_rep_warnings:
                msg = (
                    "Setting a ModelNode `upstream_ref` with a FeatureSet will result in multiple "
                    "representations of the same column being combined into input/target tensors. "
                )
                hint = "Use `FeatureSet(...).reference(...)` is this is not intentional."
                warn(msg, category=UserWarning, stacklevel=2, hints=hint)

            # with warnings.catch_warnings(record=True) as w:
            #     warnings.simplefilter("always", UserWarning)
            #     ref = upstream_ref.reference()
            #     dup_rep_warnings = [ww for ww in w if "Multiple representations selected" in str(ww.message)]
            # if dup_rep_warnings:
            #     msg = (
            #         "Setting a ModelNode `upstream_ref` with a FeatureSet will result in multiple "
            #         "representations of the same column being combined into input/target tensors. "
            #     )
            #     hint = "Use `FeatureSet(...).reference(...)` is this is not intentional."
            #     warn(msg, category=UserWarning, stacklevel=2, hints=hint)
        elif isinstance(upstream_ref, ExperimentNodeReference):
            ref = upstream_ref
        elif isinstance(upstream_ref, ExperimentNode):
            ref = upstream_ref.reference()
        else:
            msg = (
                f"`upstream_ref` must be of type ExperimentReference or ExperimentNode. Received: {type(upstream_ref)}."
            )
            raise TypeError(msg)

        super().__init__(label=label, upstream_refs=ref)

        # Set model (cast to BaseModel if explicit subclass not provided)
        self._model: BaseModel = wrap_model(model)
        self._freeze = False  # make stage trainable as default

        # Error checking on optimizer (can be None)
        self._optimizer = optimizer
        self._check_valid_optimizer(required=False)

    # ================================================
    # ComputeNode Interface
    # ================================================
    @property
    def node_shapes(self) -> NodeShapes:
        """
        Input and output shapes expected by this ModelNode.

        Description:
            Defines the input and output shapes after model building.
            Model must be built before this property can be accessed.

        """
        if self._model.input_shape is None:
            if self.is_built:
                raise RuntimeError("Input shape is None after model building.")
            raise RuntimeError("Model must be built before accessing input_shape")

        if self._model.output_shape is None:
            if self.is_built:
                raise RuntimeError("Output shape is None after model building.")
            raise RuntimeError("Model must be built before accessing output_shape")

        # ModelNodes only support single input, single output
        return NodeShapes(
            input_shapes={0: self._model.input_shape},
            output_shapes={0: self._model.output_shape},
        )

    @property
    def max_upstream_refs(self) -> int:
        return 1

    @property
    def is_built(self) -> bool:
        """
        Checks if the model has been built (i.e., instantiated with input/output shape).

        Returns:
            bool: True if built, False otherwise.

        """
        return self._model.is_built

    def infer_output_shape(
        self,
        input_shapes: dict[str, tuple[int, ...]],
    ) -> dict[str, tuple[int, ...]]:
        """
        Infer the expected output shape of this ModelNode without building the backend model.

        Description:
            This method is used during graph construction to determine the shape of data
            produced by this stage, based on its input shape(s) and internal configuration.

            Expected behavior:
            - If `self.output_shape` is already defined, it will be returned directly.
            - If the underlying `BaseModel` defines a static method `infer_output_shape(input_shape)`,
            it will be used to compute the output shape from the given input.
            - If the output shape cannot be inferred, an error is raised.

        Note that this method:
            - Assumes the stage only supports a single input; an error will be raised
                if multiple input shapes are provided.
            - Does NOT instantiate or build the backend model.

        Args:
            input_shapes (dict[str, tuple[int, ...]]):
                Input shapes from upstream connections. ModelNode's expect exactly one element.

        Returns:
            dict[str, tuple[int, ...]]: The inferred output shape (one element).

        Raises:
            ValueError: If multiple input shapes are provided or output shape cannot be inferred.

        """
        # Get only input tuple (drop keys, but sort by them for reproducibility)
        inp_shapes: list[tuple[int, ...]] = [v for k, v in sorted(input_shapes.items())]
        if self.max_upstream_refs is not None and len(inp_shapes) > self.max_upstream_refs:
            msg = f"ModelNode only support a single input. Received {len(inp_shapes)}: {input_shapes}"
            raise ValueError(msg)
        inp_shape: tuple[int, ...] = inp_shapes[0]

        # Return output_shape is already known
        try:
            out_shapes: dict[str, tuple[int, ...]] = self.node_shapes.output_shapes
            if out_shapes is not None:
                return out_shapes
        except RuntimeError:
            pass

        # Pass inferencing task to BaseModel (if supports it)
        meth = getattr(self._model, "infer_output_shapes", None)  # returns list of shapes
        if callable(meth):
            out_shapes: dict[str, tuple[int, ...]] = dict(enumerate(meth(inp_shape)))
            if len(out_shapes) > 1:
                msg = f"Detected more than 1 inferred output shape: {out_shapes}"
                raise RuntimeError(msg)
            return out_shapes

        meth = getattr(self._model, "infer_output_shape", None)  # returns single shape
        if callable(meth):
            return {0: meth(inp_shape)}

        # Otherwise, raise error
        msg = f"Cannot infer output shape for ModelNode `{self.label}`."
        raise ValueError(msg)

    def _build_optimizer(self, *, force: bool = False):
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

    def build(
        self,
        input_shape: tuple[int, ...] | None = None,
        output_shape: tuple[int, ...] | None = None,
        *,
        force: bool = False,
    ):
        """
        Build the ModelNode by initializing the underlying BaseModel and its optimizer.

        Args:
            input_shape (tuple[int, ...] | None, optional):
                Input shape to construct this model with.
                Defaults to None.

            output_shape (tuple[int, ...] | None, optional):
                Output shape to construct this model with. If not provided, the BaseModel
                must be capable of inferring it internally or during construction.
                Defaults to None.

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
    def forward(self, batch: Batch, **kwargs) -> Batch: ...
    @overload
    def forward(self, roles: RoleData, **kwargs) -> RoleData: ...
    @overload
    def forward(self, data: SampleData, **kwargs) -> SampleData: ...
    def forward(
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

        def _forward_sample_data(d: SampleData) -> SampleData:
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
            out = RoleData(data={k: _forward_sample_data(v) for k, v in x.role_data.items()})

            return Batch(
                batch_size=x.batch_size,
                role_data=out,
                shapes=out.shapes,
                role_weights=x.role_weights,
                role_masks=x.role_masks,
            )

        msg = f"Input must be of type SampleData or RoleData or Batch. Received: {type(x)}"
        raise TypeError(msg)

    # ================================================
    # Representation
    # ================================================
    def _summary_rows(self) -> list[tuple]:
        def _reduce_dict(x: dict) -> str | list[tuple]:
            if len(x) == 1:
                k = next(iter(x.keys()))
                return str(x[k])
            return [(str(k), str(v)) for k, v in x.items()]

        return [
            ("label", self.label),
            ("upstream_ref", safe_cast_to_summary_rows(self.upstream_ref)),
            ("downstream_refs", [safe_cast_to_summary_rows(r) for r in self._downstream_refs]),
            ("input_shape", _reduce_dict(self.input_shapes) if self.is_built else "NOT BUILT YET"),
            ("output_shapes", _reduce_dict(self.output_shapes) if self.is_built else "NOT BUILT YET"),
            ("model", safe_cast_to_summary_rows(self._model)),
            ("optimizer", safe_cast_to_summary_rows(self._optimizer)),
            ("backend", safe_cast_to_summary_rows(self.backend)),
            ("frozen", f"{'True' if self.freeze else 'False'}"),
        ]

    def __repr__(self):
        return (
            f"ModelNode(label='{self.label}', "
            f"upstream_refs={self._upstream_refs}, "
            f"downstream_refs={self._downstream_refs}, "
            f"model={self._model!r}, "
            f"optimizer={self._optimizer}, "
            f"backend={self.backend})"
        )

    def __str__(self):
        return f"ModelNode('{self.label}')"

    # ================================================
    # ModelNodes Properties & Dunders
    # ================================================
    @overload
    def __call__(self, batch: Batch, **kwargs) -> Batch: ...
    @overload
    def __call__(self, roles: RoleData, **kwargs) -> RoleData: ...
    @overload
    def __call__(self, data: SampleData, **kwargs) -> SampleData: ...
    def __call__(
        self,
        x: SampleData | RoleData | Batch,
        **kwargs,
    ) -> SampleData | RoleData | Batch:
        """
        Shorthand to call `forward()` on input.

        Args:
            x (SampleData | RoleData | Batch): Input data to the model.
            **kwargs: Any additional keyword arguments to provide to BaseModel.forward

        Returns:
            SampleData | RoleData | Batch:
                Outputs from the model. Output type matches input.

        """
        return self.forward(x=x, **kwargs)

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

    def _validate_source(
        self,
        ctx: ResolutionContext,
        losses: list[AppliedLoss] | None = None,
    ):
        """
        Validates that the required sources for input and losses are present in `ctx`.

        Args:
            ctx (ResolutionContext):
                Execution context to validate losses on.
            losses (list[AppliedLoss], optional):
                List of applied losses.

        Raises:
            ValueError: If any expected input or loss role is missing.

        """
        # Check that self.upstream_ref exists in batch
        ups_node: ExperimentNode = self.upstream_ref.resolve(ctx=ctx)
        if isinstance(ups_node, (FeatureSet, FeatureSetView)):
            # This node must have direct inputs
            if self.node_id not in ctx.execution.input_batches:
                msg = f"ExecutionContext missing input data for ModelNode '{self.label}'."
                raise ValueError(msg)
        elif ups_node.node_id not in ctx.execution.model_outputs:
            msg = f"ExecutionContext missing output data from upstream node '{ups_node.label}'."
            raise ValueError(msg)

    # ================================================
    # Trainable Mixin
    # ================================================
    @property
    def model(self) -> BaseModel:
        return self._model

    @property
    def backend(self) -> Backend:
        """
        Returns the backend associated with the wrapped model.

        Returns:
            Backend: TORCH, TENSORFLOW, SCIKIT, ...

        """
        return self._model.backend

    @property
    def freeze(self) -> bool:
        """
        Indicates whether this stage is frozen (not trainable).

        Returns:
            bool: True if frozen, False if trainable.

        """
        return self._freeze

    @freeze.setter
    def freeze(self, value: bool):
        """
        Sets the stage to be frozen (non-trainable) or trainable.

        Args:
            value (bool): True to freeze, False to unfreeze.

        Raises:
            ValueError: If value is not a boolean.

        """
        if not isinstance(value, bool):
            msg = f"Freeze must be a boolean, got {type(value)}"
            raise TypeError(msg)
        self._freeze = value

    def get_input_data(
        self,
        ctx: ResolutionContext,
    ) -> Batch:
        """Retrieves Batch data for this ModelNode at the current execution step."""
        # If head node, get input
        if self.node_id in ctx.execution.input_batches:
            return ctx.execution.input_batches[self.node_id]
        # Otherwise, get output of upstream node
        upstream_node: ExperimentNode = self.upstream_ref.resolve(ctx=ctx)
        if upstream_node.node_id in ctx.execution.model_outputs:
            return ctx.execution.model_outputs[upstream_node.node_id]

        msg = f"Failed to get input data for ModelNode '{self.label}'."
        raise RuntimeError(msg)

    def _train_step_torch(
        self,
        ctx: ResolutionContext,
        losses: list[AppliedLoss],
    ):
        """
        Runs a training step using PyTorch: forward, loss, backward, optimizer.

        Args:
            ctx (ResolutionContext):
                Context (input/output data) for the given execution step.
            losses (list[AppliedLoss]):
                List of losses to be applied in this execution step.

        """
        # Set optimizer and train mode
        self._model.train()
        self._optimizer.zero_grad()

        loss_records: list[LossRecord] = []

        # Forward pass (ctx.execution modified inplace)
        out_batch: Batch = self.forward(self.get_input_data(ctx=ctx))
        if ctx.execution.model_outputs is None:
            ctx.execution.model_outputs = {}
        ctx.execution.model_outputs[self.node_id] = out_batch

        # Compute losses
        if losses is not None:
            for loss in losses:
                weighted_raw_loss = loss.compute(ctx=ctx)
                lr = LossRecord(
                    value=weighted_raw_loss,
                    label=loss.label,
                    contributes_to_update=True,
                )
                loss_records.append(lr)

        # Backward + opt step
        lc = LossCollection(records=loss_records)
        lc.trainable.backward()
        self._optimizer.step()

        if ctx.execution.model_losses is None:
            ctx.execution.model_losses = {}
        ctx.execution.model_losses[self.node_id] = lc

    def _train_step_tensorflow(
        self,
        ctx: ResolutionContext,
        losses: list[AppliedLoss],
    ):
        """
        Runs a training step using Tensorflow: forward, loss, backward, optimizer.

        Args:
            ctx (ResolutionContext):
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
            out_batch: Batch = self.forward(
                self.get_input_data(ctx=ctx),
                training=True,
            )
            if ctx.execution.model_outputs is None:
                ctx.execution.model_outputs = {}
            ctx.execution.model_outputs[self.node_id] = out_batch

        # Compute losses
        if losses is not None:
            for loss in losses:
                weighted_raw_loss = loss.compute(ctx=ctx)
                lr = LossRecord(
                    value=weighted_raw_loss,
                    label=loss.label,
                    contributes_to_update=True,
                )
                loss_records.append(lr)

        # Backward + opt step
        lc = LossCollection(records=loss_records)
        grads = tape.gradient(lc.total, self._model.trainable_variables)
        self._optimizer.step(grads=grads, variables=self._model.trainable_variables)

        if ctx.execution.model_losses is None:
            ctx.execution.model_losses = {}
        ctx.execution.model_losses[self.node_id] = lc

    def _train_step_scikit(
        self,
        ctx: ResolutionContext,
        losses: list[AppliedLoss],
    ):
        # TODO:
        raise NotImplementedError("Training for scikit model not implemented yet.")

    def train_step(
        self,
        ctx: ResolutionContext,
        losses: list[AppliedLoss],
    ):
        """
        Performs a training step (forward, loss, backward, optimizer step) for this stage.

        Only callable if this stage has an optimizer and is not frozen. Otherwise, training
        must be delegated to `ModelGraph`.

        Args:
            ctx (ResolutionContext):
                Context (input/output data) for the given execution step.
            losses (list[AppliedLoss]):
                List of losses to be applied in this execution step.

        Raises:
            RuntimeError: If stage is frozen or optimizer is missing.

        """
        # If stage is frozen, raise error
        if self.freeze:
            raise RuntimeError(
                "`train_step` called with `freeze=True`. Use `eval_step` instead.",
            )

        # Ensure batch_input matches expected data in losses
        self._validate_source(ctx=ctx, losses=losses)

        # Ensure optimizer is set and matches model backend
        self._check_valid_optimizer(required=True)

        if self.backend == Backend.TORCH:
            return self._train_step_torch(ctx=ctx, losses=losses)

        if self.backend == Backend.TENSORFLOW:
            return self._train_step_tensorflow(ctx=ctx, losses=losses)

        if self.backend == Backend.SCIKIT:
            return self._train_step_scikit(ctx=ctx, losses=losses)

        msg = f"Unknown backend: {self.backend}"
        raise ValueError(msg)

    # ================================================
    # Evaluable Mixin
    # ================================================
    def _eval_step_torch(
        self,
        ctx: ResolutionContext,
        losses: list[AppliedLoss] | None = None,
    ):
        """
        Runs an evaluation step using PyTorch: forward + loss (no gradients).

        Args:
            ctx (ResolutionContext):
                Context (input/output data) for the given execution step.
            losses (list[AppliedLoss]):
                Optional list of losses to be applied in this execution step.

        """
        # Set eval mode
        self._model.eval()

        loss_records: list[LossRecord] = []

        # Forward pass (ctx.execution modified inplace)
        with torch.no_grad():
            out_batch: Batch = self.forward(self.get_input_data(ctx=ctx))
            if ctx.execution.model_outputs is None:
                ctx.execution.model_outputs = {}
            ctx.execution.model_outputs[self.node_id] = out_batch

            # Compute losses
            if losses is not None:
                for loss in losses:
                    weighted_raw_loss = loss.compute(ctx=ctx)
                    lr = LossRecord(
                        value=weighted_raw_loss,
                        label=loss.label,
                        contributes_to_update=False,  # not used in opt. stepping
                    )
                    loss_records.append(lr)

        # Record loss records
        lc = LossCollection(records=loss_records)
        if ctx.execution.model_losses is None:
            ctx.execution.model_losses = {}
        ctx.execution.model_losses[self.node_id] = lc

    def _eval_step_tensorflow(
        self,
        ctx: ResolutionContext,
        losses: list[AppliedLoss] | None = None,
    ):
        """
        Runs an evaluation step using Tensorflow: forward + loss (no gradients).

        Args:
            ctx (ResolutionContext):
                Context (input/output data) for the given execution step.
            losses (list[AppliedLoss]):
                Optional list of losses to be applied in this execution step.

        """
        loss_records: list[LossRecord] = []

        # Forward pass (ctx.execution modified inplace)
        out_batch: Batch = self.forward(
            self.get_input_data(ctx=ctx),
            training=False,
        )
        if ctx.execution.model_outputs is None:
            ctx.execution.model_outputs = {}
        ctx.execution.model_outputs[self.node_id] = out_batch

        # Compute losses
        if losses is not None:
            for loss in losses:
                weighted_raw_loss = loss.compute(ctx=ctx)
                lr = LossRecord(
                    value=weighted_raw_loss,
                    label=loss.label,
                    contributes_to_update=False,  # not used in opt. stepping
                )
                loss_records.append(lr)

        # Record loss records
        lc = LossCollection(records=loss_records)
        if ctx.execution.model_losses is None:
            ctx.execution.model_losses = {}
        ctx.execution.model_losses[self.node_id] = lc

    def _eval_step_scikit(
        self,
        ctx: ResolutionContext,
        losses: list[AppliedLoss] | None = None,
    ):
        # TODO:
        raise NotImplementedError("Evaluation for scikit model not implemented yet.")

    def eval_step(
        self,
        ctx: ResolutionContext,
        losses: list[AppliedLoss] | None = None,
    ):
        """
        Performs an evaluation step (forward pass and loss computation) for this stage.

        Only callable if this stage is frozen. No gradient tracking is performed.

        Args:
            ctx (ResolutionContext):
                Context (input/output data) for the given execution step.
            losses (list[AppliedLoss]):
                Optional list of losses to be applied in this execution step.

        Raises:
            RuntimeError: If stage is not frozen.

        """
        # If stage is not frozen, raise error
        if not self.freeze:
            raise RuntimeError(
                "`eval_step` called with `freeze=False`. Use `train_step` for training.",
            )

        self._validate_source(ctx=ctx, losses=losses)

        if self.backend == Backend.TORCH:
            return self._eval_step_torch(ctx=ctx, losses=losses)

        if self.backend == Backend.TENSORFLOW:
            return self._eval_step_tensorflow(ctx=ctx, losses=losses)

        if self.backend == Backend.SCIKIT:
            return self._eval_step_scikit(ctx=ctx, losses=losses)

        msg = f"Unknown backend: {self.backend}"
        raise ValueError(msg)
