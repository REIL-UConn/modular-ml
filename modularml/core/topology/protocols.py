"""Protocol interfaces implemented by topology nodes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, TypeVar, runtime_checkable

from modularml.utils.data.data_format import DataFormat

if TYPE_CHECKING:
    from modularml.core.data.execution_context import ExecutionContext
    from modularml.core.references.experiment_reference import ExperimentNodeReference
    from modularml.core.references.featureset_reference import FeatureSetReference
    from modularml.core.training.applied_loss import AppliedLoss
    from modularml.utils.nn.backend import Backend

# ================================================
# Forwardable
# ================================================
T = TypeVar("T")


@runtime_checkable
class Forwardable(Protocol[T]):
    """A node that can perform a forward computation."""

    @property
    def backend(self) -> Backend:
        """
        The required data backend for forward pass execution.

        Returns:
            Backend: TORCH, TENSORFLOW, SCIKIT, ...

        """
        ...

    def forward(self, inputs: dict[ExperimentNodeReference, T], **kwargs) -> T:
        """
        Perform forward computation through the node.

        Args:
            inputs (dict[ExperimentNodeReference, T]):
                Input data to perform a forward pass on.
            **kwargs: Additional keyword arguments specific to each
                implementation.

        Returns:
            T:
                The output data matches the type of the input values.

        """
        ...

    def get_input_data(
        self,
        inputs: dict[tuple[str, FeatureSetReference], T],
        outputs: dict[str, T],
        *,
        fmt: DataFormat = DataFormat.NUMPY,
    ) -> dict[ExperimentNodeReference, T]:
        """
        Retrieve input data for this node at the current execution step.

        Args:
            inputs (dict[tuple[str, FeatureSetReference], T]): Materialized
                data sourced from :class:`FeatureSetReference` instances.
            outputs (dict[str, T]): Cached results from upstream nodes
                keyed by :attr:`GraphNode.node_id`.
            fmt (DataFormat): Output format requested when resolving
                :class:`BatchView` objects.

        Returns:
            dict[ExperimentNodeReference, T]: Data keyed by each upstream
                reference of this node.

        """
        ...


# ================================================
# Evaluable
# ================================================
@runtime_checkable
class Evaluable(Forwardable[T], Protocol):
    """A node that supports evaluation (forward + loss, no grads)."""

    def eval_step(
        self,
        ctx: ExecutionContext,
        losses: list[AppliedLoss] | None = None,
    ) -> None:
        """
        Run evaluation logic for the node.

        Args:
            ctx (ExecutionContext):
                Execution context containing batch data and intermediate caches.
            losses (list[AppliedLoss] | None):
                Loss functions evaluated during the step. When omitted, only
                forward outputs are materialized.

        """
        ...


# ================================================
# Trainable
# ================================================
@runtime_checkable
class Trainable(Evaluable[T], Protocol):
    """A node that supports gradient-based training."""

    @property
    def is_frozen(self) -> bool:
        """
        Indicates whether this object is frozen (not trainable).

        Returns:
            bool: True if frozen, False if trainable.

        """
        ...

    def freeze(self, *args, **kwargs):
        """
        Freeze the trainable state to disable gradient updates.

        Args:
            *args: Positional arguments forwarded to implementations.
            **kwargs: Keyword arguments forwarded to implementations.

        """
        ...

    def unfreeze(self, *args, **kwargs):
        """
        Unfreeze the trainable state to enable gradient updates.

        Args:
            *args: Positional arguments forwarded to implementations.
            **kwargs: Keyword arguments forwarded to implementations.

        """
        ...

    def train_step(
        self,
        ctx: ExecutionContext,
        losses: list[AppliedLoss],
    ) -> None:
        """
        Execute a full training step including loss/optimizer updates.

        Args:
            ctx (ExecutionContext):
                Execution context that supplies batches, samplers, and caches.
            losses (list[AppliedLoss]):
                Loss objects to evaluate and aggregate during the step.

        """
        ...


# ================================================
# Fittable
# ================================================
@runtime_checkable
class Fittable(Forwardable[T], Protocol):
    """A node that supports batch fitting (e.g., scikit-learn `.fit()`)."""

    @property
    def is_frozen(self) -> bool:
        """
        Indicates whether this object is frozen (not fittable).

        Returns:
            bool: True if frozen, False if fittable.

        """
        ...

    def freeze(self, *args, **kwargs):
        """
        Freeze the fittable state to prevent fit updates.

        Args:
            *args: Positional arguments forwarded to implementations.
            **kwargs: Keyword arguments forwarded to implementations.

        """
        ...

    def unfreeze(self, *args, **kwargs):
        """
        Unfreeze the fittable state to allow fit updates.

        Args:
            *args: Positional arguments forwarded to implementations.
            **kwargs: Keyword arguments forwarded to implementations.

        """
        ...

    def fit_step(
        self,
        ctx: ExecutionContext,
        losses: list[AppliedLoss] | None = None,
    ) -> None:
        """
        Execute a fitting iteration for batch-oriented estimators.

        Args:
            ctx (ExecutionContext):
                Execution context with current batch.
            losses (list[AppliedLoss] | None):
                Optional loss functions to compute once fitting completes.

        """
        ...
