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
            **kwargs: Additional key-word arguments specific to each subclass.

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
        Retrieves input data for this node at the current execution step.

        Returns:
            dict[ExperimentNodeReference, T]:
                Data keyed by each upstream_ref of this node.

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
    ) -> None: ...


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
        """Freezes the trainable state (prevents training updates)."""
        ...

    def unfreeze(self, *args, **kwargs):
        """Unfreezes the trainable state (allows training updates)."""
        ...

    def train_step(
        self,
        ctx: ExecutionContext,
        losses: list[AppliedLoss],
    ) -> None: ...


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
        """Freezes the fittable state (prevents fitting updates)."""
        ...

    def unfreeze(self, *args, **kwargs):
        """Unfreezes the fittable state (allows fitting updates)."""
        ...

    def fit_step(
        self,
        ctx: ExecutionContext,
        losses: list[AppliedLoss] | None = None,
    ) -> None: ...
