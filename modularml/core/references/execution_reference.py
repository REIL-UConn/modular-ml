from typing import Any, TypeAlias

from modularml.core.data.execution_context import ExecutionContext
from modularml.core.references.experiment_reference import ExperimentReference

TensorLike: TypeAlias = Any


class ExecutionReference(ExperimentReference):
    """Reference that resolves against a single execution step."""

    def resolve(self, ctx: ExecutionContext) -> TensorLike:
        if not isinstance(ctx, ExecutionContext):
            msg = f"Context must be an ExecutionContext. Received: {type(ctx)}."
            raise TypeError(msg)

        return self._resolve_execution(ctx=ctx)

    def _resolve_execution(self, ctx: ExecutionContext):
        raise NotImplementedError
