from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from modularml.context.execution_context import ExecutionContext
from modularml.core.references.execution_reference import ExecutionReference
from modularml.core.references.experiment_reference import ResolutionError


@dataclass(frozen=True)
class ModelOutputReference(ExecutionReference):
    # ModelNode-specifiers
    node_label: str
    node_id: str

    # IO-specifiers
    role: str | None = None
    domain: Literal["outputs"] = "outputs"

    def _resolve_execution(self, ctx: ExecutionContext):
        if not isinstance(ctx, ExecutionContext):
            msg = f"Context must be either an ExecutionContext. Received: {type(ctx)}."
            raise TypeError(msg)
        if self.node_id not in ctx.outputs:
            msg = "The referenced node does not exist in the given context."
            raise ResolutionError(msg)

        # Get batch from context
        batch_output = ctx.outputs[self.node_id]

        # Get role data
        role = self.role
        if role is None:
            if len(batch_output.available_roles) != 1:
                msg = (
                    "ModelOutputReference must specify a `role` when multiple "
                    "roles exist in the output data. "
                    f"Available roles: {batch_output.available_roles}."
                )
                raise ResolutionError(msg)
            role = batch_output.available_roles[0]

        # Get output data (domain=outputs)
        return batch_output.get_data(
            role=role,
            domain=self.domain,
        )

    # ================================================
    # Configurable
    # ================================================
    def get_config(self) -> dict[str, Any]:
        """
        Return configuration required to reconstruct this reference.

        Returns:
            dict[str, Any]: Reference configuration.

        """
        return {
            "node_label": self.node_label,
            "node_id": self.node_id,
            "role": self.role,
            "domain": self.domain,
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> ModelOutputReference:
        """
        Construct a reference from configuration.

        Args:
            config (dict[str, Any]): Reference configuration.

        Returns:
            ReferenceLike: Reconstructed reference..

        """
        return cls(**config)
