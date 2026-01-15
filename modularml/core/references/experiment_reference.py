from __future__ import annotations

from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any

from modularml.context.experiment_context import ExperimentContext
from modularml.core.io.protocols import Configurable
from modularml.core.references.reference_like import ReferenceLike

if TYPE_CHECKING:
    from modularml.core.experiment.experiment_node import ExperimentNode
    from modularml.core.topology.graph_node import GraphNode


class ResolutionError(RuntimeError):
    pass


@dataclass(frozen=True)
class ExperimentReference(ReferenceLike, Configurable):
    """Base class for references resolvable at the Experiment scope."""

    def resolve(self, ctx: ExperimentContext | None = None):
        if ctx is None:
            ctx = ExperimentContext.get_active()
        return self._resolve_experiment(ctx)

    def _resolve_experiment(self, ctx: ExperimentContext):
        raise NotImplementedError

    def to_string(
        self,
        *,
        separator: str = ".",
        include_node_id: bool = False,
    ) -> str:
        """
        Joins all non-null fields into a single string.

        Example:
        ``` python
            ref = DataReference(node='PulseFeatures', domain='features', key='voltage')
            ref.to_string()
            # 'PulseFeatures.features.voltage'
        ```

        """
        attrs = {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if getattr(self, f.name) is not None
            and (f.name != "node_id" or include_node_id)
        }
        return separator.join(v for v in attrs.values())

    # ================================================
    # Configurable
    # ================================================
    def get_config(self) -> dict[str, Any]:
        """Return a JSON-serializable configuration."""
        raise NotImplementedError

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> ExperimentReference:
        """Reconstructs the reference from config."""
        raise NotImplementedError


@dataclass(frozen=True)
class ExperimentNodeReference(ExperimentReference):
    """Reference to an ExperimentNode by label or id."""

    node_label: str | None = None
    node_id: str | None = None

    def resolve(
        self,
        ctx: ExperimentContext | None = None,
    ) -> ExperimentNode:
        """Resolves this reference to a ExperimentNode instance."""
        return super().resolve(ctx=ctx)

    def _resolve_experiment(
        self,
        ctx: ExperimentContext,
    ) -> ExperimentNode:
        if not isinstance(ctx, ExperimentContext):
            msg = (
                "ExperimentNodeReference requires an ExperimentContext."
                f"Received: {type(ctx)}."
            )
            raise TypeError(msg)

        # Prefer node_id resolution if given
        if self.node_id is not None:
            if not ctx.has_node(node_id=self.node_id):
                msg = (
                    f"No node exists with ID='{self.node_id}' in the given "
                    "ExperimentContext."
                )
                raise ResolutionError(msg)
            return ctx.get_node(
                node_id=self.node_id,
                enforce_type="ExperimentNode",
            )

        # Fallback to node label
        if self.node_label is not None:
            if not ctx.has_node(label=self.node_label):
                msg = (
                    f"No node exists with label='{self.node_label}' in the given "
                    "ExperimentContext."
                )
                raise ResolutionError(msg)
            return ctx.get_node(
                label=self.node_label,
                enforce_type="ExperimentNode",
            )

        raise ResolutionError("Both node_label and node_id cannot be None.")

    # ================================================
    # Configurable
    # ================================================
    def get_config(self) -> dict[str, Any]:
        """Return a JSON-serializable configuration."""
        return {
            "node_id": self.node_id,
            "node_label": self.node_label,
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> ExperimentReference:
        """Reconstructs the reference from config."""
        return cls(**config)


@dataclass(frozen=True)
class GraphNodeReference(ExperimentNodeReference):
    """Reference to an GraphNode by label or id."""

    node_label: str | None = None
    node_id: str | None = None

    def resolve(
        self,
        ctx: ExperimentContext | None = None,
    ) -> GraphNode:
        """Resolves this reference to a GraphNode instance."""
        return super().resolve(ctx=ctx)

    def _resolve_experiment(
        self,
        ctx: ExperimentContext,
    ) -> GraphNode:
        if not isinstance(ctx, ExperimentContext):
            msg = (
                "GraphNodeReference requires an ExperimentContext."
                f"Received: {type(ctx)}."
            )
            raise TypeError(msg)

        # Prefer node_id resolution if given
        if self.node_id is not None:
            if not ctx.has_node(node_id=self.node_id):
                msg = (
                    f"No node exists with ID='{self.node_id}' in the given "
                    "ExperimentContext."
                )
                raise ResolutionError(msg)
            return ctx.get_node(node_id=self.node_id, enforce_type="GraphNode")

        # Fallback to node label
        if self.node_label is not None:
            if not ctx.has_node(label=self.node_label):
                msg = (
                    f"No node exists with label='{self.node_label}' in the given "
                    "ExperimentContext."
                )
                raise ResolutionError(msg)
            return ctx.get_node(label=self.node_label, enforce_type="GraphNode")

        raise ResolutionError("Both node_label and node_id cannot be None.")

    # ================================================
    # Configurable
    # ================================================
    def get_config(self) -> dict[str, Any]:
        """Return a JSON-serializable configuration."""
        return {
            "exp_node_type": "GraphNode",
            "node_id": self.node_id,
            "node_label": self.node_label,
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> GraphNodeReference:
        """Reconstructs the reference from config."""
        ref_type = config.pop("exp_node_type", None)
        if ref_type is None:
            msg = "Config must contain `exp_node_type`."
            raise ValueError(msg)
        if ref_type != "GraphNode":
            msg = "Invalid config for a GraphNode."
            raise ValueError(msg)
        return cls(**config)
