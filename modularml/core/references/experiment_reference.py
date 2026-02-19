"""Experiment-level reference helpers for nodes and execution data."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any

from modularml.core.experiment.experiment_context import ExperimentContext
from modularml.core.io.protocols import Configurable
from modularml.core.references.reference_like import ReferenceLike

if TYPE_CHECKING:
    from modularml.core.experiment.experiment_node import ExperimentNode
    from modularml.core.topology.graph_node import GraphNode


class ResolutionError(RuntimeError):
    """Raised when a reference target cannot be resolved."""


@dataclass(frozen=True)
class ExperimentReference(ReferenceLike, Configurable):
    """
    Base class for references resolvable at the experiment scope.

    Attributes:
        None: This base dataclass does not declare fields.

    """

    def resolve(self, ctx: ExperimentContext | None = None):
        """
        Resolve the reference within an experiment context.

        Args:
            ctx (ExperimentContext | None): Context to resolve against. Defaults to active context.

        Returns:
            Any: Resolved object/value.

        """
        if ctx is None:
            ctx = ExperimentContext.get_active()
        return self._resolve_experiment(ctx)

    def _resolve_experiment(self, ctx: ExperimentContext):
        """
        Resolve the reference for a concrete :class:`ExperimentContext`.

        Args:
            ctx (ExperimentContext): Experiment context to resolve against.

        Returns:
            Any: Resolved object or value.

        Raises:
            NotImplementedError: Always for the abstract base class.

        """
        raise NotImplementedError

    def to_string(
        self,
        *,
        separator: str = ".",
        include_node_id: bool = False,
    ) -> str:
        """
        Join all non-null fields into a dotted path representation.

        Args:
            separator (str): Separator used when concatenating parts. Defaults to ".".
            include_node_id (bool): Whether to include the `node_id` field. Defaults to False.

        Returns:
            str: Dotted string representation of the reference.

        Example:
        ```python
            >>> ref = DataReference(node="PulseFeatures", domain="features", key="voltage")
            >>> ref.to_string()
            "PulseFeatures.features.voltage"
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
        """
        Return a JSON-serializable configuration.

        Returns:
            dict[str, Any]: Configuration suitable for :meth:`from_config`.

        Raises:
            NotImplementedError: Always for the abstract base class.

        """
        raise NotImplementedError

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> ExperimentReference:
        """
        Reconstruct the reference from configuration.

        Args:
            config (dict[str, Any]): Serialized reference settings.

        Returns:
            ExperimentReference: Reconstructed reference.

        Raises:
            NotImplementedError: Always for the abstract base class.

        """
        raise NotImplementedError


@dataclass(frozen=True)
class ExperimentNodeReference(ExperimentReference):
    """
    Reference to an :class:`ExperimentNode` by label or ID.

    Attributes:
        node_label (str | None): Preferred node label.
        node_id (str | None): Preferred node identifier.

    """

    node_label: str | None = None
    node_id: str | None = None

    def resolve(
        self,
        ctx: ExperimentContext | None = None,
    ) -> ExperimentNode:
        """
        Resolve this reference to an :class:`ExperimentNode`.

        Args:
            ctx (ExperimentContext | None): Context to resolve against.

        Returns:
            ExperimentNode: Resolved node instance.

        """
        return super().resolve(ctx=ctx)

    def _resolve_experiment(
        self,
        ctx: ExperimentContext,
    ) -> ExperimentNode:
        """
        Resolve the reference into an :class:`ExperimentNode`.

        Args:
            ctx (ExperimentContext): Experiment context containing the target node.

        Returns:
            ExperimentNode: Matching experiment node.

        Raises:
            TypeError: If `ctx` is not an :class:`ExperimentContext`.
            ResolutionError: If the node does not exist in the context.

        """
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
        """
        Return a JSON-serializable configuration.

        Returns:
            dict[str, Any]: Serialized configuration data.

        """
        return {
            "node_id": self.node_id,
            "node_label": self.node_label,
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> ExperimentReference:
        """
        Reconstruct the reference from configuration.

        Args:
            config (dict[str, Any]): Serialized reference settings.

        Returns:
            ExperimentReference: Rehydrated reference.

        """
        return cls(**config)


@dataclass(frozen=True)
class GraphNodeReference(ExperimentNodeReference):
    """
    Reference to a :class:`GraphNode` by label or ID.

    Attributes:
        node_label (str | None): Preferred node label.
        node_id (str | None): Preferred node identifier.

    """

    node_label: str | None = None
    node_id: str | None = None

    def resolve(
        self,
        ctx: ExperimentContext | None = None,
    ) -> GraphNode:
        """
        Resolve this reference to a :class:`GraphNode`.

        Args:
            ctx (ExperimentContext | None): Context to resolve against.

        Returns:
            GraphNode: Resolved node instance.

        """
        return super().resolve(ctx=ctx)

    def _resolve_experiment(
        self,
        ctx: ExperimentContext,
    ) -> GraphNode:
        """
        Resolve the reference into a :class:`GraphNode`.

        Args:
            ctx (ExperimentContext): Experiment context containing the target node.

        Returns:
            GraphNode: Matching graph node.

        Raises:
            TypeError: If `ctx` is not an :class:`ExperimentContext`.
            ResolutionError: If the node does not exist in the context.

        """
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
        """
        Return a JSON-serializable configuration.

        Returns:
            dict[str, Any]: Serialized configuration data.

        """
        return {
            "exp_node_type": "GraphNode",
            "node_id": self.node_id,
            "node_label": self.node_label,
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> GraphNodeReference:
        """
        Reconstruct the reference from configuration.

        Args:
            config (dict[str, Any]): Serialized reference data.

        Returns:
            GraphNodeReference: Rehydrated reference instance.

        Raises:
            ValueError: If the configuration is missing required metadata.

        """
        ref_type = config.pop("exp_node_type", None)
        if ref_type is None:
            msg = "Config must contain `exp_node_type`."
            raise ValueError(msg)
        if ref_type != "GraphNode":
            msg = "Invalid config for a GraphNode."
            raise ValueError(msg)
        return cls(**config)
