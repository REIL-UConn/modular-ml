"""Base graph node abstractions for ModularML model graphs."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Literal

from modularml.core.data.schema_constants import STREAM_DEFAULT
from modularml.core.experiment.experiment_context import ExperimentContext
from modularml.core.experiment.experiment_node import ExperimentNode
from modularml.core.experiment.phases.phase import InputBinding
from modularml.core.references.experiment_reference import (
    ExperimentNodeReference,
    GraphNodeReference,
    ResolutionError,
)
from modularml.utils.data.formatting import ensure_list
from modularml.utils.errors.error_handling import ErrorMode
from modularml.utils.errors.exceptions import GraphNodeInputError, GraphNodeOutputError
from modularml.utils.logging.warnings import warn

if TYPE_CHECKING:
    from modularml.core.data.featureset import FeatureSet
    from modularml.core.data.featureset_view import FeatureSetView
    from modularml.core.sampling.base_sampler import BaseSampler


class GraphNode(ABC, ExperimentNode):
    """
    Abstract base class for nodes inside a :class:`ModelGraph`.

    Description:
        Each node exposes upstream (input) and downstream (output)
        references resolved through :class:`GraphNodeReference`. Subclasses
        define shape contracts, supported edge counts, and execution
        semantics.

    Attributes:
        _upstream_refs (list[GraphNodeReference]):
            References to nodes that feed this node.
        _downstream_refs (list[GraphNodeReference]):
            References to nodes fed by this node.

    """

    def __init__(
        self,
        label: str,
        upstream_refs: GraphNodeReference | list[GraphNodeReference] | None = None,
        downstream_refs: GraphNodeReference | list[GraphNodeReference] | None = None,
        *,
        node_id: str | None = None,
        register: bool = True,
    ):
        """
        Initialize a graph node and register upstream/downstream refs.

        Args:
            label (str):
                Unique identifier for this node.
            upstream_refs (GraphNodeReference | list[GraphNodeReference] | None):
                Reference or list of references feeding this node.
            downstream_refs (GraphNodeReference | list[GraphNodeReference] | None):
                Reference or list of references fed by this node.
            node_id (str | None):
                Identifier used during deserialization.
            register (bool):
                Whether to register the node with the active :class:`ExperimentContext`.
                Set to False for deserialization.

        Raises:
            TypeError: If `upstream_refs` or `downstream_refs` are not
                :class:`GraphNodeReference` instances.

        """
        super().__init__(label=label, node_id=node_id, register=register)

        # Normalize inputs as lists
        self._upstream_refs: list[GraphNodeReference] = ensure_list(upstream_refs)
        self._downstream_refs: list[GraphNodeReference] = ensure_list(downstream_refs)

        # Validate connections
        self._validate_connections()

    def _validate_connections(self):
        """
        Enforce connection limits and ensure references resolve.

        Raises:
            ValueError: If upstream or downstream references cannot be
                resolved in the active :class:`ExperimentContext`.

        """
        # Enforce max_upstream_refs
        if (
            self.max_upstream_refs is not None
            and len(self._upstream_refs) > self.max_upstream_refs
        ):
            msg = (
                f"{len(self._upstream_refs)} upstream_refs provided, but "
                f"max_upstream_refs = {self.max_upstream_refs}."
            )
            if (
                self._handle_fatal_error(GraphNodeInputError, msg, ErrorMode.RAISE)
                is False
            ):
                self._upstream_refs = self._upstream_refs[: self.max_upstream_refs]

        # Enforce max_downstream_refs
        if (
            self.max_downstream_refs is not None
            and len(self._downstream_refs) > self.max_downstream_refs
        ):
            msg = (
                f"{len(self._downstream_refs)} downstream_refs provided, but "
                f"max_downstream_refs = {self.max_downstream_refs}."
            )
            if (
                self._handle_fatal_error(GraphNodeOutputError, msg, ErrorMode.RAISE)
                is False
            ):
                self._downstream_refs = self._downstream_refs[
                    : self.max_downstream_refs
                ]

        # Ensure referenced connections exist in this ExperimentContext
        def _val_ref_existence(
            refs: list[GraphNodeReference],
            direction: Literal["upstream", "downstream"],
        ):
            """
            Ensure references resolve inside the active experiment context.

            Args:
                refs (list[GraphNodeReference]):
                    References to validate.
                direction (Literal['upstream', 'downstream']):
                    Direction label used to format error messages.

            Raises:
                ValueError: If any reference fails to resolve.

            """
            exp_ctx = ExperimentContext.get_active()
            failed: list[GraphNodeReference] = []
            for r in refs:
                try:
                    _ = r.resolve(ctx=exp_ctx)
                except ResolutionError:  # noqa: PERF203
                    failed.append(r)
            if failed:
                details = "\n".join(
                    f"  - {ref.__class__.__name__}: {ref!r}" for ref in failed
                )
                msg = (
                    f"The following {direction} reference(s) could not be resolved "
                    f"in the current ExperimentContext:\n{details}"
                )
                raise ValueError(msg)

        _val_ref_existence(self._upstream_refs, "upstream")
        _val_ref_existence(self._downstream_refs, "downstream")

    @property
    def max_upstream_refs(self) -> int | None:
        """
        Return the maximum number of upstream nodes allowed.

        Returns:
            int | None: Limit enforced for upstream references.

        """
        return None

    @property
    def max_downstream_refs(self) -> int | None:
        """
        Return the maximum number of downstream nodes allowed.

        Returns:
            int | None: Limit enforced for downstream references.

        """
        return None

    @property
    def upstream_ref(self) -> GraphNodeReference | ExperimentNodeReference | None:
        """
        Return the upstream reference when only one connection exists.

        Returns:
            GraphNodeReference | ExperimentNodeReference | None:
                Single upstream reference, if defined.

        Raises:
            RuntimeError: If multiple upstream connections are allowed
                and :meth:`get_upstream_refs` should be used instead.

        """
        if self.max_upstream_refs == 1:
            return self.get_upstream_refs()[0] if self._upstream_refs else None
        raise RuntimeError(
            "This node allows multiple upstream_refs. Use `get_upstream_refs()`",
        )

    @property
    def downstream_ref(self) -> GraphNodeReference | None:
        """
        Return the downstream reference when only one connection exists.

        Returns:
            GraphNodeReference | None: Single downstream reference.

        Raises:
            RuntimeError: If multiple downstream connections are allowed
                and :meth:`get_downstream_refs` should be used instead.

        """
        if self.max_downstream_refs == 1:
            return self.get_downstream_refs()[0] if self._downstream_refs else None
        raise RuntimeError(
            "This node allows multiple downstream_refs. Use `get_downstream_refs()`",
        )

    # ================================================
    # Representation
    # ================================================
    def _summary_rows(self) -> list[tuple]:
        """
        Return key/value pairs used in textual summaries.

        Returns:
            list[tuple]: Metadata pairs showing labels and references.

        """
        return [
            ("label", self.label),
            ("upstream_refs", [f"{r!r}" for r in self._upstream_refs]),
            ("downstream_refs", [f"{r!r}" for r in self._downstream_refs]),
            ("node_id", self.node_id),
        ]

    def __repr__(self):
        """
        Return developer-friendly representation for debugging.

        Returns:
            str: String capturing label and referenced nodes.

        """
        return (
            "GraphNode("
            f"label='{self.label}', "
            f"upstream_refs={self._upstream_refs}, "
            f"downstream_refs={self._downstream_refs}"
            ")"
        )

    def __str__(self):
        """
        Return stringified label for logging.

        Returns:
            str: Node label formatted for readability.

        """
        return f"GraphNode('{self.label}')"

    # ================================================
    # Referencing
    # ================================================
    def reference(self) -> GraphNodeReference:
        """
        Return a :class:`GraphNodeReference` targeting this node.

        Returns:
            GraphNodeReference: Reference pointing at this node.

        """
        return GraphNodeReference(
            node_id=self.node_id,
            node_label=self.label,
        )

    # ================================================
    # Connection Management
    # ================================================
    def get_upstream_refs(
        self,
        error_mode: ErrorMode = ErrorMode.RAISE,
    ) -> list[GraphNodeReference | ExperimentNodeReference]:
        """
        Return all upstream (input) references.

        Args:
            error_mode (ErrorMode):
                Error handling policy when upstream connections are disallowed
                or invalid.

        Returns:
            list[GraphNodeReference | ExperimentNodeReference]:
                Upstream connection references.

        Raises:
            GraphNodeInputError: When upstream connections are disallowed
                and `error_mode` requests an exception.

        """
        if not self.allows_upstream_connections:
            handled = self._handle_benign_error(
                GraphNodeInputError,
                "This node does not allow upstream connections.",
                error_mode,
            )
            return [] if handled is False else self._upstream_refs
        return self._upstream_refs

    def get_downstream_refs(
        self,
        error_mode: ErrorMode = ErrorMode.RAISE,
    ) -> list[GraphNodeReference]:
        """
        Return all downstream (output) references.

        Args:
            error_mode (ErrorMode):
                Error handling policy when downstream connections are disallowed
                or invalid.

        Returns:
            list[GraphNodeReference]: Downstream connection references.

        Raises:
            GraphNodeOutputError: When downstream connections are
                disallowed and `error_mode` requests an exception.

        """
        if not self.allows_downstream_connections:
            handled = self._handle_benign_error(
                GraphNodeOutputError,
                "This node does not allow downstream connections.",
                error_mode,
            )
            return [] if handled is False else self._downstream_refs
        return self._downstream_refs

    def add_upstream_ref(
        self,
        ref: GraphNodeReference | ExperimentNodeReference,
        error_mode: ErrorMode = ErrorMode.RAISE,
    ):
        """
        Add an upstream connection reference.

        Args:
            ref (GraphNodeReference | ExperimentNodeReference):
                Reference pointing to an upstream node.
            error_mode (ErrorMode):
                Error handling policy for duplicates or limit violations.

        """
        if (
            not self.allows_upstream_connections
            and self._handle_fatal_error(
                GraphNodeInputError,
                "Upstream connections not allowed.",
                error_mode,
            )
            is False
        ):
            return

        if (
            ref in self._upstream_refs
            and self._handle_benign_error(
                GraphNodeInputError,
                f"Upstream reference '{ref!r}' already exists.",
                error_mode,
            )
            is False
        ):
            return

        if (
            self.max_upstream_refs is not None
            and len(self._upstream_refs) >= self.max_upstream_refs
            and self._handle_fatal_error(
                GraphNodeInputError,
                f"Only {self.max_upstream_refs} upstream_refs allowed. Received: {self._upstream_refs}",
                error_mode,
            )
            is False
        ):
            return

        self._upstream_refs.append(ref)

    def remove_upstream_ref(
        self,
        ref: GraphNodeReference | ExperimentNodeReference,
        error_mode: ErrorMode = ErrorMode.RAISE,
    ):
        """
        Remove an upstream connection reference.

        Args:
            ref (GraphNodeReference | ExperimentNodeReference):
                Reference to remove.
            error_mode (ErrorMode):
                Error handling policy if the reference does not exist.

        """
        if (
            not self.allows_upstream_connections
            and self._handle_benign_error(
                GraphNodeInputError,
                "Upstream connections not allowed.",
                error_mode,
            )
            is False
        ):
            return

        if (
            ref not in self._upstream_refs
            and self._handle_benign_error(
                GraphNodeInputError,
                f"Upstream reference '{ref!r}' does not exist.",
                error_mode,
            )
            is False
        ):
            return

        self._upstream_refs.remove(ref)

    def clear_upstream_refs(self, error_mode: ErrorMode = ErrorMode.RAISE):
        """
        Remove all upstream references after validating permissions.

        Args:
            error_mode (ErrorMode): Error handling policy when upstream
                connections are disallowed.

        """
        if (
            not self.allows_upstream_connections
            and self._handle_benign_error(
                GraphNodeInputError,
                "Upstream connections not allowed.",
                error_mode,
            )
            is False
        ):
            return
        self._upstream_refs = []

    def set_upstream_refs(
        self,
        upstream_refs: list[GraphNodeReference | ExperimentNodeReference],
        error_mode: ErrorMode = ErrorMode.RAISE,
    ):
        """
        Replace upstream connections with a supplied list.

        Args:
            upstream_refs (list[GraphNodeReference | ExperimentNodeReference]):
                References applied in order.
            error_mode (ErrorMode):
                Error handling policy for violations.

        """
        self.clear_upstream_refs(error_mode=error_mode)
        for ref in upstream_refs:
            self.add_upstream_ref(ref, error_mode=error_mode)

    def add_downstream_ref(
        self,
        ref: GraphNodeReference,
        error_mode: ErrorMode = ErrorMode.RAISE,
    ):
        """
        Add a downstream connection reference.

        Args:
            ref (GraphNodeReference):
                Reference pointing to the downstream node.
            error_mode (ErrorMode):
                Error handling policy for duplicates or maximum limits.

        """
        if (
            not self.allows_downstream_connections
            and self._handle_fatal_error(
                GraphNodeOutputError,
                "Downstream connections not allowed.",
                error_mode,
            )
            is False
        ):
            return

        if (
            ref in self._downstream_refs
            and self._handle_benign_error(
                GraphNodeOutputError,
                f"Downstream reference '{ref!r}' already exists.",
                error_mode,
            )
            is False
        ):
            return

        if (
            self.max_downstream_refs is not None
            and len(self._downstream_refs) >= self.max_downstream_refs
            and self._handle_fatal_error(
                GraphNodeOutputError,
                f"Only {self.max_downstream_refs} downstream_refs allowed.",
                error_mode,
            )
            is False
        ):
            return

        self._downstream_refs.append(ref)

    def remove_downstream_ref(
        self,
        ref: GraphNodeReference,
        error_mode: ErrorMode = ErrorMode.RAISE,
    ):
        """
        Remove a downstream connection reference.

        Args:
            ref (GraphNodeReference):
                Reference to remove.
            error_mode (ErrorMode):
                Error handling policy if the reference does not exist.

        """
        if (
            not self.allows_downstream_connections
            and self._handle_benign_error(
                GraphNodeOutputError,
                "Downstream connections not allowed.",
                error_mode,
            )
            is False
        ):
            return

        if (
            ref not in self._downstream_refs
            and self._handle_benign_error(
                GraphNodeOutputError,
                f"Downstream reference '{ref!r}' does not exist.",
                error_mode,
            )
            is False
        ):
            return

        self._downstream_refs.remove(ref)

    def clear_downstream_refs(self, error_mode: ErrorMode = ErrorMode.RAISE):
        """
        Remove all downstream references after validating permissions.

        Args:
            error_mode (ErrorMode):
                Error handling policy when downstream connections are disallowed.

        """
        if (
            not self.allows_downstream_connections
            and self._handle_benign_error(
                GraphNodeOutputError,
                "Downstream connections not allowed.",
                error_mode,
            )
            is False
        ):
            return
        self._downstream_refs = []

    def set_downstream_refs(
        self,
        downstream_refs: list[str],
        error_mode: ErrorMode = ErrorMode.RAISE,
    ):
        """
        Replace downstream connections with supplied references.

        Args:
            downstream_refs (list[str]): New downstream reference values.
            error_mode (ErrorMode): Error handling policy for violations.

        """
        self.clear_downstream_refs(error_mode=error_mode)
        for ref in downstream_refs:
            self.add_downstream_ref(ref, error_mode=error_mode)

    # ================================================
    # Abstract Methods / Properties
    # ================================================
    @property
    @abstractmethod
    def allows_upstream_connections(self) -> bool:
        """Return True if this node supports upstream connections."""

    @property
    @abstractmethod
    def allows_downstream_connections(self) -> bool:
        """Return True if this node supports downstream connections."""

    # ================================================
    # Internal Helpers
    # ================================================
    def _handle_fatal_error(self, exc_class, message: str, error_mode: ErrorMode):
        """
        Raise or suppress fatal errors according to :class:`ErrorMode`.

        Args:
            exc_class (type[Exception]): Exception class to raise.
            message (str): Error description.
            error_mode (ErrorMode): Error handling policy.

        Returns:
            bool | None: False when the error should be ignored; otherwise
                this method raises the exception.

        """
        if error_mode == ErrorMode.IGNORE:
            return False
        if error_mode in (ErrorMode.WARN, ErrorMode.COERCE, ErrorMode.RAISE):
            raise exc_class(message)
        msg = f"Unsupported ErrorMode: {error_mode}"
        raise NotImplementedError(msg)

    def _handle_benign_error(self, exc_class, message: str, error_mode: ErrorMode):
        """
        Handle recoverable errors according to :class:`ErrorMode`.

        Args:
            exc_class (type[Exception]):
                Exception class to raise when escalation is required.
            message (str):
                Error description.
            error_mode (ErrorMode):
                Error handling policy.

        Returns:
            bool | None: False when the error was ignored or coerced;
                otherwise the exception is raised.

        """
        if error_mode == ErrorMode.RAISE:
            raise exc_class(message)
        if error_mode == ErrorMode.WARN:
            warn(message, UserWarning, stacklevel=2)
            return False
        if error_mode in (ErrorMode.COERCE, ErrorMode.IGNORE):
            return False
        msg = f"Unsupported ErrorMode: {error_mode}"
        raise NotImplementedError(msg)

    # ================================================
    # Input Binding
    # ================================================
    def create_input_binding(
        self,
        *,
        sampler: BaseSampler | None = None,
        upstream: FeatureSet | FeatureSetView | str | None = None,
        split: str | None = None,
        stream: str = STREAM_DEFAULT,
    ) -> InputBinding:
        """
        Create an :class:`InputBinding` targeting this node.

        Args:
            sampler (BaseSampler | None):
                Sampler used for training bindings. Required for training phases
                and omitted for evaluation bindings.
            upstream (FeatureSet | FeatureSetView | str | None):
                Upstream FeatureSet identifier. Required when multiple upstream
                FeatureSets exist; a label, :class:`FeatureSet`, or
                :class:`FeatureSetView` may be supplied.
            split (str | None):
                Optional split name (for example, `train`). When omitted, the full
                FeatureSet is used.
            stream (str):
                Sampler stream name to consume when the sampler outputs multiple
                streams.

        Returns:
            InputBinding: Binding configured for training or evaluation.

        """
        if sampler is None:
            return InputBinding.for_evaluation(
                node=self,
                upstream=upstream,
                split=split,
            )
        return InputBinding.for_training(
            node=self,
            sampler=sampler,
            upstream=upstream,
            split=split,
            stream=stream,
        )

    # ================================================
    # Configurable
    # ================================================
    def get_config(self) -> dict[str, Any]:
        """
        Return configuration for reconstructing the graph node.

        Returns:
            dict[str, Any]: Serialized representation of the node.

        """
        cfg = super().get_config()
        cfg.update(
            {
                "upstream_refs": self._upstream_refs,
                "downstream_refs": self._downstream_refs,
            },
        )
        return cfg

    @classmethod
    def from_config(cls, config: dict[str, Any], *, register: bool = True) -> GraphNode:
        """
        Instantiate a graph node or subclass from serialized config.

        Args:
            config (dict[str, Any]):
                Serialized configuration previously produced by :meth:`get_config`.
            register (bool):
                Whether to register the node with the active :class:`ExperimentContext`.

        Returns:
            GraphNode: Reconstructed node (or subclass instance).

        Raises:
            ValueError: If an unknown graph node subclass is requested.

        """
        # Create GraphNode instance if subclass not specified
        if "graph_node_type" not in config:
            return cls(register=register, **config)

        # Create subclasses directly
        g_type = config["graph_node_type"]
        if g_type == "ModelNode":
            from modularml.core.topology.model_node import ModelNode

            return ModelNode.from_config(config=config, register=register)

        if g_type == "ComputeNode":
            from modularml.core.topology.compute_node import ComputeNode

            return ComputeNode.from_config(config=config, register=register)

        msg = f"Unknown GraphNode subclass: {g_type}."
        raise ValueError(msg)

    # ================================================
    # Stateful
    # ================================================
    def get_state(self) -> dict[str, Any]:
        """
        Return state snapshot capturing super state and references.

        Returns:
            dict[str, Any]: State data used by :meth:`set_state`.

        """
        state = {
            "super": super().get_state(),
            "upstream_refs": self._upstream_refs,
            "downstream_refs": self._downstream_refs,
        }
        return state

    def set_state(self, state: dict[str, Any]):
        """
        Restore node state from :meth:`get_state` output.

        Args:
            state (dict[str, Any]):
                Serialized state previously produced by :meth:`get_state`.

        """
        super().set_state(state["super"])
        self._upstream_refs = state["upstream_refs"]
        self._downstream_refs = state["downstream_refs"]
