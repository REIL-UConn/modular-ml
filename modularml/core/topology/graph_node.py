from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Literal

from modularml.context.experiment_context import ExperimentContext
from modularml.core.data.schema_constants import STREAM_DEFAULT
from modularml.core.experiment.experiment_node import ExperimentNode
from modularml.core.experiment.phase import InputBinding
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
    from pathlib import Path

    from modularml.core.data.featureset import FeatureSet
    from modularml.core.data.featureset_view import FeatureSetView
    from modularml.core.sampling.base_sampler import BaseSampler


class GraphNode(ABC, ExperimentNode):
    """
    Abstract base class for all nodes within a ModelGraph.

    Each node is identified by a unique `label` and may have one or more upstream (input)
    and downstream (output) connections. GraphNode defines the base interface for managing
    these connections and enforcing structural constraints like maximum allowed connections.

    Subclasses must define the `input_shape` and `output_shape` properties, as well as
    whether the node supports incoming and outgoing edges.
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
        Initialize a GraphNode with optional upstream and downstream connections.

        Args:
            label (str):
                Unique identifier for this node.
            upstream_refs (GraphNodeReference | list[GraphNodeReference] | None):
                References of upstream connections.
            downstream_refs (GraphNodeReference | list[GraphNodeReference] | None):
                References of downstream connections.
            node_id (str, optional):
                Used only for de-serialization.
            register (bool, optional):
                Used only for de-serialization.

        Raises:
            TypeError: If `upstream_refs` or `downstream_refs` are not valid types.

        """
        super().__init__(label=label, node_id=node_id, register=register)

        # Normalize inputs as lists
        self._upstream_refs: list[GraphNodeReference] = ensure_list(upstream_refs)
        self._downstream_refs: list[GraphNodeReference] = ensure_list(downstream_refs)

        # Validate connections
        self._validate_connections()

    def _validate_connections(self):
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
        Maximum number of upstream (input) nodes allowed.

        Returns:
            int | None: If None, unlimited upstream nodes are allowed.

        """
        return None

    @property
    def max_downstream_refs(self) -> int | None:
        """
        Maximum number of downstream (output) nodes allowed.

        Returns:
            int | None: If None, unlimited downstream nodes are allowed.

        """
        return None

    @property
    def upstream_ref(self) -> GraphNodeReference | ExperimentNodeReference | None:
        """
        Return the single upstream reference, if only one is allowed.

        Raises:
            RuntimeError: If multiple upstream references are allowed.

        """
        if self.max_upstream_refs == 1:
            return self.get_upstream_refs()[0] if self._upstream_refs else None
        raise RuntimeError(
            "This node allows multiple upstream_refs. Use `get_upstream_refs()`",
        )

    @property
    def downstream_ref(self) -> GraphNodeReference | None:
        """
        Return the single downstream reference, if only one is allowed.

        Raises:
            RuntimeError: If multiple downstream references are allowed.

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
        return [
            ("label", self.label),
            ("upstream_refs", [f"{r!r}" for r in self._upstream_refs]),
            ("downstream_refs", [f"{r!r}" for r in self._downstream_refs]),
            ("node_id", self.node_id),
        ]

    def __repr__(self):
        return f"GraphNode(label='{self.label}', upstream_refs={self._upstream_refs}, downstream_refs={self._downstream_refs})"

    def __str__(self):
        return f"GraphNode('{self.label}')"

    # ================================================
    # Referencing
    # ================================================
    def reference(self) -> GraphNodeReference:
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
        Retrieve all upstream (input) references.

        Args:
            error_mode (ErrorMode): Error handling strategy if input is invalid.

        Returns:
            list[GraphNodeReference | ExperimentNodeReference]: List of upstream connection references.

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
        Retrieve all downstream (output) references.

        Args:
            error_mode (ErrorMode): Error handling strategy if input is invalid.

        Returns:
            list[GraphNodeReference]: List of downstream connection references.

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
        Add a new upstream connection.

        Args:
            ref (GraphNodeReference | ExperimentNodeReference): Reference of upstream connection.
            error_mode (ErrorMode): Error handling mode for duplicates or limits.

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
        Remove an upstream reference.

        Args:
            ref (GraphNodeReference | ExperimentNodeReference): Upstream reference to remove.
            error_mode (ErrorMode): Error handling mode if ref not found.

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
        Remove all upstream connections.

        Args:
            error_mode (ErrorMode): Error handling mode if disallowed.

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
        Replace all upstream connections with a new list of references.

        Args:
            upstream_refs (list[GraphNodeReference | ExperimentNodeReference]): List of new upstream references.
            error_mode (ErrorMode): Error handling mode for violations.

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
        Add a new downstream connection.

        Args:
            ref (GraphNodeReference): Reference of downstream connection.
            error_mode (ErrorMode): Error handling mode for duplicates or limits.

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
        Remove a downstream reference.

        Args:
            ref (GraphNodeReference): Downstream reference to remove.
            error_mode (ErrorMode): Error handling mode if ref not found.

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
        Remove all downstream connections.

        Args:
            error_mode (ErrorMode): Error handling mode if disallowed.

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
        Replace all downstream connections with a new list of references.

        Args:
            downstream_refs (list[GraphNodeReference]): List of new downstream references.
            error_mode (ErrorMode): Error handling mode for violations.

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
        """
        Whether this node allows incoming (upstream) connections.

        Returns:
            bool: True if input connections are allowed.

        """

    @property
    @abstractmethod
    def allows_downstream_connections(self) -> bool:
        """
        Whether this node allows outgoing (downstream) connections.

        Returns:
            bool: True if output connections are allowed.

        """

    # ================================================
    # Internal Helpers
    # ================================================
    def _handle_fatal_error(self, exc_class, message: str, error_mode: ErrorMode):
        """Raise or suppress a fatal error based on the provided error mode."""
        if error_mode == ErrorMode.IGNORE:
            return False
        if error_mode in (ErrorMode.WARN, ErrorMode.COERCE, ErrorMode.RAISE):
            raise exc_class(message)
        msg = f"Unsupported ErrorMode: {error_mode}"
        raise NotImplementedError(msg)

    def _handle_benign_error(self, exc_class, message: str, error_mode: ErrorMode):
        """Raise, warn, or ignore a non-fatal error based on the error mode."""
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
        Create an InputBinding for an input connection to this node.

        Args:
            sampler (BaseSampler):
                A sampler to use to generate batches from the upstream FeatureSet
                (e.g., random batches, contrastive roles, paired samples).
                Required if this binding is for a TrainPhase.

            upstream (FeatureSet | FeatureSetView | str | None):
                Identifies which upstream FeatureSet connection of this node this
                binding applies to.
                Accepted values:
                - FeatureSet instance
                - FeatureSetView instance
                - FeatureSet node ID or label (str)
                - None, only if this node has exactly one upstream FeatureSet

                If this node has multiple upstream FeatureSets, this argument is
                required to disambiguate which input is being bound.

            split (str, optional):
                Optional split name of the upstream FeatureSet (e.g. "train", "val").
                If provided, only rows from this split are sampled.
                If None, the entire FeatureSet is used.

            stream (str, optional):
                Output stream name from the sampler to feed into this node.
                Required only if the sampler produces multiple streams.
                Defaults to STREAM_DEFAULT.

        Returns:
            InputBinding:
                A fully specified training InputBinding that can be passed directly
                to an ExperimentPhase.

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
        return cls(register=register, **config)

    # ================================================
    # Serialization
    # ================================================
    def save(self, filepath: Path, *, overwrite: bool = False) -> Path:
        """
        Serializes this GraphNode to the specified filepath.

        Args:
            filepath (Path):
                File location to save to. Note that the suffix may be overwritten
                to enforce the ModularML file extension schema.
            overwrite (bool, optional):
                Whether to overwrite any existing file at the save location.
                Defaults to False.

        Returns:
            Path: The actual filepath to write the FeatureSet is saved.

        """
        raise NotImplementedError

    @classmethod
    def load(
        cls,
        filepath: Path,
        *,
        allow_packaged_code: bool = False,
        overwrite: bool = True,
    ) -> GraphNode:
        """
        Load a GraphNode from file.

        Args:
            filepath (Path):
                File location of a previously saved GraphNode.
            allow_packaged_code : bool
                Whether bundled code execution is allowed.
            overwrite (bool):
                Whether to replace any colliding node registrations in ExperimentContext
                If False, a new node_id is assigned to the reloaded GraphNode. Otherwise,
                the existing GraphNode is removed from the ExperimentContext registry.
                Defaults to True.

        Returns:
            GraphNode: The reloaded GraphNode.

        """
        raise NotImplementedError
