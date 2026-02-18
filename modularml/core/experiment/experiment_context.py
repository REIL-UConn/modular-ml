"""Experiment context registry and lifecycle helpers."""

from __future__ import annotations

import os
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any
from weakref import ref

from matplotlib.pylab import Enum

from modularml.utils.environment.environment import IN_NOTEBOOK
from modularml.utils.logging.warnings import warn

if TYPE_CHECKING:
    from modularml.core.data.featureset import FeatureSet
    from modularml.core.experiment.experiment import Experiment
    from modularml.core.experiment.experiment_node import ExperimentNode
    from modularml.core.topology.compute_node import ComputeNode
    from modularml.core.topology.model_graph import ModelGraph


_ACTIVE_EXPERIMENT_CONTEXT: ContextVar[ExperimentContext | None] = ContextVar(
    "_ACTIVE_EXPERIMENT_CONTEXT",
    default=None,
)


class RegistrationPolicy(Enum):
    """Controls behavior when registering objects with duplicate labels."""

    ERROR = "error"
    OVERWRITE = "overwrite"
    OVERWRITE_WARN = "overwrite_warn"
    NO_REGISTER = "no_register"

    @classmethod
    def from_value(cls, value):
        """
        Cast a string or enum value to :class:`RegistrationPolicy`.

        Args:
            value (str | RegistrationPolicy): Source value.

        Returns:
            RegistrationPolicy: Normalized policy value.

        Raises:
            ValueError: If the provided value cannot be mapped to a policy.

        """
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            v = value.strip().lower()
            for policy in cls:
                if policy.value == v:
                    return policy
        msg = f"Invalid registration policy: {value!r}. Expected one of: {[p.value for p in cls]}"
        raise ValueError(msg)


class ExperimentContext:
    """Registry and lifecycle controller for a single Experiment."""

    def __init__(
        self,
        *,
        experiment: Experiment | None = None,
        registration_policy: RegistrationPolicy | str | None = None,
    ):
        self._experiment_ref = ref(experiment) if experiment else None

        # Registries
        self._nodes_by_id: dict[str, ExperimentNode] = {}
        self._node_label_to_id: dict[str, str] = {}
        self._mg: ModelGraph | None = None

        # Registration policy
        if registration_policy is None:
            self._policy = self._resolve_default_policy()
        else:
            self._policy = RegistrationPolicy.from_value(registration_policy)

    # ================================================
    # Active Context Helpers
    # ================================================
    @classmethod
    def _set_active(cls, ctx: ExperimentContext):
        _ACTIVE_EXPERIMENT_CONTEXT.set(ctx)

    @classmethod
    def get_active(cls) -> ExperimentContext:
        """
        Return the active :class:`ExperimentContext`.

        Args:
            cls (type[ExperimentContext]): Ignored class reference.

        Returns:
            ExperimentContext: Currently active context.

        Raises:
            RuntimeError: If no active context is set.

        """
        ctx = _ACTIVE_EXPERIMENT_CONTEXT.get()
        if ctx is None:
            raise RuntimeError("There is no active ExperimentContext.")
        return ctx

    @contextmanager
    def activate(self):
        """
        Activates a new ExperimentContext within the context scope.

        Example Usage:
        ```python
        with ExperimentContext(experiment=my_exp).activate():
            ref.resolve()  # resolves using a context of `my_exp`
        ```

        Yields:
            ExperimentContext

        """
        token = _ACTIVE_EXPERIMENT_CONTEXT.set(self)
        try:
            yield self
        finally:
            _ACTIVE_EXPERIMENT_CONTEXT.reset(token)

    @contextmanager
    def temporary(self):
        """
        Create a fully isolated temporary execution scope.

        Description:
            All modifications to:
                - registered nodes
                - model graph
                - registration policy
                - experiment binding

            will be reverted when the context exits.

            This is primarily used for cross-validation and
            other meta-execution procedures.

        Example:
        ```python
            ctx = ExperimentContext.get_active()
            with ctx.temporary():
                ctx.set_registration_policy("overwrite")
                ctx.register_experiment_node(new_fs)
                run_fold()
            # context fully restored
        ```

        Yields:
            ExperimentContext

        """
        # Record context state
        old_state = self.get_state()
        token = _ACTIVE_EXPERIMENT_CONTEXT.set(self)
        try:
            yield self
        finally:
            # Reset state and active context
            self.set_state(old_state)
            _ACTIVE_EXPERIMENT_CONTEXT.reset(token)

    # ================================================
    # Policy Management
    # ================================================
    def _resolve_default_policy(self) -> RegistrationPolicy:
        """
        Determine the default registration policy based on environment.

        Priority (highest to lowest):
            1. Environment variable
            2. Jupyter notebook detection
            3. Script default
        """
        # 1. Explicit env override
        env = os.getenv("MODULARML_EXP_REGISTRATION_POLICY")
        if env:
            return RegistrationPolicy.from_value(env)

        # 2. If running in Jupyter Notebook -> default to OVERWRITE_WARN
        if IN_NOTEBOOK:
            return RegistrationPolicy.OVERWRITE_WARN

        # 3. Else --> default to ERROR
        return RegistrationPolicy.ERROR

    def set_registration_policy(self, policy: str | RegistrationPolicy):
        """
        Permanently set the registration policy.

        Args:
            policy (str | RegistrationPolicy): Policy name or enum.

        """
        self._policy = RegistrationPolicy.from_value(policy)

    @contextmanager
    def use_policy(self, policy: str | RegistrationPolicy):
        """
        Temporarily override the registration policy inside a context.

        Args:
            policy (str | RegistrationPolicy): Policy to use within the scope.

        Yields:
            None: Control returns to the caller once the context exits.

        """
        old = self._policy
        self._policy = RegistrationPolicy.from_value(policy)
        try:
            yield
        finally:
            self._policy = old

    @contextmanager
    def dont_register(self):
        """
        Temporarily disable ExperimentNode registration.

        Any nodes created inside this context will not be registered
        to the active ExperimentContext.

        Example Usage:
        ```python
        with ExperimentContext.dont_register():
            internal_copy = ModelStage.from_state(...)
        ```

        """
        old = self._policy
        self._policy = RegistrationPolicy.NO_REGISTER
        try:
            yield
        finally:
            self._policy = old

    # ================================================
    # Experiment Lifecycle
    # ================================================
    def set_experiment(
        self,
        experiment: Experiment,
        *,
        reset_registries: bool = False,
    ):
        """
        Set the experiment reference for this context.

        Args:
            experiment (Experiment): Experiment to associate.
            reset_registries (bool, optional):
                Whether to clear node registries prior to association.

        """
        self._experiment_ref = ref(experiment)
        if reset_registries:
            self.clear_registries()

    def get_experiment(self) -> Experiment | None:
        """Return the Experiment active in this context, if defined."""
        return None if self._experiment_ref is None else self._experiment_ref()

    # ================================================
    # Registry Helpers
    # ================================================
    def clear_registries(self):
        """Clear all registered items."""
        self._nodes_by_id.clear()
        self._node_label_to_id.clear()

    def register_experiment_node(
        self,
        node: ExperimentNode,
        *,
        check_label_collision: bool = True,
    ):
        """
        Register a node with optional collision handling.

        Args:
            node (ExperimentNode):
                Node to register in this context.
            check_label_collision (bool, optional):
                Whether to enforce uniqueness for labels. Defaults to True.

        Raises:
            TypeError: If `node` is not an :class:`ExperimentNode`.
            ValueError: If duplicates are encountered under
                :attr:`RegistrationPolicy.ERROR`.

        """
        from modularml.core.experiment.experiment_node import ExperimentNode

        # Validate node
        if not isinstance(node, ExperimentNode):
            msg = f"`node` must be an ExperimentNode. Received: {type(node)}"
            raise TypeError(msg)
        node_id = node.node_id
        label = node.label

        # ID collision checks
        if node_id in self._nodes_by_id:
            if self._policy is RegistrationPolicy.ERROR:
                msg = f"ExperimentNode with ID '{node_id}' is already registered."
                raise ValueError(msg)
            if self._policy in (
                RegistrationPolicy.OVERWRITE,
                RegistrationPolicy.OVERWRITE_WARN,
            ):
                old = self._nodes_by_id[node_id]
                self._node_label_to_id.pop(old.label, None)
                if self._policy == RegistrationPolicy.OVERWRITE_WARN:
                    msg = f"Overwriting existing node with ID '{old.node_id}'."
                    warn(msg, category=UserWarning, stacklevel=2)
            else:
                return

        # Label collision checks
        if check_label_collision and (label in self._node_label_to_id):
            if self._policy is RegistrationPolicy.ERROR:
                msg = f"ExperimentNode label '{label}' already exists."
                raise ValueError(msg)
            if self._policy in (
                RegistrationPolicy.OVERWRITE,
                RegistrationPolicy.OVERWRITE_WARN,
            ):
                old_id = self._node_label_to_id[label]
                old = self._nodes_by_id.pop(old_id, None)
                if self._policy == RegistrationPolicy.OVERWRITE_WARN:
                    msg = f"Overwriting existing node with label '{old.label}'."
                    warn(msg, category=UserWarning, stacklevel=2)

        # Register unique node UUID and string-based label
        self._nodes_by_id[node.node_id] = node
        self._node_label_to_id[node.label] = node.node_id

    def remove_node(
        self,
        *,
        node_id: str | None = None,
        label: str | None = None,
        error_if_missing: bool = True,
    ):
        """
        Remove a registered ExperimentNode from this context.

        Exactly one of `node_id` or `label` must be provided.

        Args:
            node_id (str | None):
                Internal node UUID to remove.
            label (str | None):
                Node label to remove.
            error_if_missing (bool):
                Whether to raise if the node does not exist.

        Returns:
            ExperimentNode | None:
                The removed node if found, otherwise None.

        Raises:
            ValueError:
                If neither or both of `node_id` / `label` are provided.
            KeyError:
                If the node does not exist and `error_if_missing=True`.

        """
        if (node_id is None) == (label is None):
            raise ValueError("Must provide exactly one of `node_id` or `label`.")

        # Resolve node_id if label was given
        if label is not None:
            node_id = self._node_label_to_id.get(label)
            if node_id is None:
                if error_if_missing:
                    msg = f"No ExperimentNode with label '{label}'."
                    raise KeyError(msg)
                return None

        # Remove node
        node = self._nodes_by_id.pop(node_id, None)
        if node is None:
            if error_if_missing:
                msg = f"No ExperimentNode with id '{node_id}'."
                raise KeyError(msg)
            return None

        # Remove label mapping
        self._node_label_to_id.pop(node.label, None)

        return node

    def update_node_label(
        self,
        node_id: str,
        new_label: str,
        *,
        check_label_collision: bool = True,
    ):
        """
        Update the label mapping for a registered node.

        Args:
            node_id (str): Identifier of the node whose label is updated.
            new_label (str): Replacement label.
            check_label_collision (bool, optional):
                Whether to enforce uniqueness of labels. Defaults to True.

        Raises:
            KeyError: If the node ID is not registered.
            ValueError: If a collision occurs and `check_label_collision` is True.

        """
        if node_id not in self._nodes_by_id:
            msg = f"Node ID '{node_id}' not registered."
            raise KeyError(msg)

        old_label = self._nodes_by_id[node_id].label

        # If unchanged -> skip
        if new_label == old_label:
            return

        # Check collision
        if check_label_collision and (new_label in self._node_label_to_id):
            msg = f"ExperimentNode label '{new_label}' already exists."
            raise ValueError(msg)

        # Update registry mapping
        self._node_label_to_id.pop(old_label, None)
        self._node_label_to_id[new_label] = node_id

    def register_model_graph(self, graph: ModelGraph):
        """
        Register a ModelGraph to this context.

        Args:
            graph (ModelGraph): Model graph instance to associate.

        Raises:
            TypeError: If `graph` is not a :class:`ModelGraph`.
            ValueError: If overwrite is disallowed and a graph already exists.

        """
        from modularml.core.topology.model_graph import ModelGraph

        # Validate graph
        if not isinstance(graph, ModelGraph):
            msg = f"`graph` must be a ModelGraph instance. Received: {type(graph)}"
            raise TypeError(msg)

        if self._policy == RegistrationPolicy.NO_REGISTER:
            return

        # Check collisions
        if self._mg is not None:
            if self._policy == RegistrationPolicy.ERROR:
                msg = "A ModelGraph has already been registered to this context."
                raise ValueError(msg)

            if self._policy == RegistrationPolicy.OVERWRITE_WARN:
                msg = f"Overwriting existing ModelGraph '{self._mg.label}'."
                warn(msg, category=UserWarning, stacklevel=2)

        # Update internal reference
        self._mg = graph

    def remove_model_graph(self):
        """Removes the registered ModelGraph from this context."""
        self._mg = None

    # ================================================
    # Node Lookup
    # ================================================
    def has_node(self, *, node_id: str | None = None, label: str | None = None) -> bool:
        """Check whether node is registered in this context."""
        if node_id is not None:
            return node_id in self._nodes_by_id
        if label is not None:
            return label in self._node_label_to_id
        raise ValueError("Must provide `node_id` or `label`.")

    def get_node(
        self,
        val: str | None = None,
        *,
        node_id: str | None = None,
        label: str | None = None,
        enforce_type: str = "ExperimentNode",
    ) -> ExperimentNode:
        """
        Retrieve the specified node, as registered in this context.

        Args:
            val (str, optional):
                Either the ID or label of a node. ID is checked first.
                If provided, `node_id` and `label` must be None.

            node_id (str, optional):
                ID of node to retrieve.
                If provided, `val` and `label` must be None.

            label (str, optional):
                Label of node to retrieve.
                If provided, `val` and `node_id` must be None.

            enforce_type (type, optional):
                If specified, additional validation is performed to ensure the
                reutrn node is of the specified type. Defaults to "ExperimentNode".

        """
        node = None

        # If val, check ID then label
        if val is not None:
            if node_id is not None or label is not None:
                msg = "`node_id` and `label` must be None if `val` is defined."
                raise ValueError(msg)
            if self.has_node(node_id=val):
                return self.get_node(node_id=val, enforce_type=enforce_type)
            return self.get_node(label=val, enforce_type=enforce_type)

        # Get node from node_id
        if node_id is not None:
            if val is not None or label is not None:
                msg = "`val` and `label` must be None if `node_id` is defined."
                raise ValueError(msg)
            try:
                node = self._nodes_by_id[node_id]
            except KeyError as exc:
                msg = f"No ExperimentNode with id '{node_id}'"
                raise KeyError(msg) from exc

        # Get node from label
        elif label is not None:
            if val is not None or node_id is not None:
                msg = "`val` and `node_id` must be None if `label` is defined."
                raise ValueError(msg)
            try:
                node = self._nodes_by_id[self._node_label_to_id[label]]
            except KeyError as exc:
                msg = f"No ExperimentNode with label '{label}'"
                raise KeyError(msg) from exc

        else:
            raise ValueError("Must provide node_id or label.")

        # Enforce node type
        if enforce_type == "ExperimentNode":
            from modularml.core.experiment.experiment_node import ExperimentNode

            if not isinstance(node, ExperimentNode):
                msg = f"Retrieved node is not of type '{enforce_type}'. Received: {type(node)}."
                raise TypeError(msg)
            return node

        if enforce_type == "GraphNode":
            from modularml.core.topology.graph_node import GraphNode

            if not isinstance(node, GraphNode):
                msg = f"Retrieved node is not of type '{enforce_type}'. Received: {type(node)}."
                raise TypeError(msg)
            return node

        if enforce_type == "ComputeNode":
            from modularml.core.topology.compute_node import ComputeNode

            if not isinstance(node, ComputeNode):
                msg = f"Retrieved node is not of type '{enforce_type}'. Received: {type(node)}."
                raise TypeError(msg)
            return node

        if enforce_type == "ModelNode":
            from modularml.core.topology.model_node import ModelNode

            if not isinstance(node, ModelNode):
                msg = f"Retrieved node is not of type '{enforce_type}'. Received: {type(node)}."
                raise TypeError(msg)
            return node

        if enforce_type == "MergeNode":
            from modularml.core.topology.merge_nodes.merge_node import MergeNode

            if not isinstance(node, MergeNode):
                msg = f"Retrieved node is not of type '{enforce_type}'. Received: {type(node)}."
                raise TypeError(msg)
            return node

        if enforce_type == "FeatureSet":
            from modularml.core.data.featureset import FeatureSet

            if not isinstance(node, FeatureSet):
                msg = f"Retrieved node is not of type '{enforce_type}'. Received: {type(node)}."
                raise TypeError(msg)
            return node

        msg = f"Unsupported `enforce_type`: {enforce_type}."
        raise ValueError(msg)

    @property
    def available_nodes(self) -> dict[str, ExperimentNode]:
        """
        All registered ExperimentNodes.

        Returns:
            dict[str, ExperimentNode]:
                Nodes keyed by node_id.

        """
        return self._nodes_by_id

    @property
    def available_computenodes(self) -> dict[str, ComputeNode]:
        """
        All registered ComputeNode.

        Returns:
            dict[str, ComputeNode]:
                Nodes keyed by node_id.

        """
        from modularml.core.topology.compute_node import ComputeNode

        cnodes = {}
        for n in self._nodes_by_id.values():
            if isinstance(n, ComputeNode):
                cnodes[n.node_id] = n
        return cnodes

    @property
    def available_featuresets(self) -> dict[str, FeatureSet]:
        """
        All registered FeatureSets.

        Returns:
            dict[str, FeatureSet]:
                Nodes keyed by node_id.

        """
        from modularml.core.data.featureset import FeatureSet

        fnodes = {}
        for n in self._nodes_by_id.values():
            if isinstance(n, FeatureSet):
                fnodes[n.node_id] = n
        return fnodes

    @property
    def model_graph(self) -> ModelGraph | None:
        """The active ModelGraph instance in this context."""
        return self._mg

    # ================================================
    # Stateful
    # ================================================
    def get_state(self) -> dict[str, Any]:
        """
        Capture the current registration state for restoration.

        Returns:
            dict[str, Any]: Snapshot that can be supplied to :meth:`set_state`.

        """
        return {
            "nodes": self._nodes_by_id.copy(),  # shallow
            "node_states": {
                k: (v.get_state() if hasattr(v, "get_state") else None)
                for k, v in self._nodes_by_id.items()
            },
            "model_graph": self._mg,
            "model_graph_state": self._mg.get_state() if self._mg is not None else None,
            "policy": self._policy,
            "experiment_ref": self._experiment_ref,
        }

    def set_state(self, state: dict[str, Any]):
        """
        Restore the context from a serialized state snapshot.

        Args:
            state (dict[str, Any]): Snapshot produced by :meth:`get_state`.

        """
        self.clear_registries()
        self._experiment_ref = state["experiment_ref"]
        self._policy = state["policy"]

        # Restore all nodes
        self._nodes_by_id = state["nodes"]
        self._node_label_to_id = {}
        for node_id, n in self._nodes_by_id.items():
            if hasattr(n, "set_state"):
                n.set_state(state["node_states"][node_id])
            self._node_label_to_id[n.label] = node_id

        # Restore model graph
        self._mg = state["model_graph"]
        if self.model_graph is not None:
            self._mg.set_state(state["model_graph_state"])
