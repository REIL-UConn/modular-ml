from __future__ import annotations

import os
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING
from weakref import ref

from matplotlib.pylab import Enum

from modularml.utils.environment.environment import IN_NOTEBOOK

if TYPE_CHECKING:
    from modularml.core.experiment.experiment import Experiment
    from modularml.core.experiment.experiment_node import ExperimentNode


_ACTIVE_EXPERIMENT_CONTEXT: ContextVar[ExperimentContext | None] = ContextVar(
    "_ACTIVE_EXPERIMENT_CONTEXT",
    default=None,
)


class RegistrationPolicy(Enum):
    """Controls behavior when registering objects with duplicate labels."""

    ERROR = "error"
    OVERWRITE = "overwrite"
    NO_REGISTER = "no_register"

    @classmethod
    def from_value(cls, value):
        """Cast a string or enum value to RegistrationPolicy."""
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

        # Registration policy
        if registration_policy is None:
            self._policy = self._resolve_default_policy()
        else:
            self._policy = RegistrationPolicy.from_value(registration_policy)

    # =====================================================
    # Active Context Helpers
    # =====================================================
    @classmethod
    def _set_active(cls, ctx: ExperimentContext):
        _ACTIVE_EXPERIMENT_CONTEXT.set(ctx)

    @classmethod
    def get_active(cls) -> ExperimentContext:
        """Returns the active ExperimentContext, if exists."""
        ctx = _ACTIVE_EXPERIMENT_CONTEXT.get()
        if ctx is None:
            raise RuntimeError("No active ExperimentContext")
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

    # =====================================================
    # Policy Management
    # =====================================================
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

        # 2. If running in Jupyter Notebook -> default to OVERWRITE
        if IN_NOTEBOOK:
            return RegistrationPolicy.OVERWRITE

        # 3. Else --> default to ERROR
        return RegistrationPolicy.ERROR

    def set_registration_policy(self, policy: str | RegistrationPolicy):
        """Permanently set the registration policy."""
        self._policy = RegistrationPolicy.from_value(policy)

    @contextmanager
    def use_policy(self, policy: str | RegistrationPolicy):
        """Temporarily override the registration policy inside a context."""
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

    # =====================================================
    # Experiment Lifecycle
    # =====================================================
    def set_experiment(self, experiment: Experiment):
        """Set the experiment reference for this context."""
        self._experiment_ref = ref(experiment)
        self.clear_registries()

    def get_experiment(self) -> Experiment | None:
        """Returns the Experimetn active in this context, if defined."""
        return None if self._experiment_ref is None else self._experiment_ref()

    # =====================================================
    # Registry Helpers
    # =====================================================
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
            if self._policy is RegistrationPolicy.OVERWRITE:
                old = self._nodes_by_id[node_id]
                self._node_label_to_id.pop(old.label, None)
            else:
                return

        # Label collision checks
        if check_label_collision and (label in self._node_label_to_id):
            if self._policy is RegistrationPolicy.ERROR:
                msg = f"ExperimentNode label '{label}' already exists."
                raise ValueError(msg)
            if self._policy is RegistrationPolicy.OVERWRITE:
                old_id = self._node_label_to_id[label]
                self._nodes_by_id.pop(old_id, None)

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

    # =====================================================
    # Node Lookup
    # =====================================================
    def has_node(self, *, node_id: str | None = None, label: str | None = None) -> bool:
        """Check whether node is registered in this context."""
        if node_id is not None:
            return node_id in self._nodes_by_id
        if label is not None:
            return label in self._node_label_to_id
        raise ValueError("Must provide `node_id` or `label`.")

    def get_node(
        self,
        *,
        node_id: str | None = None,
        label: str | None = None,
        enforce_type: str = "ExperimentNode",
    ) -> ExperimentNode:
        """
        Retrieve the specified node, as registered in this context.

        Args:
            node_id (str, optional):
                ID of node to retrieve.
            label (str, optional):
                Label of node to retrieve. If both node_id and label are provided,
                only node_id is used to resolve the node.
            enforce_type (type, optional):
                If specified, additional validation is performed to ensure the
                reutrn node is of the specified type. Defaults to "ExperimentNode".

        """
        node = None
        if node_id is not None:
            try:
                node = self._nodes_by_id[node_id]
            except KeyError as exc:
                msg = f"No ExperimentNode with id '{node_id}'"
                raise KeyError(msg) from exc

        elif label is not None:
            try:
                node = self._nodes_by_id[self._node_label_to_id[label]]
            except KeyError as exc:
                msg = f"No ExperimentNode with label '{label}'"
                raise KeyError(msg) from exc

        else:
            raise ValueError("Must provide node_id or label.")

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
    def available_nodes(self) -> list[str]:
        """List of registered ExperimentNode labels."""
        return list(self._node_label_to_id.keys())

    @property
    def available_featuresets(self) -> list[str]:
        """List of registered FeatureSet labels."""
        from modularml.core.data.featureset import FeatureSet

        avail_fs_labels = []
        for n in self._nodes_by_id.values():
            if isinstance(n, FeatureSet):
                avail_fs_labels.append(n.label)
        return avail_fs_labels

    @property
    def available_modelnodes(self) -> list[str]:
        """List of registered ModelNode labels."""
        from modularml.core.topology.model_node import ModelNode

        avail_mn_labels = []
        for n in self._nodes_by_id.values():
            if isinstance(n, ModelNode):
                avail_mn_labels.append(n.label)
        return avail_mn_labels
