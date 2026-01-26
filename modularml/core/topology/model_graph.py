from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from modularml.context.experiment_context import ExperimentContext
from modularml.core.io.checkpoint import Checkpoint
from modularml.core.io.protocols import Configurable, Stateful
from modularml.core.references.featureset_reference import FeatureSetReference
from modularml.core.topology.compute_node import ComputeNode, TForward
from modularml.core.topology.graph_node import GraphNode
from modularml.core.topology.model_node import ModelNode
from modularml.core.topology.protocols import Evaluable, Forwardable, Trainable
from modularml.core.training.loss_record import LossCollection, LossRecord
from modularml.core.training.optimizer import Optimizer
from modularml.utils.data.comparators import deep_equal
from modularml.utils.data.data_format import DataFormat, get_data_format_for_backend
from modularml.utils.data.dummy_data import make_dummy_sample_data
from modularml.utils.environment.optional_imports import check_tensorflow, check_torch
from modularml.utils.errors.error_handling import ErrorMode
from modularml.utils.errors.exceptions import BackendNotSupportedError
from modularml.utils.logging.warnings import catch_warnings, warn
from modularml.utils.nn.backend import (
    Backend,
    backend_requires_optimizer,
    is_valid_backend,
)
from modularml.utils.topology.graph_search_utils import (
    find_upstream_featuresets,
    get_subgraph_nodes,
)

if TYPE_CHECKING:
    from modularml.context.execution_context import ExecutionContext
    from modularml.core.data.batch import Batch
    from modularml.core.data.sample_data import SampleData
    from modularml.core.references.experiment_reference import (
        ExperimentNodeReference,
        GraphNodeReference,
    )
    from modularml.core.training.applied_loss import AppliedLoss

tf = check_tensorflow()
torch = check_torch()


class ModelGraph(Configurable, Stateful):
    def __init__(
        self,
        nodes: list[str | GraphNode] | None,
        optimizer: Optimizer | None = None,
        label: str = "model-graph",
        *,
        ctx: ExperimentContext | None = None,
        register: bool = True,
    ):
        """
        Initialize a ModelGraph from a list of modular nodes and an optional global optimizer.

        Args:
            nodes (list[str | GraphNode], optional):
                A list of GraphNodes (e.g., ModeNode) or their labels to construct a ModelGraph
                around. If None, all registered GraphNodes in the active ExperimentContext are used.

            optimizer (Optional[Optimizer], optional):
                A shared optimizer to use for all `nodes` that require one. If provided,
                the graph will ensure that all such stages have a matching backend and override
                any stage-level optimizers. If not provided, each stage that requires an optimizer
                must define one locally.

            label (str, optional):
                Optional label to assign to this ModelGraph instance. Defaults to "model-graph-0".

            ctx (ExperimentContext, optional):
                The ExperimentContext this ModelGraph should exist within. If None, uses the
                active ExperimentContext. Defaults to None.

            register (bool, optional):
                Used only for de-serialization.

        Raises:
            ValueError:
                If duplicate node labels are provided or if graph connectivity is invalid.
            RuntimeError:
                If required optimizers are missing or if backends are incompatible.

        """
        # Register to context
        self.exp_ctx = ctx or ExperimentContext.get_active()
        if register:
            self.exp_ctx.register_model_graph(self)

        # Update default label to next available name
        self.label = label

        # Nodes comprising this graph, keyed by node_id
        self._nodes: dict[str, GraphNode] = {}
        if nodes is not None:
            for n in nodes:
                cn = n
                if isinstance(n, str):
                    if self.exp_ctx.has_node(label=n):
                        cn = self.exp_ctx.get_node(label=n)
                    else:
                        msg = f"String value given in `nodes` ('{n}') does not exist in the active Experiment Context."
                        raise ValueError(msg)

                if not isinstance(cn, GraphNode):
                    msg = f"All objects in `nodes` must be of type GraphNode. Received: {type(cn)}."
                    raise TypeError(msg)
                self._nodes[cn.node_id] = cn
        else:
            self._nodes = self.exp_ctx.available_computenodes

        self._optimizer = optimizer
        # Nodes that require an optimizer
        self._nodes_req_opt: dict[str, ModelNode] | None = None
        # Nodes used in building a global optimizer
        self._opt_built_from_node_ids: set[str] | None = None
        self._optimizer_state: dict[str, Any] | None = None

        # Connection helpers
        self._head_nodes: dict[str, GraphNode] = {}
        self._tail_nodes: dict[str, GraphNode] = {}
        self._rebuild_connections()

    # ================================================
    # Properties & Dunders
    # ================================================
    @property
    def nodes(self) -> dict[str, GraphNode]:
        """All GraphNode in the ModelGraph, keyed by `node_id`."""
        return self._nodes

    @property
    def node_labels(self) -> set[str]:
        """Returns the set of unique node labels in this ModelGraph."""
        lbls = []
        for n in self.nodes.values():
            lbls.append(n.label)

        unique_labels = set(lbls)
        if len(unique_labels) != len(lbls):
            msg = "This ModelGraph contains nodes with identical labels. Only unique labels are returned."
            hint = "Use the `nodes` property to retrieve all unique nodes and ids."
            warn(msg, category=UserWarning, hints=hint, stacklevel=2)
        return unique_labels

    @property
    def head_nodes(self) -> dict[str, GraphNode]:
        """
        Head nodes of the ModelGraph.

        Description:
            Head nodes are GraphNodes whose inputs originate directly from
            FeatureSets (i.e., they have no upstream GraphNode dependencies).

        Returns:
            dict[str, GraphNode]:
                Mapping of node_id to GraphNode for all head nodes.

        """
        return self._head_nodes

    @property
    def tail_nodes(self) -> dict[str, GraphNode]:
        """
        Tail nodes of the graph.

        Description:
            Tail nodes are GraphNodes whose outputs are not consumed
            by any other GraphNode in the ModelGraph.

        Returns:
            dict[str, GraphNode]:
                Mapping of node_id to GraphNode for all tail nodes.

        """
        return self._tail_nodes

    @property
    def is_built(self) -> bool:
        return self._built

    def __eq__(self, other: ModelGraph):
        if not isinstance(other, ModelGraph):
            msg = f"Cannot compare equality between ModelGraph and {type(other)}"
            raise TypeError(msg)

        if not self.label == other.label:
            return False

        if not deep_equal(self.get_config(), other.get_config()):
            return False

        return deep_equal(self.get_state(), other.get_state())

    __hash__ = None

    # ================================================
    # Error Checking Methods
    # ================================================
    def _validate_graph_connections(self):
        """
        Validates the internal graph structure.

        Perform the following checks:
        - Ensures nodes are valid GraphNode instances.
        - Propagates inputs to upstream node outputs.
        - Validates input/output limits.
        - Ensures the graph is a DAG (no cycles).
        - Ensures all nodes are reachable from at least one base node.

        Raises:
            TypeError: If any node is not a GraphNode.
            KeyError: If a node references a non-existent input.
            ValueError: If input/output constraints are violated or if a cycle is detected.
            UserWarning: If unreachable nodes are found or mixed backends are used.

        """
        used_backends = []  # record all node backends (for checking)
        frontier = []  # get base nodes (for traversal / connection checks)

        # Ensure node inherits from GraphNode, and input/ouput properties are fully set
        for n_id, node in self._nodes.items():
            if not isinstance(node, GraphNode):
                msg = f"ModelGraph nodes must be of type GraphNode. Received: {node}"
                raise TypeError(msg)

            # Record backend for later checking
            if hasattr(node, "backend") and is_valid_backend(node.backend):
                used_backends.append(node.backend)

            # Record base nodes (any GraphNode with upstream_refs = FeatureSetReference)
            all_up_refs = node.get_upstream_refs()
            if all(isinstance(x, FeatureSetReference) for x in all_up_refs):
                frontier.append(n_id)

            # Validate all upstream connections
            for ups_ref in all_up_refs:
                # Ensure node exists in current ctx
                ups_node = ups_ref.resolve(ctx=self.exp_ctx)

                # If a FeatureSet (or view), continue
                if isinstance(ups_ref, FeatureSetReference):
                    continue

                # Ensure this upstream node (a GraphNode) is also in the graph
                if ups_node.node_id not in self._nodes:
                    msg = (
                        f"Upstream node '{ups_node.label}' for node '{node.label}'"
                        "not found in ModelGraph."
                    )
                    raise KeyError(msg)

                if not isinstance(ups_node, GraphNode):
                    msg = (
                        "Non-FeatureSet references must resolve to GraphNodes. "
                        f"Received: {type(ups_node)}."
                    )
                    raise TypeError(msg)

                # Ensure the upstream node also references this node in its output (if has outputs)
                # Using "coerce" ignores the warning if the reference already exists
                # but still raises error if reached max number of downstream nodes
                ups_node.add_downstream_ref(
                    node.reference(),
                    error_mode=ErrorMode.COERCE,
                )

        # Warn if using mixed backend: not thoroughly tested
        if len(set(used_backends)) > 1:
            msg = (
                "Mixed backends detected in ModelGraph. Though allowed, minimal testing has been "
                "conducted. Gradient flow may break during training."
            )
            warn(msg, category=UserWarning, stacklevel=2)

        # Ensure is DAG (check for cycles)
        visited = set()
        visiting = set()

        def dfs(node: GraphNode):
            """Depth first search."""
            if node.node_id in visiting:
                msg = f"Cycle detected in graph at node '{node.label}'. Graph must be acyclic."
                raise ValueError(msg)
            if node.node_id in visited:
                return
            visiting.add(node.node_id)

            for dwn_ref in node.get_downstream_refs(error_mode=ErrorMode.IGNORE):
                dfs(dwn_ref.resolve(self.exp_ctx))

            visiting.remove(node.node_id)
            visited.add(node.node_id)

        # Perform depth-first-search starting at head nodes
        for root_node_id in frontier:
            node = self.exp_ctx.get_node(node_id=root_node_id)
            dfs(node)

        # Ensure reachability of all nodes
        reachable = set()
        queue = list(frontier)
        while queue:
            cur_node_id = queue.pop(0)
            if cur_node_id in reachable:
                continue
            reachable.add(cur_node_id)
            cur_node: GraphNode = self._nodes[cur_node_id]
            dwn_node_ids: list[str] = [
                ref.node_id
                for ref in cur_node.get_downstream_refs(
                    error_mode=ErrorMode.IGNORE,
                )
            ]
            queue.extend(dwn_node_ids)

        unreachable_node_ids = set(self._nodes.keys()) - reachable
        if unreachable_node_ids:
            node_labels = [self.nodes[un].label for un in unreachable_node_ids]
            msg = f"Unreachable nodes detected in ModelGraph: {sorted(node_labels)}."
            hint = "Verify upstream reference atributes of all GraphNodes."
            warn(msg, category=UserWarning, stacklevel=2, hints=hint)

    def _topological_sort(self) -> list[str]:
        """
        Perform a topological sort of the ModelGraph using Kahn's algorithm.

        Returns:
            List[str]: A list of node labels in topological (execution) order.

        Raises:
            ValueError: If a cycle is detected in the graph.

        """
        in_degree = defaultdict(int)  # Number of incoming edges (keyed by node ID)
        children = defaultdict(list)  # Outgoing edges (keyed by node ID)
        all_node_ids = set(self._nodes.keys())

        # Initialize in-degrees
        for node_id in all_node_ids:
            in_degree[node_id] = 0

        # Record in-degree (number of inputs) and out-degree for each node
        for node_id, node in self._nodes.items():
            # Get parents of this node (ie, upstream)
            parent_node_ids: list[str] = [
                ref.node_id
                for ref in node.get_upstream_refs(
                    error_mode=ErrorMode.IGNORE,
                )
                if not isinstance(ref, FeatureSetReference)
            ]
            for parent_id in parent_node_ids:
                if parent_id not in self._nodes:
                    p_node = self.exp_ctx.get_node(node_id=parent_id)
                    msg = f"Invalid upstream_node '{p_node.label}' for node `{node.label}`."
                    raise KeyError(msg)
                in_degree[node_id] += 1
                children[parent_id].append(node_id)

        # Init a queue with base nodes (no inputs)
        sorted_node_ids: list[str] = []
        queue = deque([node_id for node_id in all_node_ids if in_degree[node_id] == 0])
        while queue:
            current = queue.popleft()
            sorted_node_ids.append(current)
            for child in children[current]:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

        if len(sorted_node_ids) != len(all_node_ids):
            unresolved = all_node_ids - set(sorted_node_ids)
            unres_node_lbls = [
                self.exp_ctx.get_node(node_id=un).label for un in unresolved
            ]
            msg = f"Cyclic dependency detected in ModelGraph: {unres_node_lbls}"
            raise ValueError(msg)

        return sorted_node_ids

    def _validate_optimizer(self):
        """
        Validate and assign optimizers to all trainable stages in the graph.

        Description:
            This method ensures that all GraphNodes which require an optimizer
            (based on their backend) are properly configured with one.

            - If a global optimizer is provided to the ModelGraph, it will be assigned
            to all relevant stages. If those stages already define a local optimizer,
            it will be overwritten with a warning.

            - If no global optimizer is provided, then every stage that requires an optimizer
            must have its own stage-level optimizer defined.

            - It also verifies that all optimizers share a consistent backend (e.g., PyTorch).

        Raises:
            RuntimeError: If any stage that requires an optimizer is missing one and no
                        global optimizer is provided.
            RuntimeError: If a global optimizer is provided but its backend doesn't match
                        a stage's backend.
            UserWarning: If a stage has its own optimizer but is being overwritten by the
                        graph-level optimizer.

        """
        # Get nodes that require optimizer (only ModelNodes)
        self._nodes_req_opt: dict[str, ModelNode] = {
            node_id: node
            for node_id, node in self._nodes.items()
            if isinstance(node, ModelNode) and backend_requires_optimizer(node.backend)
        }

        # Ensure all stages have their own optimizer if global one not provided
        if self._optimizer is None:
            for node in self._nodes_req_opt.values():
                if not hasattr(node, "_optimizer") or node._optimizer is None:
                    msg = (
                        f"ModelNode '{node.label}' is missing an optimizer. "
                        f"Provide one at the stage level or to ModelGraph."
                    )
                    raise RuntimeError(msg)

        # Ensure all stages have the same backend
        else:
            used_backends = []
            for node in self._nodes_req_opt.values():
                used_backends.append(node.backend)

                # Overwrite existing optimizers at stage-level (and warn)
                if node._optimizer is not None:
                    msg = (
                        f"An optimizer was provided to both the ModelGraph and the '{node.label}' "
                        f"ModelNode. The optimizer for '{node.label}' will be overwritten."
                    )
                    warn(msg, category=UserWarning, stacklevel=2)
                    node._optimizer = None

            # Warn if using mixed backend: not thoroughly tested
            if len(set(used_backends)) > 1:
                msg = (
                    "A global optimizer was provided to ModelGraph, but the underlying stages have "
                    "differing backends. All backends must match to use a single optimizer."
                )
                raise RuntimeError(msg)
            self._optimizer.backend = used_backends[0]

    def _rebuild_connections(self):
        # Clear downstream references (auto-generated in validation)
        for n in self._nodes.values():
            if hasattr(n, "clear_downstream_refs"):
                n.clear_downstream_refs(ErrorMode.IGNORE)

        # Validate graph and connections
        self._validate_graph_connections()

        # Cache head/tail nodes (must be after validation ^)
        head_nodes: dict[str, GraphNode] = {}
        tail_nodes: dict[str, GraphNode] = {}
        for n_id, node in self._nodes.items():
            # Head nodes: inputs from a FeatureSet
            ups = node.get_upstream_refs(error_mode=ErrorMode.IGNORE)
            if ups and all(isinstance(r, FeatureSetReference) for r in ups):
                head_nodes[n_id] = node

            # Tail nodes: no downstream consumers
            dwn = node.get_downstream_refs(error_mode=ErrorMode.IGNORE)
            if len(dwn) == 0:
                tail_nodes[n_id] = node

        self._head_nodes = head_nodes
        self._tail_nodes = tail_nodes

        # Topological sort
        self._sorted_node_ids = self._topological_sort()

        # If an optimizer is provided, check that:
        # 1. all optimizer-requiring stages have same backend
        # 2. warn if stages have their own optimizer (will be overwritten)
        self._validate_optimizer()

        self._built = False

    # ================================================
    # Connection Modifiers
    # ================================================
    def _resolve_existing(self, val: str | GraphNode) -> GraphNode:
        """
        Verifies that the given values corresponds to a node in this graph.

        Args:
            val (str | GraphNode):
                Node ID, label, or instance of a node in this graph.

        Returns:
            GraphNode:
                The node instance of the existing node. Throws an error
                if the value cannot be resolved to an existing node.

        """
        # Normalize value to GraphNode instance
        existing_node: GraphNode | None = None
        if isinstance(val, str):
            if val in self._nodes:
                existing_node = self._nodes[val]
            else:
                # Get existing node labels, if not unique, throw error
                existing_node_lbls = None
                with catch_warnings() as cw:
                    existing_node_lbls = self.node_labels
                    if cw.match("contains nodes with identical labels"):
                        existing_node_lbls = None
                if existing_node_lbls is None:
                    msg = (
                        "ModelGraph contains nodes with identical labels. Existing nodes "
                        "must be referenced with either their node ID or the actual instance."
                    )
                    raise ValueError(msg)
                # Get node instance with that label
                matches = [n for n in self._nodes.values() if n.label == val]
                if not matches:
                    msg = f"No node exists in this graph with label '{val}'."
                    raise ValueError(msg)
                existing_node = matches[0]
        elif isinstance(val, GraphNode):
            if val.node_id not in self._nodes:
                msg = f"No node exists in this graph with id '{val.node_id}'."
                raise ValueError(msg)
            existing_node = val
        else:
            msg = f"Existing node value must be of type `str` or `GraphNode`. Received: {type(val)}."
            raise TypeError(msg)

        return existing_node

    def add_node(self, node: GraphNode) -> ModelGraph:
        """
        Add a new node to the graph.

        Description:
            This modifies graph structure only; no existing node states
            are reset or copied. The added node must already be registered
            in the ExperimentContext.

        Args:
            node (GraphNode): Node to add.

        Returns:
            ModelGraph: self

        Raises:
            ValueError: If a node with the same `node_id` already exists.

        """
        if not isinstance(node, GraphNode):
            msg = f"Expected GraphNode, got {type(node)}"
            raise TypeError(msg)
        if node.node_id in self._nodes:
            msg = f"Node '{node.label}' already exists in ModelGraph."
            raise ValueError(msg)

        self._nodes[node.node_id] = node
        self._rebuild_connections()
        return self

    def replace_node(
        self,
        old_node: str | GraphNode,
        new_node: GraphNode,
    ) -> ModelGraph:
        """
        Replace an existing node while preserving all upstream and downstream connections.

        Description:
            Connectivity of the graph is preserved, and learned state of the other nodes
            is unaffected. The replaced node's state is replaced with the new node.

        Args:
            old_node (str | GraphNode):
                Existing node in the graph to be replaced. Provided argument value can be
                the existing node's label, ID, or the actual node instance.
                The state of the existing node is not changed; the graph connections
                are simply redirected to the `new_node`.

            new_node (GraphNode):
                New node instance to take the spot of `old_node`.

        Returns:
            ModelGraph: self

        """
        # Normalize old_node valus
        old_node = self._resolve_existing(val=old_node)

        # Validate new_node type
        if not isinstance(new_node, GraphNode):
            msg = f"New node must be of type GraphNode, got {type(new_node)}"
            raise TypeError(msg)

        # Grab connection to/from old_node
        ups_refs: list[GraphNodeReference] = old_node.get_upstream_refs(
            error_mode=ErrorMode.IGNORE,
        )
        dwn_refs: list[GraphNodeReference] = old_node.get_downstream_refs(
            error_mode=ErrorMode.IGNORE,
        )

        # Update new_node to match old_node connections
        new_node.set_upstream_refs(ups_refs, error_mode=ErrorMode.COERCE)
        new_node.set_downstream_refs(dwn_refs, error_mode=ErrorMode.COERCE)

        # Update all nodes upstream of old_node
        for ref in ups_refs:
            # Get all downstream refs of the upstream node, removing the old_node ref
            u_dwn_refs = [
                r
                for r in self._nodes[ref.node_id].get_downstream_refs(
                    error_mode=ErrorMode.IGNORE,
                )
                if r.node_id != old_node.node_id
            ]
            # Replace with cleaned refs
            self._nodes[ref.node_id].set_downstream_refs(
                u_dwn_refs,
                error_mode=ErrorMode.IGNORE,
            )
            # Add new_node reference
            self._nodes[ref.node_id].add_downstream_ref(
                new_node.reference(),
                error_mode=ErrorMode.IGNORE,
            )

        # Update all nodes downstream of old_node
        for ref in dwn_refs:
            # Get all upstream refs of the downstream node, removing the old_node ref
            u_ups_refs = [
                r
                for r in self._nodes[ref.node_id].get_upstream_refs(
                    error_mode=ErrorMode.IGNORE,
                )
                if r.node_id != old_node.node_id
            ]
            # Replace with cleaned refs
            self._nodes[ref.node_id].set_upstream_refs(
                u_ups_refs,
                error_mode=ErrorMode.IGNORE,
            )
            # Add new_node reference
            self._nodes[ref.node_id].add_upstream_ref(
                new_node.reference(),
                error_mode=ErrorMode.IGNORE,
            )

        # Replace self._nodes
        _ = self._nodes.pop(old_node.node_id)
        self._nodes[new_node.node_id] = new_node
        self._rebuild_connections()
        return self

    def insert_node_between(
        self,
        new_node: GraphNode,
        *,
        upstream: str | GraphNode,
        downstream: str | GraphNode,
    ) -> ModelGraph:
        """
        Insert a node between two, already connected, nodes.

        Description:
            Insert a new node between a connection of two existing node.
            The old connection (upstream -> downstream) is replaced with
            (upstream -> new_node -> downstream).
            An error will be thrown if the existing nodes are not already
            connected.

        Args:
            new_node (GraphNode):
                New node instance to be inserted.

            upstream (str | GraphNode):
                Node ID, label, or instance of an existing ModelGraph node.
                The `new_node` will be inserted downstream of this node.

            downstream (str | GraphNode):
                Node ID, label, or instance of an existing ModelGraph node.
                The `new_node` will be inserted upstream of this node.

        Returns:
            ModelGraph: self

        """
        # Normalize existing node valus
        ups_node: GraphNode = self._resolve_existing(val=upstream)
        dwn_node: GraphNode = self._resolve_existing(val=downstream)

        # Validate new_node type
        if not isinstance(new_node, GraphNode):
            msg = f"New node must be of type GraphNode, got {type(new_node)}"
            raise TypeError(msg)

        # Clear any references on new_node
        new_node.clear_upstream_refs(error_mode=ErrorMode.COERCE)
        new_node.clear_downstream_refs(error_mode=ErrorMode.COERCE)

        # Validate that `downstream` connects to `upstream`
        existing_dwn_to_ups: GraphNodeReference | FeatureSetReference = None
        for ref in dwn_node.get_upstream_refs(error_mode=ErrorMode.IGNORE):
            if ref.node_id == ups_node.node_id:
                existing_dwn_to_ups = ref
        if existing_dwn_to_ups is None:
            dwn_ups_n_lbls = [
                ref.node_label
                for ref in dwn_node.get_upstream_refs(
                    error_mode=ErrorMode.IGNORE,
                )
            ]
            ups_dwn_n_lbls = [
                ref.node_label
                for ref in ups_node.get_downstream_refs(
                    error_mode=ErrorMode.IGNORE,
                )
            ]
            msg = f"`downstream` does not take input `upstream`. Detected inputs from: {dwn_ups_n_lbls}."
            raise ValueError(msg)

        # Validate that `upstream` connect to `downstream`
        existing_ups_to_dwn: GraphNodeReference = None
        for ref in ups_node.get_downstream_refs(error_mode=ErrorMode.IGNORE):
            if ref.node_id == dwn_node.node_id:
                existing_ups_to_dwn = ref
                break
        if existing_ups_to_dwn is None:
            ups_dwn_n_lbls = [
                ref.node_label
                for ref in ups_node.get_downstream_refs(
                    error_mode=ErrorMode.IGNORE,
                )
            ]
            msg = f"`upstream` does not output to `downstream`. Detected outputs to: {ups_dwn_n_lbls}."
            raise ValueError(msg)

        # Replace downstream connection of `upstream`
        # 'ups -> dwn' with 'ups -> new'
        ups_node.remove_downstream_ref(
            ref=existing_ups_to_dwn,
            error_mode=ErrorMode.RAISE,
        )
        ups_node.add_downstream_ref(
            ref=new_node.reference(),
            error_mode=ErrorMode.RAISE,
        )

        # Replace upstream connection of `downstream`
        # 'ups -> dwn' with 'new -> dwn'
        dwn_node.remove_upstream_ref(
            ref=existing_dwn_to_ups,
            error_mode=ErrorMode.RAISE,
        )
        dwn_node.add_upstream_ref(
            ref=new_node.reference(),
            error_mode=ErrorMode.RAISE,
        )

        # Update `new_node` connection
        new_node.add_upstream_ref(ref=ups_node.reference())
        new_node.add_downstream_ref(ref=dwn_node.reference())

        # Add new_node to graph
        self._nodes[new_node.node_id] = new_node
        self._rebuild_connections()
        return self

    def insert_node_before(
        self,
        new_node: GraphNode,
        *,
        downstream: str | GraphNode,
    ) -> ModelGraph:
        """
        Insert a node before an existing node.

        Description:
            Inserts a new node before an existing GraphNode. All inputs
            to the existing node are attached to the new node. The
            existing node will now only receive input from the new node.

        Args:
            new_node (GraphNode):
                New node instance to be inserted.

            downstream (str | GraphNode):
                Node ID, label, or instance of an existing ModelGraph node.
                The `new_node` will be inserted upstream of this node.

        Returns:
            ModelGraph: self

        """
        # Normalize existing node valus
        dwn_node: GraphNode = self._resolve_existing(val=downstream)

        # Validate new_node type
        if not isinstance(new_node, GraphNode):
            msg = f"New node must be of type GraphNode, got {type(new_node)}"
            raise TypeError(msg)

        # Clear any references on new_node
        new_node.clear_upstream_refs(error_mode=ErrorMode.COERCE)
        new_node.clear_downstream_refs(error_mode=ErrorMode.COERCE)

        # Move upstream refs of `downstream` to new node
        for ref in dwn_node.get_upstream_refs(error_mode=ErrorMode.IGNORE):
            new_node.add_upstream_ref(ref=ref)

        # `downstream` should now only get input data from `new_node`
        dwn_node.clear_upstream_refs()
        dwn_node.add_upstream_ref(new_node.reference())
        new_node.add_downstream_ref(dwn_node.reference())

        # Add new_node to graph
        self._nodes[new_node.node_id] = new_node
        self._rebuild_connections()
        return self

    def insert_node_after(
        self,
        new_node: GraphNode,
        *,
        upstream: str | GraphNode,
    ) -> ModelGraph:
        """
        Insert a node after an existing node.

        Description:
            Inserts a new node after an existing GraphNode.
            A new output connection is added between the existing
            node and the new node. All other output connection are
            left undisturbed. The new node will only receive input
            from the existing node.

        Args:
            new_node (GraphNode):
                New node instance to be inserted.

            upstream (str | GraphNode):
                Node ID, label, or instance of an existing ModelGraph node.
                The `new_node` will be inserted downstream of this node.

        Returns:
            ModelGraph: self

        """
        # Normalize existing node value
        ups_node: GraphNode = self._resolve_existing(val=upstream)

        # Validate new_node type
        if not isinstance(new_node, GraphNode):
            msg = f"New node must be of type GraphNode, got {type(new_node)}"
            raise TypeError(msg)

        # Clear any references on new_node
        new_node.clear_upstream_refs(error_mode=ErrorMode.COERCE)
        new_node.clear_downstream_refs(error_mode=ErrorMode.COERCE)

        # Attach new_node as an output of `upstream`
        ups_node.add_downstream_ref(new_node.reference())
        new_node.add_upstream_ref(ups_node.reference())

        # Add new_node to graph
        self._nodes[new_node.node_id] = new_node
        self._rebuild_connections()
        return self

    def remove_node(self, node: str | GraphNode) -> ModelGraph:
        """
        Remove an existing node from the graph.

        Description:
            The existing node is removed an all connections are updated.
            Any nodes downstream of `node` will re-route inputs to *all*
            nodes that previously provided input to `node`.

        Args:
            node (GraphNode):
                Node ID, label, or instance of an existing ModelGraph node
                to be removed.

        Examples:
        1. Removing an existing single-input, single-output node.
            Given: `A -> B -> C`
            Remove: `B`
            Result: `A -> C`

        2. Removing an existing multi-input, single-output node.
            Given: `[A, B] -> C -> D`
            Remove: `C`
            Result: `[A, B] -> D`
            Note that `D` must be able to accept multiple inputs or an error
            will be thrown.

        3. Removing an existing single-input, multi-output node.
            Given: `A -> B -> [C, D]`
            Remove: `B`
            Result: `A -> [C, D]`
            Note that `A` must be able to accept multiple outputs or an error
            will be thrown.

        Returns:
            ModelGraph: self

        """
        # Normalize existing node value
        node: GraphNode = self._resolve_existing(val=node)

        # Get all upstream and downstream refs for later use
        all_ups_refs = node.get_upstream_refs(error_mode=ErrorMode.IGNORE)
        all_dwn_refs = node.get_downstream_refs(error_mode=ErrorMode.IGNORE)

        # Update all nodes downstream of `node`
        # They now should take input from all nodes upstream of `node`
        for dwn_ref in all_dwn_refs:
            dwn_node = dwn_ref.resolve(ctx=self.exp_ctx)

            # Remove reference to `node`
            dwn_node.remove_upstream_ref(node.reference())

            # Add connection to all of `node`'s upstream refs
            for r in all_ups_refs:
                dwn_node.add_upstream_ref(r)

        # Update all nodes upstream of `node`
        # They now should output to all nodes downstream of `node`
        for ups_ref in all_ups_refs:
            ups_node = ups_ref.resolve(ctx=self.exp_ctx)

            # Remove reference to `node`
            ups_node.remove_downstream_ref(node.reference())

            # Add connection to all of `node`'s downstream refs
            for r in all_dwn_refs:
                ups_node.add_downstream_ref(r)

        # Remove node from the graph
        _ = self._nodes.pop(node.node_id)
        self._rebuild_connections()
        return self

    # ================================================
    # Graph Construction
    # ================================================
    def _select_optimizer_parameters(
        self,
        nodes_to_include: list[str] | None = None,
        *,
        include_only_unfrozen: bool = True,
    ) -> tuple[dict[str, list[Any]], set[str]]:
        """
        Collect trainable parameters / variables from ModelNodes for optimizer usage.

        Args:
            nodes_to_include (list[str] | None):
                A list of node IDs to consider for parameter extraction.
                If None, all nodes in this graph are considered.

            include_only_unfrozen (bool, optional):
                If True, only nodes in `nodes_to_include` that are not frozen
                will be used for parameter extraction. If False, all nodes in
                `nodes_to_include` are used (i.e., ignores any frozen state).

        Returns:
            dict[str, list[Any]]:
                A dict with the set of node_ids actually contributing parameters,
                and backend specific fields:

        - `"backend": Backend,`
        - `"contributing_nodes": set[str],`
        - `"parameters": list[torch.nn.Parameter],  # PyTorch only`
        - `"variables": list[tf.Variable],  # TensorFlow only`

        """
        if self._optimizer is None:
            raise ValueError("No global optimizer exists for the graph.")

        # Select candidate nodes
        node_ids = (
            set(nodes_to_include)
            if nodes_to_include is not None
            else set(self._nodes_req_opt.keys())
        )
        selected_nodes: list[ModelNode] = []
        for nid in node_ids:
            node = self._nodes.get(nid)
            if node is None:
                continue
            if not isinstance(node, ModelNode):
                continue
            if include_only_unfrozen and node.is_frozen:
                continue
            selected_nodes.append(node)
        used_node_ids = {n.node_id for n in selected_nodes}

        # Collect backend-specific trainables
        backend = self._optimizer.backend
        if backend == Backend.TORCH:
            parameters = []
            for node in selected_nodes:
                if not hasattr(node.model, "parameters"):
                    msg = f"ModelNode '{node.label}' does not expose .parameters()"
                    raise AttributeError(msg)
                parameters.extend(list(node.model.parameters()))
            return {
                "backend": Backend.TORCH,
                "parameters": parameters,
                "contributing_nodes": used_node_ids,
            }

        if backend == Backend.TENSORFLOW:
            variables = []
            for node in selected_nodes:
                if not hasattr(node.model, "trainable_variables"):
                    msg = (
                        f"ModelNode '{node.label}' does not expose trainable_variables"
                    )
                    raise AttributeError(msg)
                variables.extend(list(node.model.trainable_variables))
            return {
                "backend": Backend.TENSORFLOW,
                "variables": variables,
                "contributing_nodes": used_node_ids,
            }

        if backend == Backend.SCIKIT:
            raise NotImplementedError("Scikit optimizers are not supported.")

        raise BackendNotSupportedError(
            backend=backend,
            message="Unknown backend for optimizer parameter collection.",
        )

    def _build_optimizer(
        self,
        nodes_to_include: list[str] | None = None,
        *,
        include_only_unfrozen: bool = True,
        force: bool = False,
    ):
        """
        Builds the global optimizer with parameters from the specified nodes.

        Args:
            nodes_to_include (list[str] | None):
                A list of node IDs to consider for parameter extraction.
                If None, all nodes in this graph are considered.

            include_only_unfrozen (bool, optional):
                If True, only nodes in `nodes_to_include` that are not frozen
                will be used for parameter extraction. If False, all nodes in
                `nodes_to_include` are used (i.e., ignores any frozen state).

            force (bool, optional):
                If False, the optimizer will only be rebuilt if the node
                parameters the optimizer relies on, has changed. Otherwise,
                the optimizer will be forcefully rebuilt.

        """
        if self._optimizer is None:
            msg = "No global optimizer exists for the graph."
            raise ValueError(msg)

        # Get info needed to build optimizer
        info = self._select_optimizer_parameters(
            nodes_to_include=nodes_to_include,
            include_only_unfrozen=include_only_unfrozen,
        )
        new_node_ids = info["contributing_nodes"]

        # Rebuild only if contributing nodes changed
        if (self._opt_built_from_node_ids == new_node_ids) and not force:
            return

        # Build optimizer
        if info["backend"] == Backend.TORCH:
            self._optimizer.build(
                parameters=info["parameters"],
                backend=Backend.TORCH,
                force_rebuild=True,
            )
        elif info["backend"] == Backend.TENSORFLOW:
            self._optimizer.build(
                backend=Backend.TENSORFLOW,
                force_rebuild=True,
            )
        elif self._optimizer.backend == Backend.SCIKIT:
            msg = "Scikit optimizers are not supported yet."
            raise NotImplementedError(msg)
        else:
            raise BackendNotSupportedError(
                backend=self._optimizer.backend,
                message="Unknown backend for optimizer building.",
            )

        # Update tracking on which nodes were used to build optimizer
        self._opt_built_from_node_ids = set(new_node_ids)
        self._optimizer_state = info

    def get_optimizer_parameters(self) -> dict[str, Any]:
        """
        State of current global optimizer (if defined).

        Returns a dict with the set of node_ids actually contributing parameters,
        and backend specific fields:

        - `"backend": Backend,`
        - `"contributing_nodes": set[str],`
        - `"parameters": list[torch.nn.Parameter],  # PyTorch only`
        - `"variables": list[tf.Variable],  # TensorFlow only`

        """
        if self._optimizer is None:
            raise ValueError("No global optimizer exists.")

        if self._optimizer_state is None:
            msg = (
                "Optimizer has not been built yet. "
                "Call train_step() or build_optimizer() first."
            )
            raise RuntimeError(msg)

        return self._optimizer_state

    def build(self, *, force: bool = False):
        """
        Build the ModelGraph by initializing all underlying models and optimizers.

        Args:
            force (bool, optional):
                If the graph is already built it will not be rebuilt unless
                `force=True`. Defaults to False.

        """
        # Skip if already built
        if self.is_built and not force:
            return

        # Revalidate all connections
        self._rebuild_connections()

        # Ensure all nodes are built
        node_out_shapes: dict[str, tuple[int, ...]] = {}
        for node_id in self._sorted_node_ids:
            node = self._nodes[node_id]

            # Check if node can and needs to be built
            if not isinstance(node, ComputeNode) or (node.is_built and not force):
                continue

            # Infer input shapes for this node
            in_shapes: dict[ExperimentNodeReference, tuple[int, ...]] | None = None
            dummy_inputs: dict[ExperimentNodeReference, SampleData] = {}
            for ups_ref in node.get_upstream_refs():
                if in_shapes is None:
                    in_shapes = {}

                # If upstream is a FeatureSet, use feature/target data for shape inference
                if isinstance(ups_ref, FeatureSetReference):
                    # Get feature and target shapes (drops leading dim of n_samples)
                    fsv = ups_ref.resolve()
                    f_shape = fsv.get_features(fmt=DataFormat.NUMPY).shape[1:]

                    # Record input shape for this ref
                    in_shapes[ups_ref] = tuple(f_shape)

                # Otherwise, we must use dummy output data from a prior node
                else:
                    # Use tracked output shape of upstream node
                    if ups_ref.node_id not in node_out_shapes:
                        msg = f"Input shape could not be inferred for node '{node.label}'."
                        raise RuntimeError(msg)
                    in_shapes[ups_ref] = node_out_shapes[ups_ref.node_id]

                # Create a dummy input for later use
                dummy_inputs[ups_ref] = make_dummy_sample_data(
                    batch_size=4,
                    feature_shape=in_shapes[ups_ref],
                )

            # Infer output shape
            out_shape: tuple[int, ...] | None = None
            if node_id in self.tail_nodes:
                # If this is a tail node, and is downstream of only one FeatureSet, we
                # can infer the output shape to be the FeatureSet.targets shape
                ups_fs_refs = find_upstream_featuresets(node=node)
                ups_fs_ids = {ref.node_id for ref in ups_fs_refs}
                if len(ups_fs_ids) == 1:
                    fsv = ups_fs_refs[0].resolve()
                    t_shape = fsv.get_targets(fmt=DataFormat.NUMPY).shape[1:]
                    out_shape = tuple(t_shape)

            # Build ComputeNode
            node.build(
                input_shapes=in_shapes,
                output_shape=out_shape,
                force=force,
            )

            # Perform dummy pass if output wasn't inferred
            if out_shape is None:
                sd_out: SampleData = node.forward(dummy_inputs)
                node_out_shapes[node_id] = tuple(sd_out.shapes.features_shape[1:])
            else:
                node_out_shapes[node_id] = out_shape

        # Build/rebuild optimizer
        if self._optimizer is not None:
            self._build_optimizer(force=force)

        # Update flag
        self._built = True

    # ================================================
    # Forward / Calling
    # ================================================
    def forward(
        self,
        inputs: dict[tuple[str, FeatureSetReference], TForward],
        *,
        active_nodes: list[str | GraphNode] | None = None,
    ) -> dict[str, Batch]:
        """
        Execute a forward pass through the ModelGraph.

        Args:
            inputs (dict[tuple[str, FeatureSetReference], TForward]):
                Mapping of (head_node_id, upstream_featureset_ref) -> TForward.
                Keys must correspond to head nodes in this graph (nodes whose upstream
                refs are FeatureSetReferences). A head node may have multiple inputs
                if it consumes multiple FeatureSets.

            active_nodes (list[str | GraphNode] | None, optional):
                Optional subset of nodes to execute. If provided, only these nodes (and
                any required upstream dependencies within this graph) are executed. If
                None, all nodes in the graph are executed.

        Returns:
            dict[str, TForward]:
                Mapping of node_id -> output for every executed node (typically all
                nodes, but may be restricted by `active_nodes`). Tail-node outputs
                can be obtained by filtering this dict with `self.tail_nodes`.

        """
        if not self.is_built:
            raise RuntimeError("ModelGraph must be built before calling forward().")

        # Resolve active nodes (and all upstream dependencies)
        if active_nodes is None:
            active_node_ids: set[str] = set(self._nodes.keys())
        else:
            active_node_ids = get_subgraph_nodes(
                graph=self,
                roots=active_nodes,
                direction="upstream",
                include_roots=True,
            )

        # Maintain execution order
        exec_order: list[str] = [
            nid for nid in self._sorted_node_ids if nid in active_node_ids
        ]

        # Compute outputs for all active node
        outputs: dict[str, TForward] = {}
        for n_id in exec_order:
            node = self.nodes[n_id]
            if not isinstance(node, Forwardable):
                continue

            # Gather inputs for this node
            inp_data = node.get_input_data(
                inputs=inputs,
                outputs=outputs,
                fmt=get_data_format_for_backend(node.backend),
            )
            # Forward pass & record outputs
            out_batch = node.forward(inp_data)
            outputs[n_id] = out_batch

        return outputs

    __call__ = forward

    # ================================================
    # Trainable Protocol
    # ================================================
    @property
    def backend(self) -> Backend | None:
        """
        The shared backend of this ModelGraph.

        Description:
            A ModelGraph's backend is only defined if a global optimizer
            if used. If the graph consists of mixed-backend nodes, None
            is returned.

        Returns:
            Backend | None:
                The backend of the global optimizer, if defined. Otherwise,
                returns None.

        """
        if self._optimizer is not None:
            return self._optimizer.backend
        return None

    def freeze(self, nodes: list[str, GraphNode] | None = None):
        """
        Sets the trainable state of `nodes` to frozen.

        Args:
            nodes (list[str, GraphNode] | None):
                A list of node IDs, labels, or instances. All specified nodes will
                have their internal state set to frozen, preventing training mutation.
                If None, all nodes in this graph will be frozen.

        """
        # Normalize node values
        if nodes is None:
            nodes = [n for n in self.nodes.values() if isinstance(n, Trainable)]
        else:
            if isinstance(nodes, (str, GraphNode)) or not isinstance(nodes, Iterable):
                nodes = [nodes]
            nodes: list[GraphNode] = [self._resolve_existing(n) for n in nodes]

        # Freeze all nodes
        for n in nodes:
            if not isinstance(n, Trainable):
                msg = f"GraphNode '{n.label}' is not Trainable. It cannot be frozen."
                raise TypeError(msg)
            n.freeze()

    def unfreeze(self, nodes: list[str, GraphNode] | None = None):
        """
        Sets the trainable state of `nodes` to unfrozen.

        Args:
            nodes (list[str, GraphNode] | None):
                A list of node IDs, labels, or instances. All specified nodes will
                have their internal state set to unfrozen, allowing training.
                If None, all nodes in this graph will be unfrozen.

        """
        # Normalize node values
        if nodes is None:
            nodes = [n for n in self.nodes.values() if isinstance(n, Trainable)]
        else:
            if isinstance(nodes, (str, GraphNode)) or not isinstance(nodes, Iterable):
                nodes = [nodes]
            nodes: list[GraphNode] = [self._resolve_existing(n) for n in nodes]

        # Unfreeze all nodes
        for n in nodes:
            if not isinstance(n, Trainable):
                msg = f"GraphNode '{n.label}' is not Trainable. It cannot be frozen."
                raise TypeError(msg)
            n.unfreeze()

    def _train_step_torch(
        self,
        ctx: ExecutionContext,
        losses: list[AppliedLoss],
        *,
        active_nodes: list[str | GraphNode] | None = None,
    ):
        """
        Graph-wise training with a PyTorch global optimizer.

        Args:
            ctx (ExecutionContext):
                Execution context containing inputs, outputs, and loss storage.

            losses (list[AppliedLoss]):
                All losses defined for this phase. Losses are filtered internally
                to the nodes they apply to.

            active_nodes (list[str | GraphNode] | None, optional):
                Optional subset of nodes to train. If provided, all upstream
                dependencies are included automatically.

        """
        # Reset optimizer gradients & get optimizer state info
        self._optimizer.zero_grad()

        # Forward pass & update ctx records
        outputs = self.forward(inputs=ctx.inputs, active_nodes=active_nodes)
        for n_id, batch in outputs.items():
            ctx.set_output(node_id=n_id, batch=batch)

        # Compute losses
        lrs_by_node: dict[str, list[LossRecord]] = defaultdict(list)
        for loss in losses:
            weighted_raw_loss = loss.compute(ctx=ctx)
            lr = LossRecord(
                value=weighted_raw_loss,
                label=loss.label,
                contributes_to_update=True,
            )
            lrs_by_node[loss.node_id].append(lr)
        lcs_by_node: dict[str, LossCollection] = {
            k: LossCollection(records=v) for k, v in lrs_by_node.items()
        }

        # Optimizer stepping using all trainable losses
        total_trainable = sum(lc.trainable for lc in lcs_by_node.values())
        total_trainable.backward()
        self._optimizer.step()

        # Update ctx records (group losses by node_id for logging)
        for n_id, lc in lcs_by_node.items():
            ctx.set_losses(node_id=n_id, loss=lc)

    def _train_step_tensorflow(
        self,
        ctx: ExecutionContext,
        losses: list[AppliedLoss],
        *,
        active_nodes: list[str | GraphNode] | None = None,
    ):
        """
        Graph-wise training with a TensorFlow global optimizer.

        Args:
            ctx (ExecutionContext):
                Execution context containing inputs, outputs, and loss storage.

            losses (list[AppliedLoss]):
                All losses defined for this phase. Losses are filtered internally
                to the nodes they apply to.

            active_nodes (list[str | GraphNode] | None, optional):
                Optional subset of nodes to train. If provided, all upstream
                dependencies are included automatically.

        """
        # Reset optimizer gradients & get optimizer state info
        self._optimizer.zero_grad()
        opt_info = self.get_optimizer_parameters()

        # Forward pass & update ctx records
        with tf.GradientTape() as tape:
            outputs = self.forward(inputs=ctx.inputs, active_nodes=active_nodes)
            for n_id, batch in outputs.items():
                ctx.set_output(node_id=n_id, batch=batch)

        # Compute losses
        lrs_by_node: dict[str, list[LossRecord]] = defaultdict(list)
        for loss in losses:
            weighted_raw_loss = loss.compute(ctx=ctx)
            lr = LossRecord(
                value=weighted_raw_loss,
                label=loss.label,
                contributes_to_update=True,
            )
            lrs_by_node[loss.node_id].append(lr)
        lcs_by_node: dict[str, LossCollection] = {
            k: LossCollection(records=v) for k, v in lrs_by_node.items()
        }

        # Optimizer stepping using all trainable losses
        total_trainable = sum(lc.trainable for lc in lcs_by_node.values())
        grads = tape.gradient(total_trainable, opt_info["variables"])
        self._optimizer.step(grads=grads, variables=opt_info["variables"])

        # Update ctx records (group losses by node_id for logging)
        for n_id, lc in lcs_by_node.items():
            ctx.set_losses(node_id=n_id, loss=lc)

    def _train_step_scikit(
        self,
        ctx: ExecutionContext,
        losses: list[AppliedLoss],
        *,
        active_nodes: list[str | GraphNode] | None = None,
    ):
        """
        Graph-wise training with a SciKit global optimizer.

        Args:
            ctx (ExecutionContext):
                Execution context containing inputs, outputs, and loss storage.

            losses (list[AppliedLoss]):
                All losses defined for this phase. Losses are filtered internally
                to the nodes they apply to.

            active_nodes (list[str | GraphNode] | None, optional):
                Optional subset of nodes to train. If provided, all upstream
                dependencies are included automatically.

        """
        # TODO: not implemented yet
        msg = "Training with a scikit global optimizer not implemented yet."
        raise NotImplementedError(msg)

    def train_step(
        self,
        ctx: ExecutionContext,
        losses: list[AppliedLoss],
        *,
        active_nodes: list[str | GraphNode] | None = None,
    ):
        """
        Execute a single training step for the ModelGraph.

        Behavior depends on whether a global optimizer is attached:

        - If `self._optimizer is None`:
            Stage-wise training is performed. Each ModelNode executes its own
            `train_step()` in topological order using its local optimizer.

        - If `self._optimizer is not None`:
            Graph-wise training is performed. A single forward pass is executed
            across the graph, all losses are computed, a single backward pass
            is performed, and the global optimizer is stepped once.

        Args:
            ctx (ExecutionContext):
                Execution context containing inputs, outputs, and loss storage.

            losses (list[AppliedLoss]):
                All losses defined for this phase. Losses are filtered internally
                to the nodes they apply to.

            active_nodes (list[str | GraphNode] | None, optional):
                Optional subset of nodes to be excuted in the forward pass. If provided,
                all upstream dependencies are included automatically. Otherwise, all
                nodes in the graph are executed.

                Note that this does not set the trainable state of the nodes, only on
                which nodes a forward pass is called. Use `freeze()` and `unfreeze`
                to set the trainable state of graph nodes.

        """
        # Ensure graph is built
        if not self.is_built:
            self.build()

        # Resolve active nodes (and all upstream dependencies)
        if active_nodes is None:
            active_node_ids: set[str] = set(self._nodes.keys())
        else:
            active_node_ids = get_subgraph_nodes(
                graph=self,
                roots=active_nodes,
                direction="upstream",
                include_roots=True,
            )

        # Maintain execution order
        exec_order: list[str] = [
            nid for nid in self._sorted_node_ids if nid in active_node_ids
        ]

        # Validate that at least one loss is applied to these active nodes
        valid = False
        for loss in losses:
            if loss.node_id in active_node_ids:
                valid = True
                break
        if not valid:
            msg = "Training must have at least one loss applied to an active node."
            raise ValueError(msg)

        # Validate not all frozen
        valid = False
        for n_id in active_node_ids:
            node = self.nodes[n_id]
            if isinstance(node, Trainable) and not node.is_frozen:
                valid = True
                break
        if not valid:
            msg = "Training must have at least unfrozen node."
            raise ValueError(msg)

        # ------------------------------------------------
        # Training Case 1: Stage-wise training (no global optimizer)
        # ------------------------------------------------
        if self._optimizer is None:
            for node_id in exec_order:
                node = self._nodes[node_id]

                # If trainable, use train_step (check if frozen)
                if isinstance(node, Trainable) and not node.is_frozen:
                    node.train_step(ctx=ctx, losses=losses)

                # If evaluable (or trainable + frozen), use eval_step
                elif isinstance(node, Evaluable):
                    node.eval_step(ctx=ctx, losses=losses)

                # If forwardable, record outputs of manual forward pass
                elif isinstance(node, Forwardable):
                    # Gather inputs for this node
                    inp_data = node.get_input_data(
                        inputs=ctx.inputs,
                        outputs=ctx.outputs,
                        fmt=get_data_format_for_backend(node.backend),
                    )
                    # Forward pass & record outputs
                    ctx.outputs[node_id] = node.forward(inp_data)

                # Otherwise, skip node
            return None

        # ------------------------------------------------
        # Training Case 2: Graph-wise training (global optimizer)
        # ------------------------------------------------
        # Rebuild optimizer with only unfrozen nodes (only rebuilds if necessary)
        self._build_optimizer(
            nodes_to_include=active_node_ids,
            include_only_unfrozen=True,
        )

        # Use backend-specific training logic
        backend = self._optimizer.backend
        if backend == Backend.TORCH:
            return self._train_step_torch(
                ctx=ctx,
                losses=losses,
                active_nodes=active_node_ids,
            )

        if backend == Backend.TENSORFLOW:
            return self._train_step_tensorflow(
                ctx=ctx,
                losses=losses,
                active_nodes=active_node_ids,
            )

        if backend == Backend.SCIKIT:
            return self._train_step_scikit(
                ctx=ctx,
                losses=losses,
                active_nodes=active_node_ids,
            )

        msg = f"Unknown backend: {backend}"
        raise BackendNotSupportedError(msg)

    # ================================================
    # Evaluable Protocol
    # ================================================
    def eval_step(
        self,
        ctx: ExecutionContext,
        losses: list[AppliedLoss],
        *,
        active_nodes: list[str | GraphNode] | None = None,
    ):
        """
        Execute a single evaluation step for the ModelGraph.

        Description:
            Performs a forward-only pass through the graph, computes all applicable
            losses, and records outputs and losses into the ExecutionContext.
            No gradients are tracked and no optimizers are stepped.

        Args:
            ctx (ExecutionContext):
                Execution context containing inputs, outputs, and loss storage.

            losses (list[AppliedLoss]):
                Losses to compute during evaluation.

            active_nodes (list[str | GraphNode] | None, optional):
                Optional subset of nodes to execute. All required upstream
                dependencies are included automatically.

        """
        # Ensure graph is built
        if not self.is_built:
            self.build()

        # Ensure all nodes frozen
        self.freeze(nodes=None)

        # Forward Pass + No Gradients
        backend = self.backend
        if backend == Backend.TORCH:
            with torch.no_grad():
                outputs = self.forward(
                    inputs=ctx.inputs,
                    active_nodes=active_nodes,
                )
        else:
            outputs = self.forward(
                inputs=ctx.inputs,
                active_nodes=active_nodes,
            )

        # Record outputs
        for n_id, batch in outputs.items():
            ctx.set_output(node_id=n_id, batch=batch)

        # Compute losses
        lrs_by_node: dict[str, list[LossRecord]] = defaultdict(list)
        for loss in losses:
            weighted_raw_loss = loss.compute(ctx=ctx)
            lr = LossRecord(
                value=weighted_raw_loss,
                label=loss.label,
                contributes_to_update=False,
            )
            lrs_by_node[loss.node_id].append(lr)
        lcs_by_node: dict[str, LossCollection] = {
            k: LossCollection(records=v) for k, v in lrs_by_node.items()
        }

        # Update ctx records
        for n_id, lc in lcs_by_node.items():
            ctx.set_losses(node_id=n_id, loss=lc)

    # ================================================
    # Configurable
    # ================================================
    def get_config(self) -> dict[str, Any]:
        """
        Retrieve the configuration details of this ModelGraph instance.

        This does not contain state information of any underlying models or optimizers.
        """
        return {
            "label": self.label,
            "nodes": [self.nodes[n_id].get_config() for n_id in self._sorted_node_ids],
            "optimizer": None
            if self._optimizer is None
            else self._optimizer.get_config(),
        }

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        *,
        register: bool = True,
    ) -> ModelGraph:
        """
        Reconstructs a ModelGraph from configuration details.

        This does not restore state information of any underlying models or optimizers.
        """
        ctx = ExperimentContext.get_active()

        # Rebuild nodes first (must register them to use a ModelGraph)
        nodes: list[GraphNode] = []
        for node_cfg in config["nodes"]:
            node = GraphNode.from_config(config=node_cfg, register=register)
            nodes.append(node)

        # Rebuild optimizer
        optimizer = None
        optimizer_cfg = config.get("optimizer")
        if optimizer_cfg is not None:
            optimizer = Optimizer.from_config(optimizer_cfg)

        # Create ModelGraph
        mg = cls(
            nodes=nodes,
            optimizer=optimizer,
            label=config.get("label", "model-graph"),
            ctx=ctx,
            register=register,
        )

        return mg

    # ================================================
    # Stateful
    # ================================================
    def get_state(self) -> dict[str, Any]:
        state = {
            "nodes": {
                n_id: self.nodes[n_id].get_state()
                for n_id in self._sorted_node_ids
                if isinstance(self.nodes[n_id], Stateful)
            },
            "optimizer": None
            if self._optimizer is None
            else self._optimizer.get_state(),
            "opt_built_from_node_ids": self._opt_built_from_node_ids,
            "is_built": self.is_built,
        }
        return state

    def set_state(self, state: dict[str, Any]) -> None:
        # Restore node state
        for n_id, n_state in state.get("nodes", {}).items():
            node = self._nodes[n_id]
            if isinstance(node, Stateful):
                node.set_state(n_state)

        # Restore optimizer
        if self._optimizer is not None and state.get("optimizer") is not None:
            self._optimizer.set_state(state["optimizer"])
        opt_nodes = state.get("opt_built_from_node_ids")
        self._opt_built_from_node_ids = None if opt_nodes is None else set(opt_nodes)

        if state.get("is_built", False):
            self._build_optimizer(
                self._opt_built_from_node_ids,
                include_only_unfrozen=False,
                force=True,
            )
            self._built = True

    # ================================================
    # Serialization
    # ================================================
    def save(self, filepath: Path, *, overwrite: bool = False) -> Path:
        """
        Serializes this ModelGraph to the specified filepath.

        Args:
            filepath (Path):
                File location to save to. Note that the suffix may be overwritten
                to enforce the ModularML file extension schema.
            overwrite (bool, optional):
                Whether to overwrite any existing file at the save location.
                Defaults to False.

        Returns:
            Path: The actual filepath to write the ModelGraph is saved.

        """
        from modularml.core.io.serialization_policy import SerializationPolicy
        from modularml.core.io.serializer import serializer

        return serializer.save(
            self,
            filepath,
            policy=SerializationPolicy.BUILTIN,
            overwrite=overwrite,
        )

    @classmethod
    def load(
        cls,
        filepath: Path,
        *,
        allow_packaged_code: bool = False,
        overwrite: bool = False,
    ) -> ModelGraph:
        """
        Load a FeaturModelGrapheSet from file.

        Args:
            filepath (Path):
                File location of a previously saved ModelGraph.
            allow_packaged_code : bool
                Whether bundled code execution is allowed.
            overwrite (bool):
                Whether to replace any colliding node registrations in ExperimentContext
                If False, new IDs are assigned to the reloaded nodes comprising the
                graph. Otherwise, any collision are overwritten with the saved nodes.
                Defaults to False.
                It is recommended to only reload a ModelGraph into a new/empty
                `ExperimentContext`.

        Returns:
            ModelGraph: The reloaded ModelGraph.

        """
        from modularml.core.io.serializer import _enforce_file_suffix, serializer

        # Append proper sufficx only if no suffix is given
        if Path(filepath).suffix == "":
            filepath = _enforce_file_suffix(path=filepath, cls=cls)

        return serializer.load(
            filepath,
            allow_packaged_code=allow_packaged_code,
            overwrite=overwrite,
        )

    # ================================================
    # Checkpointing
    # ================================================
    def save_checkpoint(
        self,
        filepath: Path,
        *,
        overwrite: bool = False,
    ) -> Path:
        """
        Save full ModelGraph state.

        Args:
            filepath (Path):
                File location to save to. Note that the suffix may be overwritten
                to enforce the ModularML file extension schema.
            overwrite (bool, optional):
                Whether to overwrite any existing file at the save location.
                Defaults to False.

        Returns:
            Path: Final path of saved ModelGraph checkpoint.

        """
        from modularml.core.io.serialization_policy import SerializationPolicy
        from modularml.core.io.serializer import serializer

        ckpt = Checkpoint()
        ckpt.add_entry(key="modelgraph", obj=self)
        for n_id, node in self.nodes.items():
            ckpt.add_entry(key=f"nodes:{n_id}", obj=node)
        if self._optimizer is not None:
            ckpt.add_entry(key="optimizer", obj=self._optimizer)

        return serializer.save(
            ckpt,
            filepath,
            policy=SerializationPolicy.BUILTIN,
            overwrite=overwrite,
        )

    def load_checkpoint(self, filepath: Path) -> ModelGraph:
        """
        Restore ModelGraph state from checkpoint.

        Args:
            filepath (Path):
                File location of a previously saved ModelGraph checkpoint.

        Returns:
            self: The ModelGraph restored to the checkpoint state.

        """
        from modularml.core.io.serializer import _enforce_file_suffix, serializer

        # Append proper suffix only if no suffix is given
        if Path(filepath).suffix == "":
            filepath = _enforce_file_suffix(path=filepath, cls=Checkpoint)

        # Load checkpoint
        ckpt: Checkpoint = serializer.load(filepath)

        # Set node states
        n_states = {
            k.split(":")[-1]: v.entry_state
            for k, v in ckpt.entries.items()
            if k.startswith("nodes")
        }
        for n_id, n_state in n_states.items():
            self.nodes[n_id].set_state(n_state)

        # Set optimizer state
        if "optimizer" in ckpt.entries:
            self._optimizer.set_state(ckpt.entries["optimizer"].entry_state)

        # Update model graph state
        mg_state = ckpt.entries["modelgraph"].entry_state
        self._opt_built_from_node_ids = mg_state["opt_built_from_node_ids"]
        self._built = mg_state["is_built"]

        return self
