"""Graph traversal helpers for resolving dependencies within model graphs."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Literal

from modularml.core.experiment.experiment_context import ExperimentContext
from modularml.core.references.featureset_reference import FeatureSetReference
from modularml.core.topology.graph_node import GraphNode
from modularml.utils.errors.error_handling import ErrorMode

if TYPE_CHECKING:
    from modularml.core.topology.graph_node import GraphNode
    from modularml.core.topology.model_graph import ModelGraph

TraversalDirection = Literal["upstream", "downstream", "both"]


def is_tail_node(node: GraphNode):
    """
    Return True if the :class:`GraphNode` has no downstream consumers.

    Args:
        node (GraphNode): :class:`GraphNode` to inspect.

    Returns:
        bool: True if downstream connections are disallowed or absent.

    """
    return (not node.allows_downstream_connections) or (
        len(node.get_downstream_refs()) == 0
    )


def is_head_node(node: GraphNode):
    """
    Return True if the :class:`GraphNode` accepts FeatureSet inputs (head node).

    Args:
        node (GraphNode): :class:`GraphNode` to inspect.

    Returns:
        bool: True if the node allows upstream FeatureSet connections.

    """
    if not node.allows_upstream_connections:
        return False
    ups = node.get_upstream_refs()
    return ups and all(isinstance(r, FeatureSetReference) for r in ups)


def find_upstream_featuresets(
    node: str | GraphNode,
) -> list[FeatureSetReference]:
    """
    Recursively find all upstream :class:`FeatureSetReference` objects feeding into `node`.

    Description:
        Traverses the graph upstream until :class:`FeatureSetReference` instances are reached.

    Args:
        node (str | GraphNode):
            The node ID, label, or :class:`GraphNode` instance to perform a search for.

    Returns:
        list[FeatureSetReference]:
            All unique upstream FeatureSets (order not guaranteed).

    Raises:
        TypeError: If `node` is neither a string nor a :class:`GraphNode`.

    """
    from modularml.core.topology.graph_node import GraphNode

    graph_node = None
    if isinstance(node, GraphNode):
        graph_node = node
    elif isinstance(node, str):
        exp_ctx = ExperimentContext.get_active()
        graph_node = exp_ctx.get_node(val=node, enforce_type="GraphNode")
    else:
        msg = (
            "Invalid `node` type. Expected string or GraphNode. "
            f"Received: {type(node)}."
        )
        raise TypeError(msg)

    found: dict[str, FeatureSetReference] = {}
    visited: set[str] = set()

    def _walk(n: GraphNode):
        if n.node_id in visited:
            return
        visited.add(n.node_id)

        for ref in n.get_upstream_refs():
            if isinstance(ref, FeatureSetReference):
                found[ref.node_id] = ref
            else:
                upstream_node = ref.resolve()
                _walk(upstream_node)

    _walk(graph_node)
    return list(found.values())


def get_subgraph_nodes(
    graph: ModelGraph,
    roots: str | GraphNode | Iterable[str | GraphNode],
    *,
    direction: TraversalDirection = "upstream",
    include_roots: bool = True,
) -> set[str]:
    """
    Collect a set of :class:`GraphNode` IDs reachable from one or more root nodes.

    Description:
        This utility traverses the ModelGraph starting from one or more root
        nodes and returns the set of GraphNodes reachable in the specified
        direction:

            - "upstream": nodes required to compute the root(s)
            - "downstream": nodes that depend on the root(s)
            - "both": union of upstream and downstream traversal

        Traversal is structural only. FeatureSet inputs are treated as external
        data sources and are not included in the result.

    Args:
        graph (ModelGraph):
            The :class:`ModelGraph` containing the connected :class:`GraphNode` objects.

        roots (str | GraphNode | Iterable[str | GraphNode]):
            One or more root nodes from which traversal begins.
            Each value may be a node ID, node label, or :class:`GraphNode` instance.

        direction ("upstream" | "downstream" | "both", optional):
            Direction of traversal relative to the root nodes.
            Defaults to "upstream".

        include_roots (bool, optional):
            Whether to include the root nodes themselves in the returned set.
            Defaults to True.

    Returns:
        set[str]:
            A set of :class:`GraphNode` IDs reachable under the specified traversal.

    Raises:
        ValueError: If `direction` is not one of the supported values.

    Notes:
        - Returned IDs always refer to nodes present in `graph.nodes`.
        - :class:`FeatureSetReference` instances are excluded from traversal results.
        - Order is not guaranteed; this is a dependency set, not an execution order.
        - For execution ordering, use `ModelGraph._sorted_node_ids`.

    """
    from modularml.core.topology.graph_node import GraphNode

    if direction not in {"upstream", "downstream", "both"}:
        msg = f"Invalid traversal direction: {direction}"
        raise ValueError(msg)

    # Normalize roots
    if isinstance(roots, (str, GraphNode)) or not isinstance(roots, Iterable):
        roots = [roots]
    root_nodes: list[GraphNode] = [graph._resolve_existing(r) for r in roots]

    visited: set[str] = set()
    result: set[str] = set()

    def visit_upstream(node: GraphNode):
        if node.node_id in visited:
            return
        visited.add(node.node_id)
        result.add(node.node_id)

        for ref in node.get_upstream_refs(error_mode=ErrorMode.IGNORE):
            if isinstance(ref, FeatureSetReference):
                continue
            if ref.node_id in graph.nodes:
                visit_upstream(graph.nodes[ref.node_id])

    def visit_downstream(node: GraphNode):
        if node.node_id in visited:
            return
        visited.add(node.node_id)
        result.add(node.node_id)

        for ref in node.get_downstream_refs(error_mode=ErrorMode.IGNORE):
            if ref.node_id in graph.nodes:
                visit_downstream(graph.nodes[ref.node_id])

    # Traverse
    for root in root_nodes:
        if direction in {"upstream", "both"}:
            visit_upstream(root)
        if direction in {"downstream", "both"}:
            visit_downstream(root)

    # Remove root nodes (if requested)
    if not include_roots:
        for root in root_nodes:
            result.discard(root.node_id)

    return result
