"""Translator between ModelGraph instances and YAML-friendly dicts."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from modularml.core.io.yaml.class_resolver import class_to_path, resolve_class
from modularml.core.io.yaml.translators.references import (
    ref_from_yaml_dict,
    ref_to_yaml_dict,
)
from modularml.core.io.yaml.utils import yaml_safe_cfg

if TYPE_CHECKING:
    from modularml.core.topology.graph_node import GraphNode
    from modularml.core.topology.model_graph import ModelGraph
    from modularml.core.training.optimizer import Optimizer


# ================================================
# Export
# ================================================
def model_graph_to_yaml_dict(graph: ModelGraph) -> dict[str, Any]:
    """
    Export a :class:`ModelGraph` to a YAML-friendly dictionary.

    The resulting dict uses human-readable node names and dotted class paths
    rather than internal UUIDs and class objects.

    Args:
        graph (ModelGraph): Graph instance to export.

    Returns:
        dict[str, Any]: Dict with a single ``model_graph`` top-level key.

    """
    nodes_yaml = []
    for node_id in graph._sorted_node_ids:
        node = graph.nodes[node_id]
        node_yaml = _node_to_yaml_dict(node)
        nodes_yaml.append(node_yaml)

    graph_dict: dict[str, Any] = {
        "graph_name": graph.label,
        "nodes": nodes_yaml,
    }

    # Include graph-level optimizer if present
    if graph._optimizer is not None:
        graph_dict["optimizer"] = _optimizer_to_dict(graph._optimizer)

    return {"model_graph": graph_dict}


def _node_to_yaml_dict(node: GraphNode) -> dict[str, Any]:
    """Convert a single GraphNode to a YAML-friendly dict."""
    from modularml.core.topology.merge_nodes.merge_node import MergeNode
    from modularml.core.topology.model_node import ModelNode

    node_dict: dict[str, Any] = {"node_name": node.label}

    if isinstance(node, ModelNode):
        node_dict["node_class"] = "model"
        model_cfg = node._model.get_config()

        # model_class is a class object in get_config() output
        model_cls = model_cfg.get("model_class")
        if model_cls is not None:
            if isinstance(model_cls, type):
                node_dict["model_class"] = class_to_path(model_cls)
            else:
                node_dict["model_class"] = str(model_cls)

        node_dict["model_kwargs"] = yaml_safe_cfg(dict(model_cfg.get("init_args", {})))

        # Include node-level optimizer if present
        if node._optimizer is not None:
            node_dict["optimizer"] = _optimizer_to_dict(node._optimizer)

        if node._freeze:
            node_dict["frozen"] = True

        if node._accelerator is not None:
            node_dict["accelerator"] = node._accelerator.device

        # Single upstream reference
        upstream_refs = node._upstream_refs
        if upstream_refs:
            node_dict["upstream"] = ref_to_yaml_dict(upstream_refs[0])

    elif isinstance(node, MergeNode):
        node_dict["node_class"] = "merge"
        node_dict["merge_node_class"] = class_to_path(type(node))

        # Subclass-specific kwargs via get_config(), excluding base/build-state keys
        _skip_keys = {
            "node_id",
            "label",
            "upstream_refs",
            "downstream_refs",
            "graph_node_type",
            "merge_node_type",
            "output_shape",
            "input_shapes",
            "is_built",
            "backend",
        }
        cfg = node.get_config()
        node_dict.update({k: v for k, v in cfg.items() if k not in _skip_keys})

        # All upstream refs as a list of dicts
        node_dict["upstream"] = [ref_to_yaml_dict(r) for r in node._upstream_refs]

    else:
        msg = f"Unsupported ExperimentNode for YAML conversion: {node!r}"
        raise TypeError(msg)

    return node_dict


def _optimizer_to_dict(optimizer: Optimizer) -> dict[str, Any]:
    """Serialize an Optimizer to a YAML-friendly dict."""
    cfg = optimizer.get_config()
    # Remove None values for cleaner YAML
    return {k: v for k, v in cfg.items() if v is not None}


def _optimizer_from_dict(d: dict[str, Any]) -> Optimizer:
    """Reconstruct an Optimizer from a YAML dict."""
    from modularml.core.training.optimizer import Optimizer

    return Optimizer.from_config(d)


# ================================================
# Import
# ================================================
def model_graph_from_yaml_dict(
    d: dict[str, Any],
    *,
    overwrite: bool = False,
) -> ModelGraph:
    """
    Reconstruct a :class:`ModelGraph` from a YAML-friendly dictionary.

    Nodes must be listed in topological order (upstream nodes first).
    Upstream references are resolved by label: first against already-built
    nodes in this call, then against the active :class:`ExperimentContext`
    (for FeatureSet references).

    Args:
        d (dict[str, Any]): Dict with a ``model_graph`` top-level key, as
            produced by :func:`model_graph_to_yaml_dict`, or the inner dict
            (without the top-level key).
        overwrite (bool, optional): When ``False`` (default), a
            :exc:`ValueError` is raised if any node label in ``d`` already
            exists in the active :class:`ExperimentContext`. When ``True``
            the existing registration is silently replaced.
            Defaults to False.

    Returns:
        ModelGraph: Reconstructed graph registered in the active context.

    Raises:
        ValueError: If ``overwrite`` is False and a node label conflict
            is detected.

    """
    from modularml.core.experiment.experiment_context import ExperimentContext
    from modularml.core.topology.model_graph import ModelGraph

    # Accept both {"model_graph": {...}} and the inner dict directly
    if "model_graph" in d:
        d = d["model_graph"]

    # Check for node label conflicts before constructing anything
    try:
        ctx = ExperimentContext.get_active()
    except RuntimeError:
        ctx = None

    if ctx is not None and not overwrite:
        conflicting = [
            cfg["node_name"]
            for cfg in d.get("nodes", [])
            if ctx.has_node(label=cfg["node_name"])
        ]
        if conflicting:
            msg = (
                f"Node label(s) {conflicting} already exist in the active "
                "ExperimentContext. Load into a fresh context or pass "
                "overwrite=True to replace them."
            )
            raise ValueError(msg)

    label = d.get("graph_name", "model-graph")
    nodes_cfg = d.get("nodes", [])

    # Reconstruct graph-level optimizer if present
    graph_optimizer = None
    if d.get("optimizer"):
        graph_optimizer = _optimizer_from_dict(d["optimizer"])

    built_nodes: dict[str, Any] = {}  # node_name → node instance
    nodes_list = []

    for node_cfg in nodes_cfg:
        node = _node_from_yaml_dict(node_cfg, built_nodes=built_nodes)
        built_nodes[node.label] = node
        nodes_list.append(node)

    return ModelGraph(nodes=nodes_list, optimizer=graph_optimizer, label=label)


def _node_from_yaml_dict(
    cfg: dict[str, Any],
    built_nodes: dict[str, Any],
) -> GraphNode:
    """Reconstruct a single graph node from its YAML dict."""
    from modularml.core.topology.model_node import ModelNode

    node_name = cfg["node_name"]
    node_class = cfg.get("node_class", "model")

    if node_class == "model":
        upstream_ref = ref_from_yaml_dict(cfg.get("upstream"), built_nodes=built_nodes)

        model_cls = resolve_class(cfg["model_class"])
        model = model_cls(**dict(cfg.get("model_kwargs") or {}))

        node_optimizer = None
        if cfg.get("optimizer"):
            node_optimizer = _optimizer_from_dict(cfg["optimizer"])

        node = ModelNode(
            label=node_name,
            model=model,
            upstream_ref=upstream_ref,
            optimizer=node_optimizer,
            accelerator=cfg.get("accelerator"),
        )
        if cfg.get("frozen", False):
            node.freeze()
        return node

    if node_class == "merge":
        from modularml.core.topology.merge_nodes.merge_node import MergeNode

        # upstream is a list of dicts for merge nodes
        upstream_refs = [
            ref_from_yaml_dict(u, built_nodes=built_nodes)
            for u in cfg.get("upstream", [])
        ]

        # Build a from_config-compatible dict
        node_cfg = dict(cfg)
        node_cfg["label"] = node_name
        node_cfg["upstream_refs"] = upstream_refs
        node_cfg["graph_node_type"] = "MergeNode"
        node_cfg["merge_node_type"] = cfg["merge_node_class"].rsplit(".", 1)[-1]

        return MergeNode.from_config(node_cfg)

    msg = f"Unsupported node_class '{node_class}' in YAML. Supported: 'model', 'merge'."
    raise ValueError(msg)
