"""Serialization helpers for ModularML reference objects."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from modularml.core.references.featureset_reference import (
    FeatureSetReference,
)

if TYPE_CHECKING:
    from modularml.core.data.featureset import FeatureSet


def ref_to_yaml_dict(ref: Any) -> dict[str, Any]:
    """
    Serialize any supported reference object to a YAML-friendly dict.

    The dict is tagged with ``ref_type`` so that :func:`ref_from_yaml_dict`
    can reconstruct the exact reference type on import.

    Supported types:

    * :class:`FeatureSetReference` — preserves ``node_label``, ``features``,
      ``targets``, and ``tags`` column selectors.
    * :class:`ExperimentNodeReference` (graph-node-to-graph-node) — preserved
      as a ``"GraphNodeReference"`` with ``node_label`` / ``node_id``.

    Args:
        ref: Reference object to serialize.

    Returns:
        dict[str, Any]: YAML-friendly dict with a ``ref_type`` key.

    Raises:
        TypeError: If the reference type is not supported.

    """
    from modularml.core.references.experiment_reference import ExperimentNodeReference

    if isinstance(ref, FeatureSetReference):
        d: dict[str, Any] = {"ref_type": "FeatureSetReference"}
        if ref.node_label is not None:
            d["node_label"] = ref.node_label
        elif ref.node_id is not None:
            d["node_id"] = ref.node_id
        if ref.features is not None:
            d["features"] = list(ref.features)
        if ref.targets is not None:
            d["targets"] = list(ref.targets)
        if ref.tags is not None:
            d["tags"] = list(ref.tags)
        return d

    if isinstance(ref, ExperimentNodeReference):
        d = {"ref_type": "GraphNodeReference"}
        if ref.node_label is not None:
            d["node_label"] = ref.node_label
        elif ref.node_id is not None:
            d["node_id"] = ref.node_id
        return d

    msg = f"Unsupported reference type for YAML: {type(ref)!r}"
    raise TypeError(msg)


def ref_from_yaml_dict(
    d: dict[str, Any] | None,
    built_nodes: dict[str, Any] | None = None,
) -> Any:
    """
    Reconstruct a reference object from its YAML dict.

    Dispatches on ``ref_type``:

    * ``"FeatureSetReference"`` — builds a :class:`FeatureSetReference` with
      the stored column selectors.
    * ``"GraphNodeReference"`` — looks up the node by label in ``built_nodes``
      first, then falls back to the active :class:`ExperimentContext`.

    Args:
        d (dict[str, Any] | None): Dict produced by :func:`ref_to_yaml_dict`.
        built_nodes (dict[str, Any] | None): Graph nodes already constructed,
            keyed by label. Only needed for ``"GraphNodeReference"`` resolution.
            Defaults to an empty dict.

    Returns:
        FeatureSetReference | GraphNode: Resolved reference or node.

    Raises:
        ValueError: If ``d`` is ``None`` or the ``ref_type`` is unknown.

    """
    from modularml.core.experiment.experiment_context import ExperimentContext

    if d is None:
        msg = "Missing required upstream reference in YAML."
        raise ValueError(msg)

    if built_nodes is None:
        built_nodes = {}

    ref_type = d.get("ref_type")

    if ref_type == "FeatureSetReference":
        node_label = d.get("node_label")
        node_id = d.get("node_id")
        features_raw = d.get("features")
        targets_raw = d.get("targets")
        tags_raw = d.get("tags")

        ctx = ExperimentContext.get_active()
        try:
            fs: FeatureSet = ctx.get_node(
                val=node_id or node_label,
                enforce_type="FeatureSet",
            )
            return fs.reference(
                features=features_raw,
                targets=targets_raw,
                tags=tags_raw,
            )
        except Exception as e:
            msg = f"Cannot resolve FeatureSetReference: label={node_label!r}, id={node_id!r}"
            raise ValueError(msg) from e

    if ref_type == "GraphNodeReference":
        node_label = d.get("node_label")
        node_id = d.get("node_id")

        if node_label and node_label in built_nodes:
            return built_nodes[node_label]
        if node_id and node_id in built_nodes:
            return built_nodes[node_id]

        ctx = ExperimentContext.get_active()
        try:
            return ctx.get_node(val=node_id or node_label)
        except Exception as e:
            msg = f"Cannot resolve GraphNodeReference: label={node_label!r}, id={node_id!r}"
            raise ValueError(msg) from e

    msg = f"Unknown ref_type in YAML: {ref_type!r}"
    raise ValueError(msg)
