"""Mermaid-based visualization for ModularML core objects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from modularml.utils.errors.error_handling import ErrorMode
from modularml.utils.topology.graph_search_utils import is_head_node, is_tail_node
from modularml.visualization.visualizer.styling import (
    APPLIED_LOSS,
    EDGE_ANIMATION_DASH_MEDIUM,
    EDGE_ANIMATION_NONE,
    FEATURE_SET,
    FEATURE_SET_VIEW,
    INACTIVE_NODE,
    MERGE_NODE,
    MODEL_NODE,
    MODEL_NODE_FROZEN,
    OUTPUT_TERMINAL,
    SAMPLER,
    EdgeAnimationSpec,
    EdgeConnectionSpec,
    FeatureSetDisplayOptions,
    ModelGraphDisplayOptions,
    NodeSpec,
)

if TYPE_CHECKING:
    from modularml.core.data.featureset import FeatureSet
    from modularml.core.experiment.phases.eval_phase import EvalPhase
    from modularml.core.experiment.phases.fit_phase import FitPhase
    from modularml.core.experiment.phases.phase import InputBinding
    from modularml.core.experiment.phases.train_phase import TrainPhase
    from modularml.core.topology.model_graph import ModelGraph
    from modularml.core.training.applied_loss import AppliedLoss


# ================================================
# Label formatting helpers
# ================================================
def _format_featureset_label(
    fs_label: str,
    n_samples: int | None = None,
    splits: list[str] | None = None,
) -> str:
    """Build rich HTML label for a FeatureSet node."""
    lines = ["<b>FeatureSet</b>", f"'{fs_label}'"]
    if n_samples is not None:
        lines.append(f"n={n_samples}")
    if splits:
        lines.append(f"splits: {', '.join(splits)}")
    return "<br>".join(lines)


def _format_model_node_label(
    node_label: str,
    backend: str | None = None,
    *,
    is_frozen: bool = False,
    upstream_features: tuple[str, ...] | None = None,
    upstream_targets: tuple[str, ...] | None = None,
    upstream_tags: tuple[str, ...] | None = None,
) -> str:
    """
    Build rich HTML label for a ModelNode.

    Args:
        node_label:
            Display name of the node.
        backend:
            Backend value string (e.g. ``"torch"``). Shown as ``<torch>``.
        is_frozen:
            Whether the node is frozen.
        upstream_features:
            Feature column selectors from FeatureSetReference (only for head nodes).
        upstream_targets:
            Target column selectors from FeatureSetReference (only for head nodes).
        upstream_tags:
            Tag column selectors from FeatureSetReference (only for head nodes).

    """
    header = "<b>ModelNode</b>"
    meta_parts: list[str] = []
    if backend is not None:
        meta_parts.append(f"&lt;{backend}&gt;")
    if is_frozen:
        meta_parts.append("frozen")
    meta_line = f"  {' · '.join(meta_parts)}" if meta_parts else ""
    lines = [header, f"'{node_label}'{meta_line}"]

    # Head-node column listing
    if upstream_features or upstream_targets or upstream_tags:
        lines.append("──────────")
        if upstream_features:
            lines.append("<b>features</b>")
            for col in upstream_features:
                lines.append(f"  {col}")
        if upstream_targets:
            lines.append("<b>targets</b>")
            for col in upstream_targets:
                lines.append(f"  {col}")
        if upstream_tags:
            lines.append("<b>tags</b>")
            for col in upstream_tags:
                lines.append(f"  {col}")

    return "<br>".join(lines)


def _format_merge_node_label(
    node_label: str,
    merge_type: str = "MergeNode",
) -> str:
    """Build rich HTML label for a MergeNode."""
    lines = [f"<b>{merge_type}</b>", f"'{node_label}'"]
    return "<br>".join(lines)


def _format_sampler_label(
    sampler_type: str,
    batch_size: int,
    split: str | None = None,
    stream: str | None = None,
) -> str:
    """Build rich HTML label for a Sampler node."""
    lines = ["<b>Sampler</b>", sampler_type]
    lines.append(f"batch_size: {batch_size}")
    if split is not None:
        lines.append(f"split: {split}")
    if stream is not None and stream != "default":
        lines.append(f"stream: {stream}")
    return "<br>".join(lines)


def _format_loss_label(
    loss_label: str,
    loss_name: str,
    weight: float = 1.0,
) -> str:
    """Build rich HTML label for an AppliedLoss node."""
    lines = ["<b>AppliedLoss</b>", f"'{loss_label}'"]
    lines.append(f"loss: {loss_name}")
    if weight != 1.0:
        lines.append(f"weight: {weight}")
    return "<br>".join(lines)


def _format_data_block_label(
    header: str,
    label: str,
    n_samples: int,
    feature_shapes: dict[str, tuple] | None = None,
    target_shapes: dict[str, tuple] | None = None,
    tag_shapes: dict[str, tuple] | None = None,
    overlaps: dict[str, int] | None = None,
    *,
    show_shapes: bool = True,
    show_n_samples: bool = True,
) -> str:
    """Build rich HTML label for a FeatureSet or FeatureSetView block."""
    lines = [f"<b>{header}</b>", f"'{label}'"]
    if show_n_samples:
        lines.append(f"n={n_samples}")

    # Column shapes by domain
    domain_sections: list[tuple[str, dict[str, tuple]]] = [
        ("features", feature_shapes or {}),
        ("targets", target_shapes or {}),
        ("tags", tag_shapes or {}),
    ]
    has_columns = any(shapes for _, shapes in domain_sections)
    if has_columns:
        lines.append("──────────")
        for domain_name, shapes in domain_sections:
            if not shapes:
                continue
            lines.append(f"<b>{domain_name}</b>")
            for col, shape in shapes.items():
                if show_shapes:
                    lines.append(f"  {col}: {shape}")
                else:
                    lines.append(f"  {col}")

    # Overlap section (only for split blocks)
    if overlaps is not None:
        lines.append("──────────")
        lines.append("<b>overlap</b>")
        for split_name, count in overlaps.items():
            lines.append(f"  {split_name}: {count}")

    return "<br>".join(lines)


# ================================================
# Internal representation: nodes, edges, graph
# ================================================
@dataclass
class NodeIR:
    """
    Internal representation of a single node for Mermaid rendering.

    Attributes:
        id (str): Unique node identifier.
        spec (NodeSpec): Styling spec for the node.
        label (str): Display name or pre-formatted HTML label.

    """

    id: str  # must be unique
    spec: NodeSpec  # node formatting
    label: str  # display title (may contain HTML)

    def get_label(self) -> str:
        """Returns formatted HTML label for this node."""
        # Pre-formatted labels (containing HTML tags) pass through directly
        if "<b>" in self.label or "<br>" in self.label:
            return self.label
        # Minimal labels (no header) pass through as-is
        if not self.spec.header:
            return self.label
        # Legacy: compose from spec header + label
        return f"{self.spec.header}<br>'{self.label}'"

    def get_tag_line(self) -> str:
        """Returns Mermaid tag line for node shape and label."""
        return (
            f'{self.id}@{{ label: "{self.get_label()!s}", shape: {self.spec.shape} }}'
        )

    def get_link_line(self) -> str:
        """Returns Mermaid class link line (e.g., `n1:::FeatureSet`)."""
        return f"{self.id}:::{self.spec.class_name}"


@dataclass
class EdgeIR:
    """
    Internal representation of a directed edge between nodes.

    Attributes:
        id (str): Unique edge identifier.
        src (NodeIR): Source node.
        dst (NodeIR): Destination node.
        conn_spec (EdgeConnectionSpec): Style and label of the edge.
        anim_spec (EdgeAnimationSpec): Animation styling of the edge.

    """

    id: str  # must be unique
    src: NodeIR  # source node
    dst: NodeIR  # destination node
    conn_spec: EdgeConnectionSpec
    anim_spec: EdgeAnimationSpec

    def get_connection(self) -> str:
        """Returns the Mermaid line for connecting two nodes."""
        return (
            f"{self.src.id} {self.id}@{self.conn_spec.get_connection()} {self.dst.id}"
        )

    def get_link_line(self) -> str:
        """Returns the class assignment line for this edge."""
        return f"class {self.id} {self.anim_spec.class_name}"


# ================================================
# Phase overlay helpers
# ================================================
def _insert_samplers(
    nodes: list[NodeIR],
    edges: list[EdgeIR],
    bindings: list[InputBinding],
    node_map: dict[str, NodeIR],
    fs_map: dict[str, NodeIR],
    mg: ModelGraph,
    n_ctr: int,
    e_ctr: int,
) -> tuple[int, int, list[EdgeIR]]:
    """Insert Sampler nodes between FeatureSets and head GraphNodes."""
    for binding in bindings:
        if binding.sampler is None:
            continue

        fs_id = binding.upstream_ref.node_id
        head_id = binding.node_id

        fs_nir = fs_map.get(fs_id)
        head_nir = node_map.get(head_id)
        if fs_nir is None or head_nir is None:
            continue

        # Remove the direct FS -> head edge
        edges = [
            e for e in edges if not (e.src.id == fs_nir.id and e.dst.id == head_nir.id)
        ]

        # Build sampler label
        sampler = binding.sampler
        batch_size = sampler.batcher.batch_size if hasattr(sampler, "batcher") else "?"
        sampler_label = _format_sampler_label(
            sampler_type=type(sampler).__name__,
            batch_size=batch_size,
            split=binding.split,
            stream=binding.stream if binding.stream != "default" else None,
        )
        sampler_nir = NodeIR(
            id=f"n{n_ctr}",
            spec=SAMPLER,
            label=sampler_label,
        )
        nodes.append(sampler_nir)
        n_ctr += 1

        # FS -> Sampler edge
        split_label = f"split: {binding.split}" if binding.split else None
        edges.append(
            EdgeIR(
                id=f"e{e_ctr}",
                src=fs_nir,
                dst=sampler_nir,
                conn_spec=EdgeConnectionSpec(style="-->", label=split_label),
                anim_spec=EDGE_ANIMATION_DASH_MEDIUM,
            ),
        )
        e_ctr += 1

        # Sampler -> HeadNode edge (show head node's input shape)
        head_node = mg.nodes.get(head_id) if mg is not None else None
        shape_label = None
        if head_node is not None:
            try:
                if hasattr(head_node, "input_shape"):
                    shape_label = str(head_node.input_shape)
                elif hasattr(head_node, "input_shapes"):
                    shape_label = str(head_node.input_shapes[binding.upstream_ref])
            except Exception:  # noqa: BLE001, S110
                pass
        edges.append(
            EdgeIR(
                id=f"e{e_ctr}",
                src=sampler_nir,
                dst=head_nir,
                conn_spec=EdgeConnectionSpec(style="-->", label=shape_label),
                anim_spec=EDGE_ANIMATION_DASH_MEDIUM,
            ),
        )
        e_ctr += 1

    return n_ctr, e_ctr, edges


def _attach_losses(
    nodes: list[NodeIR],
    edges: list[EdgeIR],
    losses: list[AppliedLoss],
    node_map: dict[str, NodeIR],
    n_ctr: int,
    e_ctr: int,
) -> tuple[int, int]:
    """Attach AppliedLoss nodes to their target ModelNodes."""
    for applied_loss in losses:
        target_nir = node_map.get(applied_loss.node_id)
        if target_nir is None:
            continue

        loss_label = _format_loss_label(
            loss_label=applied_loss.label,
            loss_name=applied_loss.loss.name or type(applied_loss.loss).__name__,
            weight=applied_loss.weight,
        )
        loss_nir = NodeIR(
            id=f"n{n_ctr}",
            spec=APPLIED_LOSS,
            label=loss_label,
        )
        nodes.append(loss_nir)
        n_ctr += 1

        # Loss -> TargetNode (loss "applied to" the node)
        edges.append(
            EdgeIR(
                id=f"e{e_ctr}",
                src=loss_nir,
                dst=target_nir,
                conn_spec=EdgeConnectionSpec(style="-.->", label=None),
                anim_spec=EDGE_ANIMATION_NONE,
            ),
        )
        e_ctr += 1

    return n_ctr, e_ctr


def _animate_active_edges(
    edges: list[EdgeIR],
    node_map: dict[str, NodeIR],
    active_ids: set[str],
) -> None:
    """Apply dash animation to all edges whose source and destination are active graph nodes."""
    # Build a set of NodeIR ids that correspond to active graph nodes
    active_nir_ids = {nir.id for nid, nir in node_map.items() if nid in active_ids}

    for edge in edges:
        if edge.src.id in active_nir_ids and edge.dst.id in active_nir_ids:
            edge.anim_spec = EDGE_ANIMATION_DASH_MEDIUM


def _annotate_split_edges(
    edges: list[EdgeIR],
    bindings: list[InputBinding],
    node_map: dict[str, NodeIR],
    fs_map: dict[str, NodeIR],
) -> None:
    """Add split annotations to FeatureSet -> HeadNode edges."""
    for binding in bindings:
        if not binding.split:
            continue

        fs_nir = fs_map.get(binding.upstream_ref.node_id)
        head_nir = node_map.get(binding.node_id)
        if fs_nir is None or head_nir is None:
            continue

        for edge in edges:
            if edge.src.id == fs_nir.id and edge.dst.id == head_nir.id:
                existing = edge.conn_spec.label or ""
                split_info = f"split: {binding.split}"
                combined = (
                    f"{existing} {split_info}".strip() if existing else split_info
                )
                edge.conn_spec = EdgeConnectionSpec(
                    style=edge.conn_spec.style,
                    label=combined,
                )


# ================================================
# GraphIR: full graph representation
# ================================================
class GraphIR:
    """Internal representation of an entire graph (nodes + edges), used for generating Mermaid diagrams."""

    def __init__(
        self,
        nodes: list[NodeIR],
        edges: list[EdgeIR],
        label: str | None = None,
    ):
        self.nodes = nodes
        self.edges = edges
        self.label = label

    # ------------------------------------------------
    # Base model graph builder
    # ------------------------------------------------
    @classmethod
    def _build_model_graph_ir(
        cls,
        mg: ModelGraph,
        opts: ModelGraphDisplayOptions | None = None,
    ) -> tuple[
        list[NodeIR],
        list[EdgeIR],
        dict[str, NodeIR],
        dict[str, NodeIR],
        int,
        int,
    ]:
        """
        Build the base IR from a ModelGraph.

        Args:
            mg: The model graph to convert.
            opts: Display options controlling what is shown on nodes.

        Returns:
            tuple containing:
                - nodes: list of NodeIR
                - edges: list of EdgeIR
                - node_map: dict mapping node_id -> NodeIR (for graph nodes)
                - fs_map: dict mapping featureset node_id -> NodeIR
                - n_ctr: next available node counter
                - e_ctr: next available edge counter

        """
        from modularml.core.references.featureset_reference import FeatureSetReference
        from modularml.core.topology.merge_nodes.merge_node import MergeNode
        from modularml.core.topology.model_node import ModelNode

        if opts is None:
            opts = ModelGraphDisplayOptions()

        nodes: list[NodeIR] = []
        edges: list[EdgeIR] = []
        node_map: dict[str, NodeIR] = {}  # graph node_id -> NodeIR
        fs_map: dict[str, NodeIR] = {}  # featureset node_id -> NodeIR
        n_ctr = 0
        e_ctr = 0

        # Create NodeIR for each GraphNode in topological order
        for node_id in mg._sorted_node_ids:
            node = mg.nodes[node_id]

            if isinstance(node, ModelNode):
                # Collect upstream features/targets/tags for head nodes
                ups_features: list[str] | None = None
                ups_targets: list[str] | None = None
                ups_tags: list[str] | None = None
                if is_head_node(node):
                    for ref in node.get_upstream_refs(error_mode=ErrorMode.IGNORE):
                        if isinstance(ref, FeatureSetReference):
                            if opts.show_features and ref.features:
                                ups_features = [f.split(".")[1] for f in ref.features]
                            if opts.show_targets and ref.targets:
                                ups_targets = [t.split(".")[1] for t in ref.targets]
                            if opts.show_tags and ref.tags:
                                ups_tags = [t.split(".")[1] for t in ref.tags]

                show_frozen = node.is_frozen and opts.show_frozen
                label = _format_model_node_label(
                    node_label=node.label,
                    backend=str(node.backend.value) if node.is_built else None,
                    is_frozen=show_frozen,
                    upstream_features=ups_features,
                    upstream_targets=ups_targets,
                    upstream_tags=ups_tags,
                )
                spec = MODEL_NODE_FROZEN if show_frozen else MODEL_NODE
                nir = NodeIR(id=f"n{n_ctr}", spec=spec, label=label)

            elif isinstance(node, MergeNode):
                merge_type = type(node).__name__
                label = _format_merge_node_label(
                    node_label=node.label,
                    merge_type=merge_type,
                )
                nir = NodeIR(id=f"n{n_ctr}", spec=MERGE_NODE, label=label)

            else:
                # Fallback for unknown GraphNode subtypes
                label = f"<b>{type(node).__name__}</b><br>'{node.label}'"
                nir = NodeIR(id=f"n{n_ctr}", spec=MODEL_NODE, label=label)

            nodes.append(nir)
            node_map[node_id] = nir
            n_ctr += 1

        # Create edges from upstream references
        for node_id in mg._sorted_node_ids:
            node = mg.nodes[node_id]
            dst_nir = node_map[node_id]

            for ref in node.get_upstream_refs(error_mode=ErrorMode.IGNORE):
                if isinstance(ref, FeatureSetReference):
                    # Create or reuse FeatureSet source node
                    fs_id = ref.node_id
                    if fs_id not in fs_map:
                        try:
                            fsv = ref.resolve()
                            fs = fsv.source
                            fs_label = _format_featureset_label(
                                fs_label=fs.label,
                                n_samples=len(fs),
                                splits=(
                                    fs.available_splits
                                    if (
                                        opts.show_splits
                                        and len(fs.available_splits) > 0
                                    )
                                    else None
                                ),
                            )
                        except Exception:  # noqa: BLE001
                            fs_label = _format_featureset_label(
                                fs_label=ref.node_label or fs_id,
                            )
                        fs_nir = NodeIR(
                            id=f"n{n_ctr}",
                            spec=FEATURE_SET,
                            label=fs_label,
                        )
                        nodes.append(fs_nir)
                        fs_map[fs_id] = fs_nir
                        n_ctr += 1

                    src_nir = fs_map[fs_id]
                    try:
                        if hasattr(node, "input_shape"):
                            edge_label = node.input_shape
                        elif hasattr(node, "input_shapes"):
                            edge_label = node.inputs_shapes[ref]
                    except Exception:  # noqa: BLE001
                        edge_label = None

                else:
                    # GraphNodeReference -> edge to another graph node
                    ups_node_id = ref.node_id
                    if ups_node_id not in node_map:
                        continue
                    src_nir = node_map[ups_node_id]

                    # Show output shape of upstream node on the edge
                    edge_label = None
                    ups_node = mg.nodes.get(ups_node_id)
                    if ups_node is not None:
                        try:
                            if hasattr(ups_node, "output_shape"):
                                edge_label = str(ups_node.output_shape)
                        except Exception:  # noqa: BLE001, S110
                            pass

                edges.append(
                    EdgeIR(
                        id=f"e{e_ctr}",
                        src=src_nir,
                        dst=dst_nir,
                        conn_spec=EdgeConnectionSpec(style="-->", label=edge_label),
                        anim_spec=EDGE_ANIMATION_NONE,
                    ),
                )
                e_ctr += 1

        # Add terminal output nodes for tail nodes
        for node_id in mg._sorted_node_ids:
            node = mg.nodes[node_id]
            if not is_tail_node(node):
                continue

            tail_nir = node_map[node_id]
            output_label = None
            if hasattr(node, "output_shape"):
                try:
                    if node.is_built:
                        output_label = str(node.output_shape)
                except Exception:  # noqa: BLE001, S110
                    pass

            terminal_nir = NodeIR(
                id=f"n{n_ctr}",
                spec=OUTPUT_TERMINAL,
                label=" ",
            )
            nodes.append(terminal_nir)
            n_ctr += 1

            edges.append(
                EdgeIR(
                    id=f"e{e_ctr}",
                    src=tail_nir,
                    dst=terminal_nir,
                    conn_spec=EdgeConnectionSpec(style="-->", label=output_label),
                    anim_spec=EDGE_ANIMATION_NONE,
                ),
            )
            e_ctr += 1

        return nodes, edges, node_map, fs_map, n_ctr, e_ctr

    # ------------------------------------------------
    # Public constructors
    # ------------------------------------------------
    @classmethod
    def from_model_graph(
        cls,
        mg: ModelGraph,
        opts: ModelGraphDisplayOptions | None = None,
    ) -> GraphIR:
        """
        Converts a ModelGraph into a GraphIR.

        Args:
            mg (ModelGraph): The model graph to convert.
            opts (ModelGraphDisplayOptions | None): Display options.

        Returns:
            GraphIR: The resulting internal graph representation.

        """
        nodes, edges, _, _, _, _ = cls._build_model_graph_ir(mg, opts=opts)
        return cls(nodes=nodes, edges=edges, label=mg.label)

    @classmethod
    def from_featureset(
        cls,
        fs: FeatureSet,
        opts: FeatureSetDisplayOptions | None = None,
    ) -> GraphIR:
        """
        Converts a FeatureSet into a GraphIR showing its structure and splits.

        The visualization is a left-to-right flowchart with:
        - The FeatureSet block on the left (label, n_samples, columns by domain with shapes)
        - Split blocks on the right, connected via the split hierarchy from ``_split_recs``
        - Each split block shows column shapes, n_samples, and overlap counts with all other splits

        Args:
            fs (FeatureSet): The FeatureSet to visualize.
            opts (FeatureSetDisplayOptions | None): Display options. Defaults to
                :class:`FeatureSetDisplayOptions` with default values.

        Returns:
            GraphIR: The resulting graph representation.

        """
        if opts is None:
            opts = FeatureSetDisplayOptions()

        def _get_shapes(
            source: Any,
            *,
            is_root: bool,
        ) -> tuple[dict | None, dict | None, dict | None]:
            """Collect domain shapes respecting display options."""
            kw = {"include_domain_prefix": False, "include_rep_suffix": True}
            feat = source.get_feature_shapes(**kw) if opts.show_features else None
            targ = source.get_target_shapes(**kw) if opts.show_targets else None
            # Tags: True -> always, "root" -> only on root, False -> never
            show_tags_here = opts.show_tags is True or (
                opts.show_tags == "root" and is_root
            )
            tags = source.get_tag_shapes(**kw) if show_tags_here else None
            return feat, targ, tags

        nodes: list[NodeIR] = []
        edges: list[EdgeIR] = []
        n_ctr = 0
        e_ctr = 0

        # -- FeatureSet root node --
        feat, targ, tags = _get_shapes(fs, is_root=True)
        fs_label = _format_data_block_label(
            header="FeatureSet",
            label=fs.label,
            n_samples=fs.n_samples,
            feature_shapes=feat,
            target_shapes=targ,
            tag_shapes=tags,
            show_shapes=opts.show_shapes,
            show_n_samples=opts.show_n_samples,
        )
        fs_nir = NodeIR(id=f"n{n_ctr}", spec=FEATURE_SET, label=fs_label)
        nodes.append(fs_nir)
        n_ctr += 1

        if not fs.available_splits:
            return cls(nodes=nodes, edges=edges, label=fs.label)

        # -- Collect all split views --
        split_views: dict[str, Any] = {}
        for split_name in fs.available_splits:
            split_views[split_name] = fs.get_split(split_name)

        # -- Create split nodes --
        split_nirs: dict[str, NodeIR] = {}
        for split_name, view in split_views.items():
            # Compute overlaps with all other splits
            overlaps: dict[str, int] | None = None
            if opts.show_overlaps:
                overlaps = {}
                for other_name, other_view in split_views.items():
                    if other_name == split_name:
                        continue
                    overlaps[other_name] = len(view.get_overlap_with(other_view))

            feat, targ, tags = _get_shapes(view, is_root=False)
            split_label = _format_data_block_label(
                header="Split",
                label=split_name,
                n_samples=view.n_samples,
                feature_shapes=feat,
                target_shapes=targ,
                tag_shapes=tags,
                overlaps=overlaps,
                show_shapes=opts.show_shapes,
                show_n_samples=opts.show_n_samples,
            )
            nir = NodeIR(id=f"n{n_ctr}", spec=FEATURE_SET_VIEW, label=split_label)
            nodes.append(nir)
            split_nirs[split_name] = nir
            n_ctr += 1

        # -- Build edges from split hierarchy (_split_recs) --
        recorded_splits: set[str] = set()
        for rec in fs._split_recs:
            # Determine parent node
            parent_split = rec.applied_to.split_name
            if parent_split is None:
                parent_nir = fs_nir
            else:
                parent_nir = split_nirs.get(parent_split)
                if parent_nir is None:
                    continue

            for produced in rec.produced_splits:
                child_nir = split_nirs.get(produced)
                if child_nir is None:
                    continue
                recorded_splits.add(produced)

                child_view = split_views[produced]
                edges.append(
                    EdgeIR(
                        id=f"e{e_ctr}",
                        src=parent_nir,
                        dst=child_nir,
                        conn_spec=EdgeConnectionSpec(
                            style="-->",
                            label=f"n={child_view.n_samples}",
                        ),
                        anim_spec=EDGE_ANIMATION_NONE,
                    ),
                )
                e_ctr += 1

        # -- Handle manually added splits (no SplitterRecord) --
        for split_name, nir in split_nirs.items():
            if split_name not in recorded_splits:
                child_view = split_views[split_name]
                edges.append(
                    EdgeIR(
                        id=f"e{e_ctr}",
                        src=fs_nir,
                        dst=nir,
                        conn_spec=EdgeConnectionSpec(
                            style="-->",
                            label=f"n={child_view.n_samples}",
                        ),
                        anim_spec=EDGE_ANIMATION_NONE,
                    ),
                )
                e_ctr += 1

        return cls(nodes=nodes, edges=edges, label=fs.label)

    @classmethod
    def from_train_phase(cls, phase: TrainPhase) -> GraphIR:
        """
        Converts a TrainPhase into a GraphIR showing the full training configuration.

        The visualization includes the ModelGraph topology with:
        - Active vs inactive nodes (inactive nodes are dimmed)
        - Sampler nodes inserted between FeatureSets and head nodes
        - AppliedLoss nodes attached to their target ModelNodes

        Args:
            phase (TrainPhase): The training phase to visualize.

        Returns:
            GraphIR: The resulting graph representation.

        """
        from modularml.core.experiment.experiment_context import ExperimentContext

        mg = ExperimentContext.get_active().model_graph
        nodes, edges, node_map, fs_map, n_ctr, e_ctr = cls._build_model_graph_ir(mg)

        # Dim inactive nodes
        active_ids = set(phase.active_nodes)
        for nid, nir in node_map.items():
            if nid not in active_ids:
                nir.spec = INACTIVE_NODE

        # Insert samplers
        n_ctr, e_ctr, edges = _insert_samplers(
            nodes,
            edges,
            phase.input_sources,
            node_map,
            fs_map,
            mg,
            n_ctr,
            e_ctr,
        )

        # Animate all edges between active nodes
        _animate_active_edges(edges, node_map, active_ids)

        # Attach losses
        if phase.losses:
            n_ctr, e_ctr = _attach_losses(
                nodes,
                edges,
                phase.losses,
                node_map,
                n_ctr,
                e_ctr,
            )

        label = f"TrainPhase: '{phase.label}' (epochs={phase.n_epochs})"
        return cls(nodes=nodes, edges=edges, label=label)

    @classmethod
    def from_eval_phase(cls, phase: EvalPhase) -> GraphIR:
        """
        Converts an EvalPhase into a GraphIR showing the evaluation configuration.

        The visualization includes the ModelGraph topology with:
        - All ModelNodes shown as frozen
        - Split annotations on FeatureSet edges
        - AppliedLoss nodes if any are defined

        Args:
            phase (EvalPhase): The evaluation phase to visualize.

        Returns:
            GraphIR: The resulting graph representation.

        """
        from modularml.core.experiment.experiment_context import ExperimentContext

        mg = ExperimentContext.get_active().model_graph
        nodes, edges, node_map, fs_map, n_ctr, e_ctr = cls._build_model_graph_ir(mg)

        # Mark all ModelNodes as frozen
        for nir in node_map.values():
            if nir.spec == MODEL_NODE:
                nir.spec = MODEL_NODE_FROZEN

        # Dim inactive nodes
        active_ids = set(phase.active_nodes)
        for nid, nir in node_map.items():
            if nid not in active_ids:
                nir.spec = INACTIVE_NODE

        # Insert samplers (if any bindings define one)
        has_samplers = any(b.sampler is not None for b in phase.input_sources)
        if has_samplers:
            n_ctr, e_ctr, edges = _insert_samplers(
                nodes,
                edges,
                phase.input_sources,
                node_map,
                fs_map,
                mg,
                n_ctr,
                e_ctr,
            )
        else:
            # Annotate edges with split info (no sampler nodes to show)
            _annotate_split_edges(edges, phase.input_sources, node_map, fs_map)

        # Animate all edges between active nodes
        _animate_active_edges(edges, node_map, active_ids)

        # Attach losses if any
        if phase.losses:
            n_ctr, e_ctr = _attach_losses(
                nodes,
                edges,
                phase.losses,
                node_map,
                n_ctr,
                e_ctr,
            )

        # Build label
        split_text = ""
        if phase.input_sources and phase.input_sources[0].split:
            split_text = f", split='{phase.input_sources[0].split}'"
        label = f"EvalPhase: '{phase.label}'{split_text}"

        return cls(nodes=nodes, edges=edges, label=label)

    @classmethod
    def from_fit_phase(cls, phase: FitPhase) -> GraphIR:
        """
        Converts a FitPhase into a GraphIR showing the fit configuration.

        The visualization includes the ModelGraph topology with:
        - All ModelNodes shown as frozen
        - Split annotations on FeatureSet edges
        - freeze_after_fit indicator in the label
        - AppliedLoss nodes if any are defined

        Args:
            phase (FitPhase): The fit phase to visualize.

        Returns:
            GraphIR: The resulting graph representation.

        """
        from modularml.core.experiment.experiment_context import ExperimentContext

        mg = ExperimentContext.get_active().model_graph
        nodes, edges, node_map, fs_map, n_ctr, e_ctr = cls._build_model_graph_ir(mg)

        # Mark all ModelNodes as frozen
        for nir in node_map.values():
            if nir.spec == MODEL_NODE:
                nir.spec = MODEL_NODE_FROZEN

        # Dim inactive nodes
        active_ids = set(phase.active_nodes)
        for nid, nir in node_map.items():
            if nid not in active_ids:
                nir.spec = INACTIVE_NODE

        # Insert samplers (if any bindings define one)
        has_samplers = any(b.sampler is not None for b in phase.input_sources)
        if has_samplers:
            n_ctr, e_ctr, edges = _insert_samplers(
                nodes,
                edges,
                phase.input_sources,
                node_map,
                fs_map,
                mg,
                n_ctr,
                e_ctr,
            )
        else:
            # Annotate edges with split info (no sampler nodes to show)
            _annotate_split_edges(edges, phase.input_sources, node_map, fs_map)

        # Animate all edges between active nodes
        _animate_active_edges(edges, node_map, active_ids)

        # Attach losses if any
        if phase.losses:
            n_ctr, e_ctr = _attach_losses(
                nodes,
                edges,
                phase.losses,
                node_map,
                n_ctr,
                e_ctr,
            )

        # Build label
        freeze_text = " (freeze_after_fit)" if phase.freeze_after_fit else ""
        label = f"FitPhase: '{phase.label}'{freeze_text}"

        return cls(nodes=nodes, edges=edges, label=label)

    # ------------------------------------------------
    # Mermaid rendering
    # ------------------------------------------------
    def to_mermaid(self) -> str:
        """
        Converts the full GraphIR into a Mermaid.js-compatible string.

        Returns:
            str: Mermaid flowchart syntax.

        """
        connections: list[str] = []  # connections statments (eg, 'n1 e1@-> n2')
        node_class_link: list[
            str
        ] = []  # node --> classDef link (each element is single line)
        node_class_defs: list[
            str
        ] = []  # node classDef statements (each element is single line)
        node_tags: list[str] = []  # node tags (e.g., 'n1@{label: "...", shape: ... }')
        edge_class_link: list[str] = []  # e.g., 'class e1 NoAnimation'
        anim_class_defs: list[str] = []  # e.g., 'classDef NoAnimation ...'

        # Add nodes classes & links
        for n in self.nodes:
            node_class_link.append(n.get_link_line())
            node_tags.append(n.get_tag_line())
            node_class_defs.append(n.spec.to_class_def())

        # Add edge classes & links
        for e in self.edges:
            connections.append(e.get_connection())
            edge_class_link.append(e.get_link_line())
            anim_class_defs.append(e.anim_spec.to_class_def())

        # Remove repeated node and animation classDefs
        node_class_defs = list(set(node_class_defs))
        anim_class_defs = list(set(anim_class_defs))

        # Build full mermaid text:
        nl = "\n\t"  # newline + tab
        all_lines = "flowchart LR"

        for conn in connections:  # all node connections
            all_lines += nl + conn
        all_lines += "\n"  # new line (not needed but visually better)

        for tag in node_tags:  # node tags
            all_lines += nl + tag
        for link in node_class_link:  # node link to classDef
            all_lines += nl + link
        for cdef in node_class_defs:  # node classDefs
            all_lines += nl + cdef
        all_lines += "\n"  # new line (not needed but visually better)

        for cdef in anim_class_defs:  # anim classDefs
            all_lines += nl + cdef
        for link in edge_class_link:  # edge link to anim classDef
            all_lines += nl + link

        return all_lines
