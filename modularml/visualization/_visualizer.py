from dataclasses import dataclass
from pathlib import Path
from typing import Any

from IPython.display import Markdown, display

from modularml.core.data.featureset import FeatureSet
from modularml.core.topology.merge_nodes.merge_node import MergeNode
from modularml.core.topology.model_graph import ModelGraph
from modularml.core.topology.model_node import ModelNode
from modularml.utils.errors.error_handling import ErrorMode


@dataclass
class NodeSpec:
    """
    Styling specification for a node in the Mermaid graph.

    Attributes:
        class_name (str): The Mermaid class name used in `classDef`.
        color (str): Text color.
        fill (str): Background color.
        stroke (str): Border color.
        header (str): Header text used in the label.
        shape (str): Shape of the node (e.g., 'rect', 'stadium').

    """

    class_name: str  # classDef name
    color: str  # text color
    fill: str  # background color
    stroke: str  # stroke color
    header: str  # header text
    shape: str  # node shape

    def to_class_def(self) -> str:
        """Converts the node spec into a Mermaid `classDef` declaration."""
        return (
            f"classDef {self.class_name} stroke-width: 2px, stroke-dasharray: 0, "
            f"stroke: {self.stroke}, "
            f"fill: {self.fill}, "
            f"color:{self.color}"  # color can't have a space after the colon from some reason
            ";"
        )


FEATURE_SET = NodeSpec(
    class_name="FeatureSet",
    color="#000000",
    fill="#E1BEE7",
    stroke="#AA00FF",
    header="FeatureSet",
    shape="rect",
)
FEATURE_SUBSET = NodeSpec(
    class_name="FeatureSubset",
    color="#000000",
    fill="#E1BEE7",
    stroke="#AA00FF",
    header="FeatureSet",
    shape="rect",
)
FEATURE_SAMPLER = NodeSpec(
    class_name="FeatureSampler",
    color="#000000",
    fill="#FFE0B2",
    stroke="#FF6D00",
    header="FeatureSampler",
    shape="rect",
)
MODEL_STAGE = NodeSpec(
    class_name="ModelNode",
    color="#000000",
    fill="#BBDEFB",
    stroke="#2962FF",
    header="ModelNode",
    shape="rect",
)
MERGE_STAGE = NodeSpec(
    class_name="MergeNode",
    color="#000000",
    fill="#B1B1B1",
    stroke="#565656",
    header="MergeNode",
    shape="rect",
)
APPLIED_LOSS = NodeSpec(
    class_name="AppliedLoss",
    color="#000000",
    fill="#FFCDD2",
    stroke="#D50000",
    header="AppliedLoss",
    shape="stadium",
)


@dataclass
class EdgeAnimationSpec:
    """
    Styling specification for edge animation in Mermaid.

    Attributes:
        class_name (str): The Mermaid class name for the edge.
        properties (dict[str, str]): CSS-style animation properties.

    """

    class_name: str
    properties: dict[str, str]

    def to_class_def(self) -> str:
        """Converts the animation spec into a Mermaid `classDef` line."""
        line = f"classDef {self.class_name} "
        for k, v in self.properties.items():
            line += f"{k}: {v}, "
        line = line[:-2]  # remove last ", "
        line += ";"  # end with semi-colon
        return line


EDGE_ANIMATION_NONE = EdgeAnimationSpec(
    class_name="NoAnimation",
    properties={
        "stroke-dasharray": "0",
    },
)
EDGE_ANIMATION_DASH_SLOW = EdgeAnimationSpec(
    class_name="DashSlowAnimation",
    properties={"stroke-dasharray": "9,5", "stroke-dashoffset": "100", "animation": "dash 8s linear infinite"},
)
EDGE_ANIMATION_DASH_MEDIUM = EdgeAnimationSpec(
    class_name="DashMediumAnimation",
    properties={"stroke-dasharray": "9,5", "stroke-dashoffset": "100", "animation": "dash 3s linear infinite"},
)
EDGE_ANIMATION_DASH_FAST = EdgeAnimationSpec(
    class_name="DashFastAnimation",
    properties={"stroke-dasharray": "9,5", "stroke-dashoffset": "100", "animation": "dash 1s linear infinite"},
)


@dataclass
class EdgeConnectionSpec:
    """
    Connection style between two nodes in Mermaid.

    Attributes:
        style (str): Edge style (e.g., '-->', '-.->').
        label (str | None): Optional label text to display on the edge.

    """

    style: str
    label: str | None = None

    def get_connection(self) -> str:
        """
        Returns the edge connection string.

        Returns:
            str: Mermaid-compatible connection string (e.g., `-->`, `-->|label|`)

        """
        if self.label is not None:
            return f'{self.style}|"{self.label}"|'
        return f"{self.style}"


@dataclass
class NodeIR:
    """
    Internal representation of a single node for Mermaid rendering.

    Attributes:
        id (str): Unique node identifier.
        spec (NodeSpec): Styling spec for the node.
        label (str): Display name of the node.

    """

    id: str  # must be unique
    spec: NodeSpec  # node formatting
    label: str  # display title

    def get_label(
        self,
    ) -> str:
        """Returns formatted HTML label for this node."""
        return f"{self.spec.header}<br>'{self.label}'"

    def get_tag_line(self) -> str:
        """Returns Mermaid tag line for node shape and label."""
        return f'{self.id}@{{ label: "{self.get_label()!s}", shape: {self.spec.shape} }}'

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

    def get_connection(
        self,
    ) -> str:
        """Returns the Mermaid line for connecting two nodes."""
        return f"{self.src.id} {self.id}@{self.conn_spec.get_connection()} {self.dst.id}"

    def get_link_line(self) -> str:
        """Returns the class assignment line for this edge."""
        return f"class {self.id} {self.anim_spec.class_name}"


class GraphIR:
    """Internal representation of an entire graph (nodes + edges), used for generating Mermaid diagrams."""

    def __init__(self, nodes: list[NodeIR], edges: list[EdgeIR], label: str | None = None):
        self.nodes = nodes
        self.edges = edges
        self.label = label

    @classmethod
    def from_model_graph(cls, mg: ModelGraph):
        """
        Converts a ModelGraph object into a GraphIR.

        Args:
            mg (ModelGraph): The model graph to convert.

        Returns:
            GraphIR: The resulting internal graph representation.

        """
        nodes: list[NodeIR] = []
        edges: list[EdgeIR] = []

        n_id_ctr = 0
        e_id_ctr = 0
        for node_lbl, node in mg._nodes.items():
            if isinstance(node, ModelNode):
                nodes.append(NodeIR(id=f"n{n_id_ctr}", spec=MODEL_STAGE, label=node_lbl))
                n_id_ctr += 1

            elif isinstance(node, MergeNode):
                nodes.append(NodeIR(id=f"n{n_id_ctr}", spec=MERGE_STAGE, label=node_lbl))
                n_id_ctr += 1

            elif isinstance(node, FeatureSet):
                nodes.append(NodeIR(id=f"n{n_id_ctr}", spec=FEATURE_SET, label=node_lbl))
                n_id_ctr += 1

            else:
                msg = f"Unkown node type in ModelGraph: {node}"
                raise TypeError(msg)

        # Go through all non-source nodes
        for node_lbl in mg.connected_node_labels:
            # Create all upstream edge connections (src -> dst)
            for ups_node_lbl in mg._nodes[node_lbl].get_upstream_nodes(error_mode=ErrorMode.COERCE):
                src: NodeIR = None
                dst: NodeIR = None
                for n in nodes:
                    if n.label == ups_node_lbl:
                        src = n
                    if n.label == node_lbl:
                        dst = n

                if src is None:
                    msg = f"Failed to find source node for input: {ups_node_lbl}. All nodes: {[n.label in nodes]}"
                    raise RuntimeError(msg)
                if dst is None:
                    msg = f"Failed to find destination node for {node_lbl}."
                    raise RuntimeError(msg)
                src_shape = str(mg._nodes[src.label].output_shape_spec.merged_shape)
                e = EdgeIR(
                    id=f"e{e_id_ctr}",
                    src=src,
                    dst=dst,
                    conn_spec=EdgeConnectionSpec(style="-->", label=src_shape),
                    anim_spec=EDGE_ANIMATION_NONE,  # EDGE_ANIMATION_DASH_MEDIUM, EDGE_ANIMATION_NONE,
                )
                edges.append(e)
                e_id_ctr += 1

        return cls(nodes=nodes, edges=edges)

    def to_mermaid(
        self,
    ) -> str:
        """
        Converts the full GraphIR into a Mermaid.js-compatible string.

        Returns:
            str: Mermaid flowchart syntax.

        """
        connections: list[str] = []  # connections statments (eg, 'n1 e1@-> n2')
        node_class_link: list[str] = []  # node --> classDef link (each element is single line)
        node_class_defs: list[str] = []  # node classDef statements (each element is single line)
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


class Visualizer:
    """
    Visualization utility for ModularML objects.

    Supports generating Mermaid diagrams from:\
    - ModelGraph\
    - (Future: TrainingPhase, Experiment, etc.)

    Methods:
        display(): Show Mermaid diagram inline in Jupyter

    """

    def __init__(self, obj: Any):
        """
        Initializes the visualizer with the target object.

        Args:
            obj (Any): Object to visualize (e.g., ModelGraph).

        """
        self.obj = obj

    def get_mermaid_str(self) -> str:
        """
        Generates the Mermaid string from the internal graph object.

        Returns:
            str: Mermaid diagram as string.

        """
        if isinstance(self.obj, ModelGraph):
            graph = GraphIR.from_model_graph(self.obj)
            return graph.to_mermaid()

        # elif isinstance(self.obj, ModelNode):
        #     stage = StageIR.from_model_stage(self.obj)
        #     return stage.to_mermaid()

        msg = f"Object type is not yet supported by Visualizer: {self.obj}"
        raise NotImplementedError(msg)

    def mermaid_to_mmd(self, filepath: str):
        """
        Writes the Mermaid string to a `.mmd` markdown file.

        Args:
            filepath (str or Path): Path to output file (suffix `.mmd` auto-added).

        Returns:
            str: Full path to the written file.

        """
        filepath: Path = Path(filepath)
        filepath = filepath.with_suffix(".mmd")

        # Write mermaid markdown to file
        with Path.open(filepath, "w") as f:
            f.write(self.get_mermaid_str())

        return str(filepath)

    def display_mermaid(self):
        """
        Displays the Mermaid diagram inline in Jupyter Notebook.

        Requires Mermaid support in the notebook viewer.
        """
        mermaid_str = self.get_mermaid_str()
        display(Markdown(f"```mermaid\n{mermaid_str}\n```"))

    def display(self, backend: str = "mermaid"):
        """
        Displays the graph using the specified backend.

        Args:
            backend (str): One of {'mermaid'} (others not yet implemented)

        """
        if backend == "mermaid":
            return self.display_mermaid()

        msg = f"Display type not supported: {backend}"
        raise NotImplementedError(msg)
