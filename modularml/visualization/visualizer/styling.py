"""Mermaid-based visualization for ModularML core objects."""

from __future__ import annotations

from dataclasses import dataclass


# ================================================
# Node styling specifications
# ================================================
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
            f"color:{self.color}"  # color can't have a space after the colon for some reason
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
FEATURE_SET_VIEW = NodeSpec(
    class_name="FeatureSetView",
    color="#000000",
    fill="#F3E5F5",
    stroke="#AA00FF",
    header="Split",
    shape="rect",
)
SAMPLER = NodeSpec(
    class_name="FeatureSampler",
    color="#000000",
    fill="#FFE0B2",
    stroke="#FF6D00",
    header="Sampler",
    shape="rect",
)
MODEL_NODE = NodeSpec(
    class_name="ModelNode",
    color="#000000",
    fill="#BBDEFB",
    stroke="#2962FF",
    header="ModelNode",
    shape="rect",
)
MODEL_NODE_FROZEN = NodeSpec(
    class_name="ModelNodeFrozen",
    color="#666666",
    fill="#E3F2FD",
    stroke="#90CAF9",
    header="ModelNode",
    shape="rect",
)
MERGE_NODE = NodeSpec(
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
OUTPUT_TERMINAL = NodeSpec(
    class_name="OutputTerminal",
    color="#FFFFFF",
    fill="#616161",
    stroke="#424242",
    header="",
    shape="circle",
)
INACTIVE_NODE = NodeSpec(
    class_name="InactiveNode",
    color="#999999",
    fill="#F5F5F5",
    stroke="#BDBDBD",
    header="",
    shape="rect",
)


# ================================================
# Display options
# ================================================
@dataclass
class DisplayOptions:
    """Base display options for Visualizer."""


@dataclass
class FeatureSetDisplayOptions(DisplayOptions):
    """
    Controls what information is shown when visualizing a FeatureSet.

    Attributes:
        show_features: Show feature columns and shapes on nodes.
        show_targets: Show target columns and shapes on nodes.
        show_tags: Show tag columns and shapes. ``"root"`` shows only on the
            FeatureSet root node, ``True`` shows on all nodes, ``False`` hides everywhere.
        show_shapes: Show column shapes next to column names.
        show_overlaps: Show overlap counts between splits.
        show_n_samples: Show sample counts on nodes.

    """

    show_features: bool = True
    show_targets: bool = True
    show_tags: bool | str = "root"  # True | False | "root"
    show_shapes: bool = True
    show_overlaps: bool = True
    show_n_samples: bool = True


@dataclass
class ModelGraphDisplayOptions(DisplayOptions):
    """
    Controls what information is shown when visualizing a ModelGraph.

    Attributes:
        show_features: Show feature columns on head nodes.
        show_targets: Show target columns on head nodes.
        show_tags: Show tag columns on head nodes.
        show_frozen: Show frozen state (label text and dimmed styling) on ModelNodes.
        show_splits: Show available splits on FeatureSet nodes

    """

    show_features: bool = True
    show_targets: bool = True
    show_tags: bool = False
    show_frozen: bool = True
    show_splits: bool = False


# ================================================
# Edge styling specifications
# ================================================
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
    properties={
        "stroke-dasharray": "9,5",
        "stroke-dashoffset": "100",
        "animation": "dash 8s linear infinite",
    },
)
EDGE_ANIMATION_DASH_MEDIUM = EdgeAnimationSpec(
    class_name="DashMediumAnimation",
    properties={
        "stroke-dasharray": "9,5",
        "stroke-dashoffset": "100",
        "animation": "dash 3s linear infinite",
    },
)
EDGE_ANIMATION_DASH_FAST = EdgeAnimationSpec(
    class_name="DashFastAnimation",
    properties={
        "stroke-dasharray": "9,5",
        "stroke-dashoffset": "100",
        "animation": "dash 1s linear infinite",
    },
)


# ================================================
# Edge connection style
# ================================================
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
