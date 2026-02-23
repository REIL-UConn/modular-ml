"""Mermaid-based visualization for ModularML core objects."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from IPython.display import Markdown, display

from modularml.visualization.visualizer.internal_representation import GraphIR

if TYPE_CHECKING:
    from modularml.visualization.visualizer.styling import DisplayOptions


# ================================================
# Visualizer: public API
# ================================================
class Visualizer:
    """
    Visualization utility for ModularML objects.

    Supports generating Mermaid diagrams from:
    - FeatureSet: split hierarchy with column shapes and overlap counts
    - ModelGraph: flowchart of nodes with shape annotations
    - TrainPhase: training configuration with samplers, losses, active nodes
    - EvalPhase: evaluation configuration with split annotations
    - FitPhase: fit configuration with freeze indicators

    For FeatureSet visualization, pass a :class:`FeatureSetDisplayOptions` to
    control which sections are shown (features, targets, tags, shapes, overlaps).

    Examples:
        ```python
        # Visualize a FeatureSet (tags only on root node by default)
        Visualizer(feature_set).display()

        # Visualize a FeatureSet with tags on all nodes
        Visualizer(feature_set, FeatureSetDisplayOptions(show_tags=True)).display()

        # Visualize a ModelGraph
        Visualizer(model_graph).display()

        # Get raw Mermaid string
        mermaid_str = Visualizer(model_graph).get_mermaid_str()

        # Export to .mmd file
        Visualizer(model_graph).mermaid_to_mmd("my_graph")
        ```

    """

    def __init__(
        self,
        obj: Any,
        display_options: DisplayOptions | None = None,
    ):
        """
        Initializes the visualizer with the target object.

        Args:
            obj (Any): Object to visualize (FeatureSet, ModelGraph, TrainPhase,
                EvalPhase, or FitPhase).
            display_options (DisplayOptions | None): Display options controlling
                what is shown. Pass :class:`FeatureSetDisplayOptions` for FeatureSet
                or :class:`ModelGraphDisplayOptions` for ModelGraph visualization.

        """
        self.obj = obj
        self.display_options = display_options
        self._graph: GraphIR | None = None

    def _build_graph(self) -> GraphIR:
        """Build the GraphIR from the wrapped object."""
        if self._graph is not None:
            return self._graph

        from modularml.core.data.featureset import FeatureSet
        from modularml.core.experiment.phases.eval_phase import EvalPhase
        from modularml.core.experiment.phases.fit_phase import FitPhase
        from modularml.core.experiment.phases.train_phase import TrainPhase
        from modularml.core.topology.model_graph import ModelGraph

        if isinstance(self.obj, FeatureSet):
            self._graph = GraphIR.from_featureset(self.obj, opts=self.display_options)
        elif isinstance(self.obj, ModelGraph):
            self._graph = GraphIR.from_model_graph(self.obj, opts=self.display_options)
        elif isinstance(self.obj, TrainPhase):
            self._graph = GraphIR.from_train_phase(self.obj)
        elif isinstance(self.obj, EvalPhase):
            self._graph = GraphIR.from_eval_phase(self.obj)
        elif isinstance(self.obj, FitPhase):
            self._graph = GraphIR.from_fit_phase(self.obj)
        else:
            msg = f"Object type not supported by Visualizer: {type(self.obj)}"
            raise NotImplementedError(msg)

        return self._graph

    def get_mermaid_str(self) -> str:
        """
        Generates the Mermaid string from the internal graph object.

        Returns:
            str: Mermaid diagram as string.

        """
        return self._build_graph().to_mermaid()

    def mermaid_to_mmd(self, filepath: str) -> str:
        """
        Writes the Mermaid string to a `.mmd` markdown file.

        Args:
            filepath (str or Path): Path to output file (suffix `.mmd` auto-added).

        Returns:
            str: Full path to the written file.

        """
        filepath: Path = Path(filepath)
        filepath = filepath.with_suffix(".mmd")

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
