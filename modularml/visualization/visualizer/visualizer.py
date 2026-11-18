"""Mermaid-based visualization for ModularML core objects."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from modularml.visualization.visualizer.internal_representation import GraphIR

if TYPE_CHECKING:
    from modularml.visualization.visualizer.internal_representation import (
        MermaidSyntax,
    )
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

    def get_mermaid_str(self, syntax: MermaidSyntax = "v11") -> str:
        """
        Generates the Mermaid string from the internal graph object.

        Args:
            syntax (MermaidSyntax): Mermaid syntax to emit. "v11" (default)
                uses `id@{ ... }` node declarations and edge ids, which only
                render on Mermaid >= 11.5. Use "classic" for output that renders
                on all Mermaid versions from 9.x onward (e.g., GitHub-rendered
                Markdown, whose bundled Mermaid version is not published).
                "classic" omits the dash animation on edges.

        Returns:
            str: Mermaid diagram as string.

        """
        return self._build_graph().to_mermaid(syntax=syntax)

    def mermaid_to_mmd(self, filepath: str, syntax: MermaidSyntax = "v11") -> str:
        """
        Writes the Mermaid string to a `.mmd` markdown file.

        Args:
            filepath (str or Path): Path to output file (suffix `.mmd` auto-added).
            syntax (MermaidSyntax): Mermaid syntax to emit ("v11" or "classic").
                See :meth:`get_mermaid_str`. Defaults to "v11".

        Returns:
            str: Full path to the written file.

        """
        filepath: Path = Path(filepath)
        filepath = filepath.with_suffix(".mmd")

        with Path.open(filepath, "w", encoding="utf-8") as f:
            f.write(self.get_mermaid_str(syntax=syntax))

        return str(filepath)

    def display_mermaid(self):
        """
        Displays the Mermaid diagram inline in Jupyter Notebook.

        Requires Mermaid support in the notebook viewer.
        """
        try:
            from IPython.display import Markdown, display
        except ImportError as exc:
            msg = (
                "Displaying a diagram inline needs IPython (a Jupyter environment). "
                "Use get_mermaid_str() or the Mermaid export methods instead."
            )
            raise ImportError(msg) from exc
        mermaid_str = self.get_mermaid_str()
        disp_str = f"```{{mermaid}}\n{mermaid_str}\n```"
        display(Markdown(disp_str))

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
