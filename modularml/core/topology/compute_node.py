from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any

from modularml.core.references.featureset_reference import FeatureSetReference
from modularml.core.topology.graph_node import GraphNode

if TYPE_CHECKING:
    from modularml.context.execution_context import ExecutionContext
    from modularml.core.data.batch import Batch
    from modularml.core.references.experiment_reference import ExperimentNodeReference
    from modularml.core.topology.node_shapes import NodeShapes
    from modularml.utils.data.data_format import DataFormat


class ComputeNode(GraphNode):
    """
    Abstract base class for computational nodes in a ModelGraph.

    This class extends `GraphNode` and represents any node that performs tensor
    computation or transformation. It defines a contract for:
    - Shape inference and tracking
    - Forward computation (e.g., neural network layers or merge operations)
    - Backend-specific logic (e.g., PyTorch, TensorFlow)
    - Construction/build logic (e.g., instantiating models or computing shapes)

    Subclasses may include model stages, merge operations, or custom layers.
    """

    def __init__(
        self,
        label: str,
        upstream_refs: ExperimentNodeReference
        | list[ExperimentNodeReference]
        | None = None,
        downstream_refs: ExperimentNodeReference
        | list[ExperimentNodeReference]
        | None = None,
        *,
        node_id: str | None = None,
        register: bool = True,
    ):
        """
        Initialize a ComputeNode.

        Args:
            label (str):
                Unique identifier for this node.
            upstream_refs (ExperimentNodeReference | list[ExperimentNodeReference] | None):
                References of upstream connections.
            downstream_refs (ExperimentNodeReference | list[ExperimentNodeReference] | None):
                References of downstream connections.
            node_id (str, optional):
                Used only for de-serialization.
            register (bool, optional):
                Used only for de-serialization.

        """
        super().__init__(
            label=label,
            upstream_refs=upstream_refs,
            downstream_refs=downstream_refs,
            node_id=node_id,
            register=register,
        )

    # ================================================
    # Representation
    # ================================================
    def _summary_rows(self) -> list[tuple]:
        return [
            ("label", self.label),
            ("upstream_refs", [f"{r!r}" for r in self._upstream_refs]),
            ("downstream_refs", [f"{r!r}" for r in self._downstream_refs]),
            ("input_shapes", [(k, str(v)) for k, v in self.input_shapes.items()]),
            ("output_shapes", [(k, str(v)) for k, v in self.output_shapes.items()]),
        ]

    def __repr__(self):
        return (
            f"ComputeNode(label='{self.label}', "
            f"upstream_refs={self._upstream_refs}, "
            f"downstream_refs={self._downstream_refs})"
        )

    def __str__(self):
        return f"ComputeNode('{self.label}')"

    # ================================================
    # GraphNode Interface
    # ================================================
    @property
    def allows_upstream_connections(self) -> bool:
        """
        Whether this node allows incoming (upstream) connections.

        Returns:
            bool: True if input connections are allowed.

        """
        return True

    @property
    def allows_downstream_connections(self) -> bool:
        """
        Whether this node allows outgoing (downstream) connections.

        Returns:
            bool: True if output connections are allowed.

        """
        return True

    # ================================================
    # ComputeNode Interface
    # ================================================
    @abstractmethod
    def infer_output_shape(
        self,
        input_shapes: dict[str, tuple[int, ...]],
    ) -> dict[str, tuple[int, ...]]:
        """
        Infer the output shapes of this node based on the given input shapes.

        Args:
            input_shapes (dict[str, tuple[int, ...]]):
                Input shapes feeding into this node.

        Returns:
            dict[str, tuple[int, ...]]: Inferred output shape(s).

        Raises:
            NotImplementedError: If the node cannot infer the shape without being built.

        """
        raise NotImplementedError

    def get_input_data(
        self,
        fmt: DataFormat,
        ctx: ExecutionContext,
    ) -> dict[ExperimentNodeReference, Batch]:
        """
        Retrieves Batch data for this ModelNode at the current execution step.

        Returns:
            dict[ExperimentNodeReference, Batch]:
                Batches keyed by each upstream_ref of this node.

        """
        input_data = {}
        for ref in self.get_upstream_refs():
            # Check if this reference pulls from FeatureSet
            inp_key = (self.node_id, ref)
            if inp_key in ctx.inputs:
                if not isinstance(ref, FeatureSetReference):
                    msg = "Invalid upstream reference in ExecutionContext.inputs."
                    raise TypeError(msg)

                # Get batch view from inputs
                bv = ctx.inputs[inp_key]

                # Materialize view to batch with specific columns
                batch = bv.materialize_batch(
                    fmt=fmt,
                    features=ref.features,
                    targets=ref.targets,
                    tags=ref.tags,
                )
                input_data[ref] = batch
                continue

            # Otherwise, get output of upstream node, and cast to this backend
            if ref.node_id in ctx.outputs:
                batch = ctx.outputs[ref.node_id]
                # Cast to format needed for this model (returns copy)
                input_data[ref] = batch.to_format(fmt=fmt)
                continue

            msg = (
                f"Failed to get input data for ComputeNode '{self.label}' upstream "
                f"reference: {ref}."
            )
            raise RuntimeError(msg)

        return input_data

    @abstractmethod
    def forward(self, inputs: Any) -> Any:
        """
        Perform forward computation through the node.

        Args:
            inputs (Any): Input tensor(s), compatible with the backend (e.g., PyTorch, TensorFlow).

        Returns:
            Any: Output tensor(s) after computation.

        """

    @abstractmethod
    def build(
        self,
        input_shapes: dict[str, tuple[int, ...]] | None = None,
        output_shapes: dict[str, tuple[int, ...]] | None = None,
        **kwargs,
    ):
        """
        Construct the internal logic of this node using the provided input and output shapes.

        Args:
            input_shapes (dict[str, tuple[int, ...]] | None):
                Shapes of data feeding into this node.
                Used to initialize models or internal transformation logic.
            output_shapes (dict[str, tuple[int, ...]] | None):
                Shapes of data expected to exit this node.
                May be used to constrain or validate internal shape inference.
            **kwargs: Additional key-word arguments specific to each subclass.

        Notes:
            - Nodes with only a single input/output can simplify this logic.
            - This method should initialize any backend-specific model components.
            - If model or shape construction fails, this method should raise an error.

        """

    @property
    @abstractmethod
    def is_built(self) -> bool:
        """
        Whether this node has been fully built (e.g., model instantiated).

        Returns:
            bool: True if the node is fully built and ready for use.

        """

    @property
    @abstractmethod
    def node_shapes(self) -> NodeShapes:
        """Input and output shapes expected by this node."""

    @property
    def input_shapes(self) -> dict[str, tuple[int, ...]]:
        """
        Shape of input data expected by this node.

        Returns:
            dict[str, tuple[int, ...]]: The expected input shape(s).

        """
        return self.node_shapes.input_shapes

    @property
    def output_shapes(self) -> dict[str, tuple[int, ...]]:
        """
        Shape of output data expected by this node.

        Returns:
            dict[str, tuple[int, ...]]: The expected output shape(s).

        """
        return self.node_shapes.output_shapes

    # ================================================
    # Configurable
    # ================================================
    def get_config(self) -> dict[str, Any]:
        cfg = super().get_config()
        return cfg

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        *,
        register: bool = True,
    ) -> ComputeNode:
        return cls(register=register, **config)
