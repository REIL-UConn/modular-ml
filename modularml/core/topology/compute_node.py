from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, TypeVar

from modularml.core.data.batch import Batch
from modularml.core.data.batch_view import BatchView
from modularml.core.data.sample_data import RoleData, SampleData
from modularml.core.references.featureset_reference import FeatureSetReference
from modularml.core.topology.graph_node import GraphNode
from modularml.utils.data.data_format import DataFormat

if TYPE_CHECKING:
    from modularml.core.references.experiment_reference import ExperimentNodeReference

TForward = TypeVar("TForward", Batch, RoleData, SampleData)


class ComputeNode(GraphNode):
    """
    Abstract base class for computational nodes in a ModelGraph.

    Implements:
        - Forwardable[TForward]
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
    def get_input_data(
        self,
        inputs: dict[tuple[str, FeatureSetReference], TForward],
        outputs: dict[str, TForward],
        *,
        fmt: DataFormat = DataFormat.NUMPY,
    ) -> dict[ExperimentNodeReference, TForward]:
        """
        Retrieves data for this ModelNode at the current execution step.

        Returns:
            dict[ExperimentNodeReference, TForward]:
                Forwardable data keyed by each upstream_ref of this node.

        """
        input_data = {}
        for ref in self.get_upstream_refs():
            # Check if this reference pulls from FeatureSet
            inp_key = (self.node_id, ref)
            if inp_key in inputs:
                if not isinstance(ref, FeatureSetReference):
                    msg = "Invalid upstream reference in `inputs`."
                    raise TypeError(msg)

                data: SampleData | RoleData | BatchView = inputs[inp_key]

                # If view, materialize
                if isinstance(data, BatchView):
                    # Materialize view to batch with specific columns
                    batch = data.materialize_batch(
                        fmt=fmt,
                        features=ref.features,
                        targets=ref.targets,
                        tags=ref.tags,
                    )
                    input_data[ref] = batch

                # Otherwise, keep as is
                else:
                    input_data[ref] = data

            # Otherwise, get output of upstream node, and cast to this backend
            elif ref.node_id in outputs:
                data = outputs[ref.node_id]
                if hasattr(data, "to_format"):
                    # Cast to format needed for this model (returns copy)
                    input_data[ref] = data.to_format(fmt=fmt)
                else:
                    input_data[ref] = data

            else:
                msg = (
                    f"Failed to get input data for ComputeNode '{self.label}' upstream "
                    f"reference: {ref}."
                )
                raise RuntimeError(msg)

        return input_data

    def forward(
        self,
        inputs: dict[ExperimentNodeReference, TForward],
        **kwargs,
    ) -> TForward:
        """
        Perform forward computation through the node.

        Args:
            inputs (dict[ExperimentNodeReference, TForward]):
                Input data to perform a forward pass on.
            **kwargs: Additional key-word arguments specific to each subclass.

        Returns:
            TForward:
                The output data matches the type of the input values.
                E.g, dict[ref, SampleData] returns SampleData

        """
        return self._forward_impl(inputs=inputs, **kwargs)

    @abstractmethod
    def _forward_impl(
        self,
        *,
        inputs: dict[ExperimentNodeReference, TForward],
        **kwargs,
    ) -> TForward: ...

    def build(
        self,
        input_shapes: dict[ExperimentNodeReference, tuple[int, ...]] | None = None,
        output_shape: tuple[int, ...] | None = None,
        **kwargs,
    ):
        """Build entry point used by ModelGraph."""
        self._build_impl(
            input_shapes=input_shapes,
            output_shape=output_shape,
            **kwargs,
        )

    @abstractmethod
    def _build_impl(
        self,
        input_shapes: dict[ExperimentNodeReference, tuple[int, ...]] | None = None,
        output_shape: tuple[int, ...] | None = None,
        **kwargs,
    ):
        """
        Construct the internal logic of this node using the provided input and output shapes.

        Args:
            input_shapes (dict[ExperimentNodeReference, tuple[int, ...]] | None):
                Shapes of data feeding into this node.
                Used to initialize models or internal transformation logic.
            output_shape (tuple[int, ...] | None):
                Shape of data expected to exit this node.
                May be used to constrain or validate internal shape inference.
            **kwargs: Additional key-word arguments specific to each subclass.

        Notes:
            - Nodes with only a single input can simplify this logic.
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
    def output_shape(self) -> tuple[int, ...]:
        """Shape of data produced by this node."""

    # ================================================
    # Configurable
    # ================================================
    def get_config(self) -> dict[str, Any]:
        cfg = super().get_config()
        cfg["graph_node_type"] = "ComputeNode"
        return cfg

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        *,
        register: bool = True,
    ) -> ComputeNode:
        if (
            "graph_node_type" not in config
            or config["graph_node_type"] != "ComputeNode"
        ):
            raise ValueError("Invalid config data for ComputeNode.")

        return cls(register=register, **config)
