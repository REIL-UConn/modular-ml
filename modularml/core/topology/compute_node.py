"""Abstract compute node definitions used in ModularML graphs."""

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
    Abstract computational node within a :class:`ModelGraph`.

    Attributes:
        input_shapes (dict[str, tuple[int, ...]]): Registered input shape
            metadata keyed by reference label.
        output_shapes (dict[str, tuple[int, ...]]): Registered output
            shape metadata keyed by reference label.

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
        Initialize the compute node with upstream/downstream refs.

        Args:
            label (str):
                Unique identifier for the node.
            upstream_refs (ExperimentNodeReference | list[ExperimentNodeReference] | None):
                Upstream references feeding the node.
            downstream_refs (ExperimentNodeReference | list[ExperimentNodeReference] | None):
                Downstream references that this node feeds.
            node_id (str | None):
                Optional ID used during deserialization.
            register (bool):
                Whether to register the node when created.

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
        """
        Return summary rows including shape metadata.

        Returns:
            list[tuple]: Summary name/value pairs for display helpers.

        """
        return [
            ("label", self.label),
            ("upstream_refs", [f"{r!r}" for r in self._upstream_refs]),
            ("downstream_refs", [f"{r!r}" for r in self._downstream_refs]),
            ("input_shapes", [(k, str(v)) for k, v in self.input_shapes.items()]),
            ("output_shapes", [(k, str(v)) for k, v in self.output_shapes.items()]),
        ]

    def __repr__(self):
        """
        Return developer-friendly representation for debugging.

        Returns:
            str: String describing critical connection metadata.

        """
        return (
            f"ComputeNode(label='{self.label}', "
            f"upstream_refs={self._upstream_refs}, "
            f"downstream_refs={self._downstream_refs})"
        )

    def __str__(self):
        """
        Return the node label for readable output.

        Returns:
            str: Readable label for :class:`ComputeNode`.

        """
        return f"ComputeNode('{self.label}')"

    # ================================================
    # GraphNode Interface
    # ================================================
    @property
    def allows_upstream_connections(self) -> bool:
        """
        Return True because compute nodes accept upstream data.

        Returns:
            bool: True because compute nodes always have inputs.

        """
        return True

    @property
    def allows_downstream_connections(self) -> bool:
        """
        Return True because compute nodes emit downstream data.

        Returns:
            bool: True because compute nodes always emit outputs.

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
        Resolve upstream data for the current execution step.

        Args:
            inputs (dict[tuple[str, FeatureSetReference], TForward]):
                Mapping of :class:`FeatureSetReference` instances produced
                by samplers.
            outputs (dict[str, TForward]): Cached outputs from upstream
                compute nodes.
            fmt (DataFormat): Output format requested when materializing
                :class:`BatchView` instances.

        Returns:
            dict[ExperimentNodeReference, TForward]: Data keyed by
                upstream references.

        Raises:
            TypeError: If :class:`FeatureSetReference` values are invalid.
            RuntimeError: If upstream data cannot be located for a
                reference.

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
        Perform the forward computation for this node.

        Args:
            inputs (dict[ExperimentNodeReference, TForward]):
                Input data keyed by upstream reference.
            **kwargs: Additional subclass-specific keyword arguments.

        Returns:
            TForward: Output of the computation.

        """
        return self._forward_impl(inputs=inputs, **kwargs)

    @abstractmethod
    def _forward_impl(
        self,
        *,
        inputs: dict[ExperimentNodeReference, TForward],
        **kwargs,
    ) -> TForward:
        """
        Implement the backend-specific forward logic.

        Args:
            inputs (dict[ExperimentNodeReference, TForward]):
                Inputs resolved through :meth:`get_input_data`.
            **kwargs: Backend-specific options required by subclasses.

        Returns:
            TForward: Forward result produced by the node implementation.

        """
        ...

    def build(
        self,
        input_shapes: dict[ExperimentNodeReference, tuple[int, ...]] | None = None,
        output_shape: tuple[int, ...] | None = None,
        **kwargs,
    ):
        """
        Build entry point used by :class:`ModelGraph`.

        Args:
            input_shapes (dict[ExperimentNodeReference, tuple[int, ...]] | None):
                Shapes of upstream tensors.
            output_shape (tuple[int, ...] | None):
                Expected downstream tensor shape.
            **kwargs: Implementation-specific keyword arguments.

        """
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
        Construct node internals using provided shapes.

        Args:
            input_shapes (dict[ExperimentNodeReference, tuple[int, ...]] | None):
                Shapes of inputs feeding this node.
            output_shape (tuple[int, ...] | None):
                Expected shape of data produced by this node.
            **kwargs: Additional keyword arguments specific to each
                subclass.

        Raises:
            RuntimeError: Implementations should raise errors when model
                or shape construction fails.

        """

    @property
    @abstractmethod
    def is_built(self) -> bool:
        """
        Whether this node has been fully built.

        Returns:
            bool: True if the internal backend model is ready to use.

        """

    @property
    @abstractmethod
    def output_shape(self) -> tuple[int, ...]:
        """
        Shape of data produced by this node.

        Returns:
            tuple[int, ...]: Output tensor shape.

        """

    # ================================================
    # Configurable
    # ================================================
    def get_config(self) -> dict[str, Any]:
        """
        Return the serialized configuration for this node.

        Returns:
            dict[str, Any]: Configuration suitable for :meth:`from_config`.

        """
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
        """
        Recreate a :class:`ComputeNode` from serialized config.

        Args:
            config (dict[str, Any]):
                Serialized node data produced by :meth:`get_config`.
            register (bool):
                Whether to register inside the active :class:`ExperimentContext`.

        Returns:
            ComputeNode: Reconstructed node instance.

        Raises:
            ValueError: If the config lacks the proper node type marker.

        """
        if (
            "graph_node_type" not in config
            or config["graph_node_type"] != "ComputeNode"
        ):
            raise ValueError("Invalid config data for ComputeNode.")

        return cls(register=register, **config)
