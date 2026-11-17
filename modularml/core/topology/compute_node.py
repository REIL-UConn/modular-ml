"""Abstract compute node definitions used in ModularML graphs."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, TypeVar

from modularml.core.data.batch import Batch
from modularml.core.data.batch_view import BatchView
from modularml.core.data.sample_data import RoleData, SampleData
from modularml.core.data.schema_constants import DOMAIN_FEATURES, DOMAIN_TARGETS
from modularml.core.references.featureset_reference import FeatureSetReference
from modularml.core.topology.graph_node import GraphNode
from modularml.utils.data.data_format import DataFormat, get_data_format_for_backend
from modularml.utils.environment.optional_imports import check_torch
from modularml.utils.nn.accelerator import Accelerator
from modularml.utils.nn.backend import Backend, infer_backend

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

    # ================================================
    # Backend / Accelerator Helpers
    # ================================================
    @staticmethod
    def _normalize_accelerator(
        accelerator: Accelerator | str | None,
    ) -> Accelerator | None:
        """Normalize accelerator configuration values."""
        if accelerator is None:
            return None
        if isinstance(accelerator, Accelerator):
            return accelerator
        if isinstance(accelerator, str):
            return Accelerator(device=accelerator)
        msg = f"Accelerator must be Accelerator, str, or None. Received: {type(accelerator)}."
        raise TypeError(msg)

    @staticmethod
    def _infer_data_backend(obj: Any) -> Backend | None:
        """Infer backend from a runtime tensor-like payload."""
        if isinstance(obj, SampleData):
            if obj.features is not None:
                return infer_backend(obj.features)
            if obj.targets is not None:
                return infer_backend(obj.targets)
            return None

        if isinstance(obj, RoleData):
            sample_data = obj.get_data(role=obj.available_roles[0])
            if sample_data.features is not None:
                return infer_backend(sample_data.features)
            if sample_data.targets is not None:
                return infer_backend(sample_data.targets)
            return None

        if isinstance(obj, Batch):
            if len(obj.role_data.available_roles) == 0:
                return None
            sample_data = obj.role_data.get_data(role=obj.role_data.available_roles[0])
            if sample_data.features is not None:
                return infer_backend(sample_data.features)
            if sample_data.targets is not None:
                return infer_backend(sample_data.targets)
            return None

        return infer_backend(obj)

    @staticmethod
    def _infer_data_format(obj: Any) -> DataFormat | None:
        """Infer DataFormat from payload backend."""
        backend = ComputeNode._infer_data_backend(obj)
        if backend in (None, Backend.NONE):
            return None
        try:
            return get_data_format_for_backend(backend)
        except ValueError:
            return None

    @staticmethod
    def _move_torch_to_device_if_needed(
        obj: Any,
        accelerator: Accelerator,
    ) -> Any:
        """
        Move tensor-like values to device, if requested.

        Moved only if they are already torch tensors and on a different device.
        """
        if obj is None:
            return obj

        target_device = accelerator.torch_device_str()
        torch = check_torch()
        if torch is None:
            msg = "Accelerator requested for torch but torch is not installed."
            raise ImportError(msg)

        if isinstance(obj, SampleData):
            changed = False
            new_data: dict[str, Any] = {}
            for k, v in obj.data.items():
                if k not in [DOMAIN_FEATURES, DOMAIN_TARGETS]:
                    new_data[k] = v
                    continue
                if isinstance(v, torch.Tensor) and str(v.device) != target_device:
                    changed = True
                    new_data[k] = accelerator.move_torch_tensor(v)
                else:
                    new_data[k] = v
            if changed:
                return SampleData(data=new_data, kind=obj._kind)
            return obj

        if isinstance(obj, RoleData):
            changed = False
            new_role_data: dict[str, SampleData] = {}
            for role, sample_data in obj._data.items():
                moved_sample_data = ComputeNode._move_torch_to_device_if_needed(
                    obj=sample_data,
                    accelerator=accelerator,
                )
                changed = changed or (moved_sample_data is not sample_data)
                new_role_data[role] = moved_sample_data
            if changed:
                return RoleData(data=new_role_data)
            return obj

        if isinstance(obj, Batch):
            moved_role_data = ComputeNode._move_torch_to_device_if_needed(
                obj=obj.role_data,
                accelerator=accelerator,
            )
            if moved_role_data is obj.role_data:
                return obj
            return Batch(
                batch_size=obj.batch_size,
                role_data=moved_role_data,
                shapes=obj.shapes,
                role_weights=dict(obj.role_weights),
                role_masks=dict(obj.role_masks),
            )

        if isinstance(obj, torch.Tensor):
            if str(obj.device) == target_device:
                return obj
            return accelerator.move_torch_tensor(obj)

        return obj

    def _coerce_data_format_and_acceleration(
        self,
        data: Any,
        *,
        fmt: DataFormat,
        accelerator: Accelerator | str | None = None,
        backend: Backend | None = None,
    ) -> Any:
        """
        Normalize a data payload to the requested format and accelerator placement.

        Conversion/mutation is avoided when the payload is already in the target
        format and device position is already correct.

        Args:
            data (Any): Input data payload to normalize.
            fmt (DataFormat): Target tensor format (e.g. :attr:`DataFormat.TORCH`).
            accelerator (Accelerator | str | None, optional):
                Hardware accelerator for device placement. PyTorch tensors are
                moved to the target device only when this is set and ``backend``
                is :attr:`Backend.TORCH`. Defaults to ``None``.
            backend (Backend | None, optional):
                Backend of the node consuming this data. Used to gate whether
                device movement is attempted. Defaults to ``None``.

        Returns:
            Any: Data in the requested format and on the correct device.

        """
        acc = self._normalize_accelerator(accelerator=accelerator)

        # Convert tensor-like format only if needed.
        current_fmt = self._infer_data_format(data)
        if current_fmt is None or current_fmt == fmt:
            converted = data
        else:
            converted = data.to_format(fmt=fmt) if hasattr(data, "to_format") else data

        if acc is not None and backend == Backend.JAX:
            from dataclasses import replace

            from modularml.utils.environment.optional_imports import ensure_jax

            jax = ensure_jax()
            device = acc.jax_device()

            def move(obj):
                if isinstance(obj, SampleData):
                    return SampleData(
                        data={
                            k: jax.device_put(v, device)
                            if k in (DOMAIN_FEATURES, DOMAIN_TARGETS) and v is not None
                            else v
                            for k, v in obj.data.items()
                        },
                        kind=obj._kind,
                    )
                if isinstance(obj, RoleData):
                    return RoleData({role: move(sd) for role, sd in obj.items()})
                if isinstance(obj, Batch):
                    return replace(obj, role_data=move(obj.role_data))
                if isinstance(obj, (jax.Array, jax.core.Tracer)):
                    return jax.device_put(obj, device)
                return obj

            return move(converted)

        # Move torch tensors to the requested device only when needed.
        if (
            acc is not None
            and backend == Backend.TORCH
            and self._infer_data_format(converted) == DataFormat.TORCH
        ):
            return self._move_torch_to_device_if_needed(converted, acc)
        return converted

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
        accelerator: Accelerator | str | None = None,
        backend: Backend | None = None,
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
            accelerator (Accelerator | str | None, optional):
                Hardware accelerator for device placement. When provided,
                PyTorch tensors are moved to the target device after format
                conversion. Defaults to ``None``.
            backend (Backend | None, optional):
                Backend hint used to gate device movement; tensors are only
                moved when ``backend`` is :attr:`Backend.TORCH`. Defaults to
                ``None``.

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
                    input_data[ref] = self._coerce_data_format_and_acceleration(
                        data=batch,
                        fmt=fmt,
                        accelerator=accelerator,
                        backend=backend,
                    )

                # Otherwise, keep as is
                else:
                    input_data[ref] = self._coerce_data_format_and_acceleration(
                        data=data,
                        fmt=fmt,
                        accelerator=accelerator,
                        backend=backend,
                    )

            # Otherwise, get output of upstream node, and cast to this backend
            elif ref.node_id in outputs:
                data = outputs[ref.node_id]
                input_data[ref] = self._coerce_data_format_and_acceleration(
                    data=data,
                    fmt=fmt,
                    accelerator=accelerator,
                    backend=backend,
                )

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
