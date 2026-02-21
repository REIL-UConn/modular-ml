from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, overload

import numpy as np

from modularml.core.data.batch import Batch
from modularml.core.data.featureset import FeatureSet
from modularml.core.data.sample_data import RoleData, SampleData
from modularml.core.data.schema_constants import (
    DOMAIN_FEATURES,
    DOMAIN_SAMPLE_UUIDS,
    DOMAIN_TAGS,
    DOMAIN_TARGETS,
)
from modularml.core.experiment.experiment_node import ExperimentNode
from modularml.core.references.experiment_reference import ExperimentNodeReference
from modularml.core.topology.compute_node import ComputeNode, TForward
from modularml.utils.data.conversion import convert_to_format
from modularml.utils.data.data_format import DataFormat, get_data_format_for_backend
from modularml.utils.data.formatting import ensure_list, ensure_tuple
from modularml.utils.logging import get_logger
from modularml.utils.logging.warnings import catch_warnings, warn
from modularml.utils.nn.backend import infer_backend

if TYPE_CHECKING:
    from modularml.utils.nn.backend import Backend

logger = get_logger("MergeNode")


class MergeNode(ComputeNode):
    """
    Base class for merging multiple upstream nodes in a model graph.

    Description:
        A MergeNode represents a node in the model graph that takes multiple upstream
        nodes and merges their outputs into a single output. This class serves as an
        abstract base for concrete merging strategies such as concatenation, averaging,
        or summation.

        Subclasses must implement the `apply_merge()` method, which defines how multiple
        input tensors are combined into a single output tensor.

    Example:
    ```python
        class MyConcatStage(MergeNode):
            def apply_merge(self, values): ...
    ```

    """

    def __init__(
        self,
        label: str,
        upstream_refs: list[ExperimentNode | ExperimentNodeReference],
        *,
        backend: Backend | None = None,
        node_id: str | None = None,
        register: bool = True,
    ):
        """
        Initialize a MergeNode.

        Args:
            label (str):
                Unique identifier for this node.
            upstream_refs (list[ExperimentNode | ExperimentNodeReference]):
                List of upstream node references from which inputs will be received.
                Accepts ExperimentNode instances (auto-converted to references) or
                ExperimentNodeReference objects directly.
            backend (Backend, optional):
                The backend to use for internally data merged.
                Defaults to None.
            node_id (str, optional):
                Used only for de-serialization.
            register (bool, optional):
                Used only for de-serialization.

        """
        # Clean up reference types
        ups_refs = []
        for ref in ensure_list(upstream_refs):
            if isinstance(ref, FeatureSet):
                dup_rep_warnings = False
                with catch_warnings() as cw:
                    ups_refs.append(ref.reference())
                    if cw.match("Multiple representations selected"):
                        dup_rep_warnings = True
                if dup_rep_warnings:
                    msg = (
                        "Setting a MergeNode `upstream_ref` with a FeatureSet will result in multiple "
                        "representations of the same column being combined into input/target tensors. "
                    )
                    hint = "Use `FeatureSet(...).reference(...)` is this is not intentional."
                    warn(msg, category=UserWarning, stacklevel=2, hints=hint)

            elif isinstance(ref, ExperimentNodeReference):
                ups_refs.append(ref)

            elif isinstance(ref, ExperimentNode):
                ups_refs.append(ref.reference())

            else:
                msg = f"`upstream_ref` must be of type ExperimentReference or ExperimentNode. Received: {type(ref)}."
                raise TypeError(msg)

        # Init ComputeNode
        super().__init__(
            label=label,
            upstream_refs=ups_refs,
            node_id=node_id,
            register=register,
        )

        # Init local attributes
        self._output_shape: tuple[int, ...] | None = None
        self._input_shapes: dict[ExperimentNodeReference, tuple[int, ...]] | None = None
        self._built = False
        self._backend: Backend | None = backend

    @property
    def input_shapes(self) -> dict[ExperimentNodeReference, tuple[int, ...]]:
        """
        Shape of data from each input expected by this MergeNode.

        Returns:
            dict[ExperimentNodeReference, tuple[int, ...]]:
                Input shapes keyed by upstream reference.

        Raises:
            RuntimeError: If node has not been built yet.

        """
        if not self.is_built:
            msg = (
                f"MergeNode '{self.label}' has not been built yet. "
                "Call `build()` first."
            )
            raise RuntimeError(msg)
        if self._input_shapes is None:
            msg = "Failed to get input shapes."
            raise RuntimeError(msg)

        return self._input_shapes

    @property
    def backend(self) -> Backend | None:
        """
        Backend associated with this MergeNode.

        Returns:
            Backend | None: The backend, or None if not set.

        """
        return self._backend

    @abstractmethod
    def apply_merge(
        self,
        values: list[Any],
        *,
        includes_batch_dim: bool = True,
        fmt: DataFormat | None = None,
        domain: str = DOMAIN_FEATURES,
    ) -> Any:
        """
        Merge logic to be implemented by subclasses.

        Args:
            values (list[Any]):
                A list of backend-specific tensors to be merged.
            includes_batch_dim (bool):
                Whether the input values have a batch dimension.
                Defaults to True.
            fmt (DataFormat | None):
                The data format expected for the returned tensor. If None,
                the data format will be inferred from the `backend` property.
                Defaults to None.
            domain (str, optional):
                The domain in which the data belongs. This allows for domain-specific
                merge logic (e.g., different concat axes for each domain).

        Returns:
            Any: Merged tensor.

        """

    # ================================================
    # ComputeNode Interface
    # ================================================
    @property
    def output_shape(self) -> tuple[int, ...]:
        """
        Shape of data produced by this MergeNode.

        Returns:
            tuple[int, ...]: Output shape after merging.

        Raises:
            RuntimeError: If node has not been built yet.

        """
        if not self.is_built:
            msg = (
                f"MergeNode '{self.label}' has not been built yet. "
                "Call `build()` first."
            )
            raise RuntimeError(msg)
        if self._output_shape is None:
            msg = "Failed to get output shape."
            raise RuntimeError(msg)

        return self._output_shape

    @property
    def is_built(self) -> bool:
        """
        Whether the MergeNode has been built (i.e., shape inference completed).

        Returns:
            bool: True if built, False otherwise.

        """
        return self._built

    def _infer_output_shape(
        self,
        *,
        input_shapes: dict[ExperimentNodeReference, tuple[int, ...]],
        includes_batch_dim: bool,
    ) -> tuple[int, ...]:
        """
        Infer the output shape based on input shapes from upstream nodes.

        Args:
            input_shapes (dict[ExperimentNodeReference, tuple[int, ...]]):
                Input shapes from upstream connections.
            includes_batch_dim (bool):
                Whether the shape values provided in `input_shapes` include the batch
                dimension or not.

        Returns:
            tuple[int, ...]:
                The inferred output shape. The returned tuple **does not** include
                the batch dimension.

        """
        # Get input tuples sorted by ref for reproducibility
        inp_shapes: list[tuple[int, ...]] = [
            v for _, v in sorted(input_shapes.items(), key=lambda x: str(x[0]))
        ]

        # Use apply_merge with dummy data to determine the output shape
        fmt = DataFormat.NUMPY
        if self._backend is not None:
            fmt = get_data_format_for_backend(backend=self._backend)

        dummy_data: list[Any] = [
            convert_to_format(
                data=np.ones(shape=x),
                fmt=fmt,
            )
            for x in inp_shapes
        ]

        merged = self.apply_merge(
            values=dummy_data,
            includes_batch_dim=includes_batch_dim,
            fmt=fmt,
            domain=DOMAIN_FEATURES,
        )

        # Return shape (without batch dimension)
        if hasattr(merged, "shape"):
            shape = ensure_tuple(merged.shape)
            if includes_batch_dim:
                return shape[1:]
            return shape

        msg = (
            "`apply_merge()` must return a tensor-like object with a `shape` "
            f"attribute. Got: {type(merged)}"
        )
        raise TypeError(msg)

    def _build_impl(
        self,
        *,
        input_shapes: dict[ExperimentNodeReference, tuple[int, ...]],
        includes_batch_dim: bool,
        output_shape: tuple[int, ...] | None = None,
        backend: Backend | None = None,
        **kwargs,  # noqa: ARG002
    ):
        """
        Construct the internal logic of this node using the provided input and output shapes.

        Args:
            input_shapes (dict[ExperimentNodeReference, tuple[int, ...]]):
                Shapes of data feeding into this node.
                Used to initialize internal merge logic.
            includes_batch_dim (bool):
                Whether the shape values provided in `input_shapes` include the batch
                dimension or not.
            output_shape (tuple[int, ...] | None):
                Shape of data expected to exit this node.
                May be used to constrain or validate internal shape inference.
            backend (Backend | None):
                The backend expected to be used by this MergeNode. If None,
                any backend can be used.
            **kwargs: Additional key-word arguments specific to each subclass.

        """
        # Set backend, if provided
        if backend is not None:
            self._backend = backend

        # Require input shapes
        if input_shapes is None:
            msg = f"MergeNode '{self.label}' requires input_shapes during build."
            raise ValueError(msg)

        # Infer output shape from merge logic
        out_shape = self._infer_output_shape(
            input_shapes=input_shapes,
            includes_batch_dim=includes_batch_dim,
        )

        # Validate output_shape (if given)
        if (output_shape is not None) and (output_shape != self._output_shape):
            msg = (
                "Merged output does not match expected output shape: "
                f"{out_shape} != {output_shape}."
            )
            raise ValueError(msg)

        # Set attributes
        self._input_shapes = {
            k: (v[1:] if includes_batch_dim else v) for k, v in input_shapes.items()
        }
        self._output_shape = out_shape
        self._built = True

    def _merge_sample_data(
        self,
        data: list[SampleData],
        fmt: DataFormat,
    ) -> SampleData:
        """
        Merge a list of SampleData objects across all domains.

        Args:
            data (list[SampleData]):
                Input SampleData objects to merge.
            fmt (DataFormat):
                Data format for features and targets. Tags and sample_uuids
                always use NUMPY.

        Returns:
            SampleData: Merged output.

        """
        merged_attrs: dict[str, Any] = {}
        for attr, attr_fmt in zip(
            [DOMAIN_FEATURES, DOMAIN_TARGETS, DOMAIN_TAGS, DOMAIN_SAMPLE_UUIDS],
            (fmt, fmt, DataFormat.NUMPY, DataFormat.NUMPY),
            strict=True,
        ):
            # Ensure we have data for the given attribute
            have_vals: list[bool] = [getattr(d, attr) is not None for d in data]

            # If none have data, just skip
            if not any(have_vals):
                continue

            # If only some do, throw warning and skip
            if not all(have_vals):
                msg = (
                    f"Not all inputs have data for the `{attr}` attribute. "
                    f"The merged results will not contain `{attr}` data."
                )
                warn(msg, stacklevel=2)
                continue

            # For attributes with data, merge
            merged_attrs[attr] = self.apply_merge(
                values=[getattr(d, attr) for d in data],
                includes_batch_dim=True,
                fmt=attr_fmt,
                domain=attr,
            )

        # Return a new, merged SampleData instance
        return SampleData(
            data=merged_attrs,
            kind="output",
        )

    def _merge_role_data(
        self,
        rds: list[RoleData],
        fmt: DataFormat,
    ) -> RoleData:
        """
        Merge a list of RoleData objects.

        Args:
            rds (list[RoleData]):
                Input RoleData objects to merge.
            fmt (DataFormat):
                Data format for features and targets.

        Returns:
            RoleData: Merged output.

        """
        ref = rds[0]
        # Ensure all inputs have the same roles
        for rd in rds[1:]:
            if rd.available_roles != ref.available_roles:
                msg = (
                    "All inputs of type RoleData must have the same roles: "
                    f"{ref.available_roles} != {rd.available_roles}."
                )
                raise ValueError(msg)

        # Get list of sample data for each role
        grouped_role_values: dict[str, list[SampleData]] = {}
        for role in ref.available_roles:
            grouped_role_values[role] = [rd.get_data(role=role) for rd in rds]

        # Merge SampleData within each role
        out = {
            k: self._merge_sample_data(v, fmt=fmt)
            for k, v in grouped_role_values.items()
        }
        return RoleData(data=out)

    @overload
    def merge(self, batches: list[Batch], **kwargs) -> Batch: ...
    @overload
    def merge(self, roles: list[RoleData], **kwargs) -> RoleData: ...
    @overload
    def merge(self, data: list[SampleData], **kwargs) -> SampleData: ...
    def merge(
        self,
        x: list[TForward],
        **kwargs,  # noqa: ARG002
    ) -> TForward:
        """
        Performs a forward pass by merging all upstream inputs.

        Args:
            x (list[TForward]):
                Input data from upstream nodes. All inputs must have the same
                value data type.
            **kwargs:
                Additional keyword arguments passed to `apply_merge`.

        Returns:
            TForward:
                Merged output matching the input type (SampleData, RoleData, or Batch).

        """
        # Ensure built
        if not self.is_built:
            # Try to auto-build from inputs (assume all inputs have batch dim)
            first = next(iter(x))
            try:
                inferred_backend = infer_backend(first)
            except Exception as e:  # noqa: BLE001
                msg = f"Failed to infer backend for {first}: {e}"
                logger.debug(msg)
                inferred_backend = None
            try:
                # Assume order of inputs match sorted upstream refs
                sorted_refs = sorted(self.get_upstream_refs(), key=str)
                self._build_impl(
                    input_shapes={
                        k: v.shapes.features_shape
                        for k, v in zip(sorted_refs, x, strict=True)
                    },
                    includes_batch_dim=True,
                    backend=self.backend or inferred_backend,
                )
            except Exception as e:
                msg = (
                    f"MergeNode '{self.label}' has not been built yet. "
                    "Call `build()` first."
                )
                raise RuntimeError(msg) from e

        # Determine merge format (for features and targets)
        if self.backend is not None:
            fmt = get_data_format_for_backend(backend=self.backend)
        else:
            fmt = DataFormat.NUMPY

        # Determine input type
        first = x[0]
        for v in x[1:]:
            if not isinstance(v, type(first)):
                msg = (
                    "All inputs must have the same data type: "
                    f"{type(v)} != {type(first)}."
                )
                raise TypeError(msg)

        # Merge based on data type
        if isinstance(first, SampleData):
            return self._merge_sample_data(x, fmt=fmt)

        if isinstance(first, RoleData):
            return self._merge_role_data(x, fmt=fmt)

        if isinstance(first, Batch):
            # Get merged role data
            out_rd = self._merge_role_data([b.role_data for b in x], fmt=fmt)

            # Merge role_weights - take mean across sample axis
            m_weights = {}
            for role in first.available_roles:
                # Get all weights for this role
                role_weights = np.vstack(
                    [b.role_weights[role] for b in x],
                )
                if role_weights.ndim != 2:
                    msg = (
                        "Input `role_weights` do not contain single dimensional arrays."
                    )
                    raise RuntimeError(msg)

                # Take mean weight
                avg_weights = np.average(role_weights, axis=0)
                if avg_weights.shape[0] != first.batch_size:
                    msg = "Failed to average role weights across batches."
                    raise RuntimeError(msg)

                m_weights[role] = avg_weights

            # Merge role_weights - take logical and across sample axis
            m_masks = {}
            for role in first.available_roles:
                # Get all masks for this role
                role_masks = [b.role_masks[role] for b in x]

                # Take logical and
                and_masks = np.logical_and(*role_masks).astype(dtype=int)
                if and_masks.shape[0] != first.batch_size:
                    msg = "Failed to average role masks across batches."
                    raise RuntimeError(msg)

                m_masks[role] = and_masks

            return Batch(
                batch_size=first.batch_size,
                role_data=out_rd,
                shapes=out_rd.shapes,
                role_weights=m_weights,
                role_masks=m_masks,
            )

        msg = f"Input must be of type SampleData or RoleData or Batch. Received: {type(x)}"
        raise TypeError(msg)

    def _forward_impl(
        self,
        *,
        inputs: dict[ExperimentNodeReference, TForward],
        **kwargs,  # noqa: ARG002
    ) -> TForward:
        """
        Perform forward pass by merging all upstream inputs.

        Args:
            inputs (dict[ExperimentNodeReference, TForward]):
                Input data from upstream nodes, keyed by reference.
                All inputs must have the same value data type.
            **kwargs:
                Additional keyword arguments passed to `apply_merge`.

        Returns:
            TForward:
                Merged output matching the input type (SampleData, RoleData, or Batch).

        """
        # Ensure built
        if not self.is_built:
            # Try to auto-build from inputs (assume all inputs have batch dim)
            first = next(iter(inputs.values()))
            try:
                inferred_backend = infer_backend(first)
            except Exception as e:  # noqa: BLE001
                msg = f"Failed to infer backend for {first}: {e}"
                logger.debug(msg)
                inferred_backend = None
            try:
                self._build_impl(
                    input_shapes={
                        k: v.shapes.features_shape for k, v in inputs.items()
                    },
                    includes_batch_dim=True,
                    backend=self.backend or inferred_backend,
                )
            except Exception as e:
                msg = (
                    f"MergeNode '{self.label}' has not been built yet. "
                    "Call `build()` first."
                )
                raise RuntimeError(msg) from e

        # To ensure reproducible order, we use the same order that upstream refs
        # we added to this node
        sorted_values = [inputs[ref] for ref in self.get_upstream_refs()]

        # Determine input type
        first = sorted_values[0]
        for v in sorted_values[1:]:
            if not isinstance(v, type(first)):
                msg = (
                    "All inputs must have the same data type: "
                    f"{type(v)} != {type(first)}."
                )
                raise TypeError(msg)

        # Merge data
        return self.merge(sorted_values)

    # ================================================
    # Representation
    # ================================================
    def __repr__(self):
        return (
            f"MergeNode(label='{self.label}', "
            f"upstream_refs={self._upstream_refs}, "
            f"downstream_refs={self._downstream_refs}, "
            f"backend={self._backend})"
        )

    def __str__(self):
        return f"MergeNode('{self.label}')"

    # ================================================
    # Configurable
    # ================================================
    def get_config(self) -> dict[str, Any]:
        """
        Return configuration details required to reconstruct this callback.

        Returns:
            dict[str, Any]:
                Configuration used to reconstruct the callback.
                Keys must be strings.

        """
        cfg = super().get_config()
        cfg.update(
            {
                "output_shape": self._output_shape,
                "input_shapes": self._input_shapes,
                "is_built": self._built,
                "backend": self._backend,
            },
        )
        return cfg

    @classmethod
    def from_config(cls, config: dict) -> MergeNode:
        """
        Construct a callback from a configuration dictionary.

        Args:
            config (dict[str, Any]):
                Configuration details. Keys must be strings.

        Returns:
            Callback: Reconstructed callback.

        """
        cb_cls_name = config.get("merge_node_type")
        if cb_cls_name == "ConcatNode":
            from modularml.core.topology.merge_nodes.concat_node import ConcatNode

            return ConcatNode.from_config(config=config)

        msg = (
            f"Unsupported MergeNode class for parent class construction: {cb_cls_name}."
        )
        raise ValueError(msg)
