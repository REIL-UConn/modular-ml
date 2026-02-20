from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from modularml.core.data.schema_constants import (
    DOMAIN_FEATURES,
    DOMAIN_SAMPLE_UUIDS,
    DOMAIN_TAGS,
    DOMAIN_TARGETS,
)
from modularml.utils.data.conversion import convert_to_format
from modularml.utils.data.data_format import (
    _TENSORLIKE_FORMATS,
    DataFormat,
    format_is_tensorlike,
    get_data_format_for_backend,
    normalize_format,
)
from modularml.utils.environment.optional_imports import ensure_tensorflow, ensure_torch
from modularml.utils.nn.padding import PadMode, map_pad_mode_to_backend

from .merge_node import MergeNode

if TYPE_CHECKING:
    from modularml.core.experiment.experiment_node import ExperimentNode
    from modularml.core.references.experiment_reference import ExperimentNodeReference


class ConcatNode(MergeNode):
    """
    A merge stage that concatenates multiple inputs along a specified axis.

    Description:
        This stage merges tensors by concatenating them along a specified axis. It
        supports automatic padding of non-concat dimensions to align shapes, allowing
        for flexible merging even when inputs vary in size. Padding behavior can be
        controlled via mode (e.g., 'constant', 'reflect', 'replicate') and value.

    Attributes:
        label (str):
            Unique identifier for this node.
        upstream_refs (list[ExperimentNode | ExperimentNodeReference]):
            Upstream node references from which inputs will be received.
        axis (int):
            The axis along which to concatenate inputs.
        pad_inputs (bool, optional):
            Whether to pad inputs before merging. Defaults to False.
        pad_mode (PadMode, optional):
            Padding mode ('constant', 'reflect', 'replicate', etc.).
            Defaults to 'constant'.
        pad_value (float, optional):
            Value to use for constant padding. Defaults to 0.0.

    Example:
    ```python
        fs_ref = FeatureSet(...).reference(columns=...)
        mn = ModelNode(...)
        stage = ConcatNode(
            label="merge",
            upstream_refs=[fs_ref, mn],
            axis=1,
            pad_inputs=True,
            pad_mode="constant",
            pad_value=0.0,
        )
    ```

    """

    def __init__(
        self,
        label: str,
        upstream_refs: list[ExperimentNode | ExperimentNodeReference],
        concat_axis: int = 0,
        *,
        concat_axis_targets: int = -1,
        concat_axis_tags: int = -1,
        pad_inputs: bool = False,
        pad_mode: str | PadMode = "constant",
        pad_value: float = 0.0,
        node_id: str | None = None,
        register: bool = True,
    ):
        """
        Initialize a ConcatNode.

        Args:
            label (str):
                Unique identifier for this node.

            upstream_refs (list[ExperimentNode | ExperimentNodeReference]):
                Upstream node references from which inputs will be received.

            concat_axis (int):
                The axis along which to concatenate inputs. Does not include the batch
                dimension. That is, for shape (with batch) of (32,1,16), axis=0 refers
                to "1". The example given below omit the batch dimension.
                * axis=0: concat along features. E.g, (1,16) + (1,16) -> (2,16))
                * axis>1: concat within features. E.g, (1,16,16) + (1,16,8) -> (1,16,24))
                    All data must have at least `concat_axis` dimensions.
                * axis=-1: concat along last axis of all data. E.g, (1, 16, 16) +
                    (1, 16, 8) -> (1, 16, 24).

            concat_axis_targets (int, optional):
                If applying concatenation to a domain-based data structure (e.g.,
                SampleData, RoleData, or Batch), concantenation can be applied to
                each domain with different concatenation axes.
                Typically, targets are just concatenated along their last axis (-1).

            concat_axis_tags (int, optional):
                Similarly to `concat_axis_targets`, an axis to concatenate the "tags"
                domains can be specified. Defaults to -1.

            pad_inputs (bool, optional):
                Whether to pad inputs before merging. Defaults to False.

            pad_mode (PadMode, optional):
                Padding mode; one of {"constant", "reflect", "replicate", "circular"}.
                Defaults to "constant".

            pad_value (float, optional):
                Value to use for constant padding. Defaults to 0.0.

            node_id (str, optional):
                Used only for de-serialization.

            register (bool, optional):
                Used only for de-serialization.

        """
        super().__init__(
            label=label,
            upstream_refs=upstream_refs,
            node_id=node_id,
            register=register,
        )
        self.concat_axis = int(concat_axis)
        self.target_axis = int(concat_axis_targets)
        self.tags_axis = int(concat_axis_tags)
        self.pad_inputs = bool(pad_inputs)
        self.pad_mode = pad_mode if isinstance(pad_mode, PadMode) else PadMode(pad_mode)
        self.pad_value = pad_value

        if self.pad_mode not in [PadMode.CONSTANT]:
            msg = f"Pad mode is not supported yet: {self.pad_mode}"
            raise NotImplementedError(msg)

    def _pad_inputs(
        self,
        values: list[Any],
        concat_axis: int,
        fmt: DataFormat | None = None,
    ) -> list[Any]:
        """
        Pad all inputs along non-concat dimensions to match the largest shape.

        Description:
            This method applies backend-specific padding logic to ensure that
            all tensors have the same shape (except for the concat axis) before
            concatenation.

        Args:
            values (list[Any]):
                List of tensors to be padded.
            concat_axis (int):
                The axis to concatenate along, relative to the actual tensors
                in `values`.
            fmt (DataFormat | None):
                The data format expected for the returned tensor. If None,
                the data format will be inferred from the `backend` property.
                Defaults to None.

        Returns:
            list[Any]: Padded tensors.

        Raises:
            ValueError: If the backend is unsupported or if padding fails.

        """
        # Determine max shape along each axis
        max_shape = np.max([np.array(v.shape) for v in values], axis=0)

        # Get padding requirements for each input tensor
        padded = []
        for v in values:
            pad_width = []
            for dim, current_shape in enumerate(v.shape):
                if dim == concat_axis:
                    pad_width.append((0, 0))  # No padding on concat axis
                else:
                    diff = max_shape[dim] - current_shape
                    pad_width.append((0, diff))

            # Apply backend-specific pad function
            if fmt == DataFormat.TORCH:
                # Verify that torch is installed
                torch = ensure_torch()

                torch_pad = [
                    p for dims in reversed(pad_width) for p in dims
                ]  # reverse & flatten
                p = torch.nn.functional.pad(
                    input=v,
                    pad=torch_pad,
                    mode=map_pad_mode_to_backend(
                        mode=self.pad_mode,
                        backend=self._backend,
                    ),
                    value=self.pad_value,
                )
                padded.append(p)

            elif fmt == DataFormat.TENSORFLOW:
                # Verify that tf is installed
                tf = ensure_tensorflow()

                tf_pad = tf.constant(pad_width)
                p = tf.pad(
                    tensor=v,
                    paddings=tf_pad,
                    mode=map_pad_mode_to_backend(
                        mode=self.pad_mode,
                        backend=self._backend,
                    ),
                    constant_values=self.pad_value,
                )
                padded.append(p)

            else:
                # Default to numpy padding
                p = np.pad(
                    array=v,
                    pad_width=pad_width,
                    mode=map_pad_mode_to_backend(
                        mode=self.pad_mode,
                        backend=self._backend,
                    ),
                    constant_values=self.pad_value,
                )
                padded.append(p)

        return padded

    def _validate_dims(
        self,
        values: list[Any],
        concat_axis: int,
    ):
        """
        Verfies that all dimensions can be concatenated along the specified axis.

        Args:
            values (list[Any]):
                List of tensors to concatenate.
            concat_axis (int):
                The axis to concatenate along, relative to the actual tensors
                in `values`.

        """
        reference_shape = values[0].shape
        for i, v in enumerate(values[1:], start=1):
            for dim, (ref_dim, val_dim) in enumerate(
                zip(reference_shape, v.shape, strict=True),
            ):
                if dim == concat_axis:
                    continue
                if ref_dim != val_dim:
                    msg = (
                        f"Mismatch in non-concat dimension {dim} between input 0 and {i}: "
                        f"{ref_dim} vs {val_dim}. Set `pad_inputs=True` to auto-align. "
                        f"{reference_shape} vs {v.shape} on axis={concat_axis}"
                    )
                    raise ValueError(msg)

    def apply_merge(
        self,
        values: list[Any],
        *,
        includes_batch_dim: bool = True,
        fmt: DataFormat | None = None,
        domain: str = DOMAIN_FEATURES,
    ) -> Any:
        """
        Concatenate input tensors along the configured axis.

        Description:
            Optionally pads the inputs to align non-concat dimensions before applying
            backend-specific concatenation.

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
            Any: Concatenated tensor, in the specified format.

        """
        # Get data format (`fmt`)
        if fmt is None:
            if self._backend is None:
                fmt = DataFormat.NUMPY
            else:
                fmt = get_data_format_for_backend(self._backend)
        else:
            fmt = normalize_format(fmt=fmt)
        if not format_is_tensorlike(fmt):
            msg = (
                f"Format {fmt} does support tensors. "
                f"Must be one of: {_TENSORLIKE_FORMATS}."
            )
            raise ValueError(msg)

        # Ensure all elements in list are converted to fmt
        values = [convert_to_format(x, fmt=fmt) for x in values]

        # Get domain-specific axis
        if domain == DOMAIN_FEATURES:
            effective_axis = self.concat_axis
        elif domain == DOMAIN_TARGETS:
            effective_axis = self.target_axis
        elif domain == DOMAIN_TAGS:
            effective_axis = self.tags_axis
        elif domain == DOMAIN_SAMPLE_UUIDS:
            effective_axis = -1
        else:
            msg = f"Unknown domain: {domain}"
            raise ValueError(msg)

        # Get true concat axis (-1 = max dimension)
        if effective_axis == -1:
            effective_axis = max([len(x.shape) for x in values]) - 1
        else:
            # Adjust for batch dimension
            effective_axis += 1 if includes_batch_dim else 0

        # Apply padding if defined
        if self.pad_inputs:
            values = self._pad_inputs(values, fmt=fmt, concat_axis=effective_axis)

        # Ensure tensors can be concatenated along `effective_axis`
        self._validate_dims(values=values, concat_axis=effective_axis)

        # Apply backend-specific concatenation
        if fmt == DataFormat.TORCH:
            # Verify that torch is installed
            torch = ensure_torch()

            return torch.cat(tensors=values, dim=effective_axis)

        if fmt == DataFormat.TENSORFLOW:
            # Verify that tf is installed
            tf = ensure_tensorflow()

            return tf.concat(values=values, axis=effective_axis)

        # Default: numpy concatenation
        return np.concatenate(values, axis=effective_axis)
