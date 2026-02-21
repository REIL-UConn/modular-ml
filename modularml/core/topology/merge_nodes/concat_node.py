from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from modularml.core.data.schema_constants import (
    DOMAIN_FEATURES,
    DOMAIN_SAMPLE_UUIDS,
    DOMAIN_TAGS,
    DOMAIN_TARGETS,
)
from modularml.core.topology.merge_nodes.merge_node import MergeNode
from modularml.core.topology.merge_nodes.merge_strategy import MergeStrategy
from modularml.utils.data.conversion import convert_to_format
from modularml.utils.data.data_format import (
    _TENSORLIKE_FORMATS,
    DataFormat,
    format_is_tensorlike,
    get_data_format_for_backend,
    normalize_format,
)
from modularml.utils.environment.optional_imports import ensure_tensorflow, ensure_torch
from modularml.utils.logging.warnings import warn
from modularml.utils.nn.padding import PadMode, map_pad_mode_to_backend

if TYPE_CHECKING:
    from modularml.core.data.sample_data import SampleData
    from modularml.core.experiment.experiment_node import ExperimentNode
    from modularml.core.references.experiment_reference import ExperimentNodeReference


# Type alias for strategy parameters
StrategyType = int | str | MergeStrategy


class ConcatNode(MergeNode):
    """
    A merge stage that concatenates multiple inputs along a specified axis.

    Description:
        This stage merges tensors by concatenating them along a specified axis. It
        supports automatic padding of non-concat dimensions to align shapes, allowing
        for flexible merging even when inputs vary in size. Padding behavior can be
        controlled via mode (e.g., 'constant', 'reflect', 'replicate') and value.

        For the targets and tags domains, non-concatenation merge strategies can be
        used instead of axis-based concatenation. For example, `"first"` selects
        targets from the first input, `"mean"` computes an element-wise average,
        or an `ExperimentNodeReference` can be passed to select targets from a
        specific upstream input.

    Attributes:
        label (str):
            Unique identifier for this node.
        upstream_refs (list[ExperimentNode | ExperimentNodeReference]):
            Upstream node references from which inputs will be received.
        concat_axis (int):
            The axis along which to concatenate feature inputs.
        target_strategy (int | MergeStrategy | ExperimentNodeReference):
            Strategy for merging targets. An int means concatenation along that
            axis; a MergeStrategy applies an aggregation; an ExperimentNodeReference
            selects targets from a specific upstream input.
        tags_strategy (int | MergeStrategy | ExperimentNodeReference):
            Strategy for merging tags (same semantics as target_strategy).
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

        # Concatenate features, use first input's targets
        stage = ConcatNode(
            label="merge",
            upstream_refs=[fs_ref, mn],
            concat_axis=1,
            concat_axis_targets="first", # first = fs_ref
        )
    ```

    """

    def __init__(
        self,
        label: str,
        upstream_refs: list[ExperimentNode | ExperimentNodeReference],
        concat_axis: int = 0,
        *,
        concat_axis_targets: StrategyType | ExperimentNodeReference = -1,
        concat_axis_tags: StrategyType | ExperimentNodeReference = -1,
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
                The axis along which to concatenate feature inputs. Does not include
                the batch dimension. That is, for shape (with batch) of (32,1,16),
                axis=0 refers to "1". The examples below omit the batch dimension.
                * axis=0: concat along features. E.g, (1,16) + (1,16) -> (2,16))
                * axis>1: concat within features. E.g, (1,16,16) + (1,16,8) -> (1,16,24))
                    All data must have at least `concat_axis` dimensions.
                * axis=-1: concat along last axis of all data. E.g, (1, 16, 16) +
                    (1, 16, 8) -> (1, 16, 24).

            concat_axis_targets (int | str | MergeStrategy | ExperimentNodeReference):
                Strategy for merging the targets domain. Accepts:
                * int: Concatenate along this axis (same semantics as `concat_axis`).
                    Defaults to -1 (last axis).
                * str or MergeStrategy: Apply an aggregation strategy. Supported
                    values: "first", "last", "mean".
                * ExperimentNodeReference: Select targets from the upstream input
                    matching this reference.

            concat_axis_tags (int | str | MergeStrategy | ExperimentNodeReference):
                Strategy for merging the tags domain. Same semantics as
                `concat_axis_targets`. Defaults to -1 (last axis).

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
        self.target_strategy = self._normalize_strategy(concat_axis_targets)
        self.tags_strategy = self._normalize_strategy(concat_axis_tags)
        self.pad_inputs = bool(pad_inputs)
        self.pad_mode = pad_mode if isinstance(pad_mode, PadMode) else PadMode(pad_mode)
        self.pad_value = pad_value

        if self.pad_mode not in [PadMode.CONSTANT]:
            msg = f"Pad mode is not supported yet: {self.pad_mode}"
            raise NotImplementedError(msg)

    # ================================================
    # Strategy normalization
    # ================================================
    @staticmethod
    def _normalize_strategy(
        value: int | str | MergeStrategy | ExperimentNodeReference,
    ) -> int | MergeStrategy | ExperimentNodeReference:
        """
        Normalize a strategy parameter to its canonical form.

        Args:
            value: Raw strategy value from the constructor.

        Returns:
            int | MergeStrategy | ExperimentNodeReference:
                Normalized strategy.

        Raises:
            ValueError: If the value is an unrecognized string.

        """
        from modularml.core.experiment.experiment_node import ExperimentNode
        from modularml.core.references.experiment_reference import (
            ExperimentNodeReference,
        )

        if isinstance(value, ExperimentNode):
            return value.reference()
        if isinstance(value, ExperimentNodeReference):
            return value
        if isinstance(value, int):
            return value
        if isinstance(value, MergeStrategy):
            return value
        if isinstance(value, str):
            return MergeStrategy(value)

        msg = (
            "Expected type to be one of int, str, MergeStrategy, or "
            f"ExperimentNodeReference. Received: {type(value)}."
        )
        raise TypeError(msg)

    @property
    def target_axis(self) -> int:
        """
        Target concatenation axis (only valid when target_strategy is int).

        Raises:
            TypeError: If the target strategy is not int-based.

        """
        if isinstance(self.target_strategy, int):
            return self.target_strategy
        msg = (
            f"target_axis is not available when target_strategy is "
            f"{self.target_strategy!r}. Use target_strategy instead."
        )
        raise TypeError(msg)

    @property
    def tags_axis(self) -> int:
        """
        Tags concatenation axis (only valid when tags_strategy is int).

        Raises:
            TypeError: If the tags strategy is not int-based.

        """
        if isinstance(self.tags_strategy, int):
            return self.tags_strategy
        msg = (
            f"tags_axis is not available when tags_strategy is "
            f"{self.tags_strategy!r}. Use tags_strategy instead."
        )
        raise TypeError(msg)

    # ================================================
    # Padding & Validation
    # ================================================
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

    # ================================================
    # Concatenation (apply_merge)
    # ================================================
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
            backend-specific concatenation. This method handles axis-based
            concatenation only. Non-concat strategies (e.g., `"first"`, `"mean"`)
            are handled by :meth:`_merge_sample_data` before reaching this method.

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
            effective_axis = self.target_strategy
        elif domain == DOMAIN_TAGS:
            effective_axis = self.tags_strategy
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

    # ================================================
    # Non-concat merge strategies
    # ================================================
    def _apply_strategy(
        self,
        values: list[Any],
        strategy: MergeStrategy,
        fmt: DataFormat,
    ) -> Any:
        """
        Apply a non-concat merge strategy to a list of tensors.

        Args:
            values (list[Any]):
                Non-None tensors to aggregate.
            strategy (MergeStrategy):
                The aggregation strategy to apply.
            fmt (DataFormat):
                Data format for the output tensor.

        Returns:
            Any: Aggregated tensor.

        """
        values = [convert_to_format(x, fmt=fmt) for x in values]

        if strategy == MergeStrategy.FIRST:
            return values[0]
        if strategy == MergeStrategy.LAST:
            return values[-1]
        if strategy == MergeStrategy.MEAN:
            return self._compute_mean(values, fmt=fmt)

        msg = f"Unsupported merge strategy: {strategy}"
        raise ValueError(msg)

    def _compute_mean(
        self,
        values: list[Any],
        fmt: DataFormat,
    ) -> Any:
        """
        Compute element-wise mean across inputs.

        Args:
            values (list[Any]):
                Tensors to average. All must have the same shape.
            fmt (DataFormat):
                Data format of the tensors.

        Returns:
            Any: Mean tensor.

        """
        if fmt == DataFormat.TORCH:
            torch = ensure_torch()
            return torch.stack(values).mean(dim=0)

        if fmt == DataFormat.TENSORFLOW:
            tf = ensure_tensorflow()
            return tf.reduce_mean(tf.stack(values), axis=0)

        # Default: numpy
        return np.mean(np.stack(values), axis=0)

    def _select_by_reference(
        self,
        data: list[SampleData],
        attr: str,
        ref: ExperimentNodeReference,
    ) -> Any | None:
        """
        Select a domain's data from the input matching the given reference.

        Args:
            data (list[SampleData]):
                Sorted list of SampleData inputs (same order as
                the references added to this node).
            attr (str):
                The domain attribute name (e.g., "targets").
            ref (ExperimentNodeReference):
                The upstream reference to select from.

        Returns:
            Any | None: The selected tensor, or None if that input has no
                data for the given domain.

        Raises:
            RuntimeError: If called outside of a forward pass.
            ValueError: If the reference is not among upstream refs.

        """
        # Find the index matching the reference
        for idx, sorted_ref in enumerate(self.get_upstream_refs()):
            if ref == sorted_ref:
                return getattr(data[idx], attr)

        available = [r.node_id for r in self.get_upstream_refs()]
        msg = f"Reference '{ref.node_id}' is not among upstream refs: {available}"
        raise ValueError(msg)

    # ================================================
    # Domain-aware sample data merging
    # ================================================
    def _merge_sample_data(
        self,
        data: list[SampleData],
        fmt: DataFormat,
    ) -> SampleData:
        """
        Merge SampleData with flexible per-domain strategies.

        Description:
            Features are always concatenated along `self.concat_axis`.
            Targets and tags support non-concat strategies (e.g., "first",
            "mean", or select-by-reference). Sample UUIDs are always
            concatenated along the last axis.

        Args:
            data (list[SampleData]):
                Input SampleData objects to merge.
            fmt (DataFormat):
                Data format for features and targets.

        Returns:
            SampleData: Merged output.

        """
        from modularml.core.data.sample_data import SampleData
        from modularml.core.references.experiment_reference import (
            ExperimentNodeReference,
        )

        merged_attrs: dict[str, Any] = {}

        domain_config: list[
            tuple[str, DataFormat, int | MergeStrategy | ExperimentNodeReference]
        ] = [
            (DOMAIN_FEATURES, fmt, self.concat_axis),
            (DOMAIN_TARGETS, fmt, self.target_strategy),
            (DOMAIN_TAGS, DataFormat.NUMPY, self.tags_strategy),
            (DOMAIN_SAMPLE_UUIDS, DataFormat.NUMPY, -1),
        ]

        for attr, attr_fmt, strategy in domain_config:
            values = [getattr(d, attr) for d in data]
            has_data = [v is not None for v in values]

            # If none have data, skip
            if not any(has_data):
                continue

            if isinstance(strategy, int):
                # Concatenation mode (original behavior)
                if not all(has_data):
                    msg = (
                        f"Not all inputs have data for the `{attr}` attribute. "
                        f"The merged results will not contain `{attr}` data."
                    )
                    warn(msg, stacklevel=2)
                    continue

                merged_attrs[attr] = self.apply_merge(
                    values=values,
                    includes_batch_dim=True,
                    fmt=attr_fmt,
                    domain=attr,
                )

            elif isinstance(strategy, MergeStrategy):
                # Non-concat aggregation — silently filter out Nones
                non_none = [v for v in values if v is not None]
                if not non_none:
                    continue
                merged_attrs[attr] = self._apply_strategy(
                    values=non_none,
                    strategy=strategy,
                    fmt=attr_fmt,
                )

            elif isinstance(strategy, ExperimentNodeReference):
                # Select from a specific upstream input
                selected = self._select_by_reference(
                    data=data,
                    attr=attr,
                    ref=strategy,
                )
                if selected is not None:
                    merged_attrs[attr] = selected

        return SampleData(data=merged_attrs, kind="output")

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
        from modularml.core.references.experiment_reference import (
            ExperimentNodeReference,
        )

        def _get_strategy_cfg(arg) -> dict[str, Any]:
            strat_cfg = {
                "type": None,
                "value": None,
            }
            if isinstance(arg, MergeStrategy):
                strat_cfg["type"] = "MergeStrategy"
                strat_cfg["value"] = arg.value
            elif isinstance(arg, ExperimentNodeReference):
                strat_cfg["type"] = "ExperimentNodeReference"
                strat_cfg["value"] = arg.get_config()
            else:
                strat_cfg["type"] = "none"
                strat_cfg["value"] = arg
            return strat_cfg

        cfg = super().get_config()
        cfg.update(
            {
                "merge_node_type": self.__class__.__qualname__,
                "concat_axis": self.concat_axis,
                "target_strategy": _get_strategy_cfg(self.target_strategy),
                "tags_strategy": _get_strategy_cfg(self.tags_strategy),
                "pad_inputs": self.pad_inputs,
                "pad_mode": self.pad_mode.value,
                "pad_value": self.pad_value,
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
        from modularml.core.references.experiment_reference import (
            ExperimentNodeReference,
        )

        def _decode_strategy_cfg(
            strat_cfg,
        ) -> int | str | ExperimentNodeReference | MergeStrategy:
            strat_type = strat_cfg.get("type")
            strat_value = strat_cfg.get("value")
            if strat_type == "none":
                return strat_value
            if strat_type == "MergeStrategy":
                return MergeStrategy(value=strat_value)
            if strat_type == "ExperimentNodeReference":
                return ExperimentNodeReference.from_config(strat_value)

            msg = "Invalid merge strategy config."
            raise ValueError(msg)

        cb_cls_name = config.get("merge_node_type")
        if cb_cls_name != cls.__qualname__:
            msg = f"Invalid config for {cls.__qualname__}."
            raise ValueError(msg)

        return cls(
            label=config["label"],
            upstream_refs=config["upstream_refs"],
            concat_axis=config["concat_axis"],
            concat_axis_targets=_decode_strategy_cfg(config["target_strategy"]),
            concat_axis_tags=_decode_strategy_cfg(config["tags_strategy"]),
            pad_inputs=config["pad_inputs"],
            pad_mode=config["pad_mode"],
            pad_value=config["pad_value"],
            node_id=config.get("node_id"),
            register=config.get("register", True),
        )
