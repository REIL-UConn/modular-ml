"""Torch SequentialCNN reference implementation with lazy building."""

import numpy as np
import torch

from modularml.core.models.torch_base_model import TorchBaseModel
from modularml.utils.data.shape_utils import ensure_tuple_shape
from modularml.utils.logging.warnings import warn
from modularml.utils.nn.activations import resolve_activation


class SequentialCNN(TorchBaseModel):
    """
    Configurable 1D CNN supporting lazy input/output shape inference.

    Attributes:
        n_layers (int): Number of convolutional blocks.
        hidden_dim (int): Output channel width for intermediate blocks.
        kernel_size (int): Convolution kernel size.
        padding (int): Padding applied to convolutions.
        stride (int): Stride per convolution.
        activation (str): Name of activation resolved via
            :func:`resolve_activation`.
        dropout (float): Dropout probability injected after activations.
        pooling (int): Max-pooling kernel size when >1.
        flatten_output (bool): Whether to append a linear projection.
        conv_layers (torch.nn.Sequential | None): Stacked conv blocks.
        fc (torch.nn.Linear | None): Optional flattening projection.

    """

    def __init__(
        self,
        *,
        input_shape: tuple[int, ...] | None = None,
        output_shape: tuple[int, ...] | None = None,
        n_layers: int = 2,
        hidden_dim: int = 16,
        kernel_size: int = 3,
        padding: int = 1,
        stride: int = 1,
        activation: str = "relu",
        dropout: float = 0.0,
        pooling: int = 1,
        flatten_output: bool = True,
        **kwargs,
    ):
        """
        Initialize the sequential CNN with optional lazy building.

        Args:
            input_shape (tuple[int, ...] | None): Shape excluding the
                batch dimension.
            output_shape (tuple[int, ...] | None): Desired output shape
                excluding the batch dimension.
            n_layers (int): Number of convolutional blocks.
            hidden_dim (int): Number of output channels in hidden blocks.
            kernel_size (int): Kernel size for convolutions.
            padding (int): Padding applied to convolutions.
            stride (int): Stride per convolution.
            activation (str): Activation name resolved via
                :func:`resolve_activation`.
            dropout (float): Dropout probability applied after blocks.
            pooling (int): Kernel size for optional :class:`MaxPool1d`.
            flatten_output (bool): Append a linear projection to match the
                target shape when True.
            **kwargs (Any): Extra keyword arguments forwarded to
                :class:`TorchBaseModel`.

        """
        # Pass all init args directly to parent class
        # This allows for automatic implementation of get_config/from_config
        super().__init__(
            input_shape=input_shape,
            output_shape=output_shape,
            n_layers=n_layers,
            hidden_dim=hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            activation=activation,
            dropout=dropout,
            pooling=pooling,
            flatten_output=flatten_output,
            **kwargs,
        )

        # Ensure shape formatting (if given)
        self._input_shape: tuple[int, ...] | None = ensure_tuple_shape(
            shape=input_shape,
            min_len=2,
            max_len=3,
            allow_null_shape=True,
        )
        self._output_shape: tuple[int, ...] | None = ensure_tuple_shape(
            shape=output_shape,
            min_len=2,
            max_len=3,
            allow_null_shape=True,
        )

        # Init args
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.activation = activation
        self.dropout = dropout
        self.pooling = pooling
        self.flatten_output = flatten_output

        # Layers (construct in build())
        self.conv_layers = None
        self.fc = None

        # Build model immediately if given all shapes
        if self._input_shape and self._output_shape:
            self.build(self._input_shape, self._output_shape)

    @property
    def input_shape(self) -> tuple[int] | None:
        """
        Return the expected input shape excluding the batch dimension.

        Returns:
            tuple[int, ...] | None: Input shape when known.

        """
        return self._input_shape

    @property
    def output_shape(self) -> tuple[int] | None:
        """
        Return the output shape excluding the batch dimension.

        Returns:
            tuple[int, ...] | None: Output shape when known.

        """
        return self._output_shape

    def build(
        self,
        input_shape: tuple[int, ...] | None = None,
        output_shape: tuple[int, ...] | None = None,
        *,
        force: bool = False,
    ):
        """
        Build convolutional blocks and optional linear projection.

        Args:
            input_shape (tuple[int, ...] | None):
                Shape excluding batch dimension.
            output_shape (tuple[int, ...] | None):
                Target shape excluding batch dimension.
            force (bool):
                Whether to rebuild existing layers.

        Raises:
            ValueError: If shapes conflict with previously supplied ones
                without forcing, or when ``input_shape`` is missing.
            UserWarning: If ``output_shape`` is missing and defaults are
                inferred.

        """
        # Set input shape (check for mismatch)
        if input_shape:
            input_shape = ensure_tuple_shape(
                shape=input_shape,
                min_len=2,
                max_len=None,
                allow_null_shape=False,
            )
            if (
                (self._input_shape is not None)
                and (input_shape != self._input_shape)
                and (not force)
            ):
                msg = (
                    f"Build called with `input_shape={input_shape}` but input shape is already defined "
                    f"with value `{self._input_shape}`. To override the existing shape, set `force=True`."
                )
                raise ValueError(msg)
            self._input_shape = input_shape

        # Set input shape (check for mismatch)
        if output_shape:
            output_shape = ensure_tuple_shape(
                shape=output_shape,
                min_len=2,
                max_len=None,
                allow_null_shape=False,
            )
            if (
                (self._output_shape is not None)
                and (output_shape != self._output_shape)
                and (not force)
            ):
                msg = (
                    f"Build called with `output_shape={output_shape}` but input shape is already defined "
                    f"with value `{self._output_shape}`. To override the existing shape, set `force=True`."
                )
                raise ValueError(msg)
            self._output_shape = output_shape

        if self.is_built and not force:
            return

        if self._input_shape is None:
            raise ValueError("Input shape must be provided before building the model.")
        if self._output_shape is None:
            msg = "No output shape provided. Using default shape of (1, hidden_dim)."
            warn(msg, category=UserWarning, stacklevel=2)
            self._output_shape = (1, self.hidden_dim)

        act_fn = resolve_activation(self.activation, backend=self.backend)

        num_features, feature_len = self._input_shape
        layers = []
        for _ in range(self.n_layers):
            layers.append(
                torch.nn.Conv1d(
                    in_channels=num_features,
                    out_channels=self.hidden_dim,
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    padding=self.padding,
                ),
            )
            layers.append(act_fn)

            if self.dropout > 0:
                layers.append(torch.nn.Dropout(self.dropout))

            if self.pooling > 1 and feature_len >= self.pooling:
                layers.append(torch.nn.MaxPool1d(kernel_size=self.pooling))
                feature_len = feature_len // self.pooling

            num_features = self.hidden_dim

        self.conv_layers = torch.nn.Sequential(*layers)

        if self.flatten_output:
            dummy = torch.zeros(1, *self._input_shape)
            with torch.no_grad():
                conv_out = self.conv_layers(dummy)
            conv_out_dim = conv_out.shape[1] * conv_out.shape[2]

            if self._output_shape is None:
                self._output_shape = (conv_out_dim,)

            flat_output_dim = int(np.prod(self._output_shape))
            self.fc = torch.nn.Linear(conv_out_dim, flat_output_dim)

        self._built = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run the forward pass through the sequential CNN.

        Args:
            x (torch.Tensor):
                Input tensor of shape `(batch, num_channels, length)`.

        Returns:
            torch.Tensor: Output tensor shaped `(batch, *output_shape)`.

        """
        # ensure input is 3D
        if x.ndim == 2:
            x = x.unsqueeze(1)  # (batch, 1, length)

        if not self.is_built:
            self.build(input_shape=tuple(x.shape[1:]))

        x = self.conv_layers(x)
        if self.flatten_output:
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            x = x.view(x.size(0), *self._output_shape)
        return x
