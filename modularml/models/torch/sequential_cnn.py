import numpy as np
import torch

from modularml.core.models.torch_base_model import TorchBaseModel
from modularml.utils.data.shape_utils import ensure_tuple_shape
from modularml.utils.logging.warnings import warn
from modularml.utils.nn.activations import resolve_activation


class SequentialCNN(TorchBaseModel):
    """
    Configurable 1D convolutional neural network (CNN) with lazy shape inference.

    This model supports stacked Conv1D layers followed optionally by dropout, pooling,
    and a final flattening linear projection to a user-defined output shape.
    It supports lazy building when input/output shapes are unknown.
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
        Initialize a sequential 1D CNN model with lazy layer building.

        Args:
            input_shape (tuple[int, ...], optional):
                Shape of the input excluding batch dim.
            output_shape (tuple[int, ...], optional):
                Shape of the output excluding batch dim.
            n_layers (int):
                Number of fully connected layers.
            hidden_dim (int):
                Number of hidden units per layer.
            kernel_size (int):
                Kernel size for convolution.
            padding (int):
                Padding for convolution.
            stride (int):
                Stride for convolution.
            activation (str):
                Activation function name (e.g., 'relu', 'tanh').
            dropout (float):
                Dropout probability (0.0 = no dropout).
            pooling (int):
                Pooling kernel size (1 = no pooling).
            flatten_output (bool):
                If True, appends a flattening linear projection to match output shape.
            kwargs:
                Additional key-word arguments to pass through to parent.

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
        """The expected input shape excluding batch dimension."""
        return self._input_shape

    @property
    def output_shape(self) -> tuple[int] | None:
        """The output shape of the model excluding batch dimension."""
        return self._output_shape

    def build(
        self,
        input_shape: tuple[int, ...] | None = None,
        output_shape: tuple[int, ...] | None = None,
        *,
        force: bool = False,
    ):
        """
        Build the CNN layers and optional final linear projection.

        Args:
            input_shape (tuple[int], optional):
                Input shape excluding batch dim.
            output_shape (tuple[int], optional):
                Output shape excluding batch dim.
            force (bool):
                If model is already instantiated, `force` determines whether
                to reinstantiate with the new shapes. Defaults to False.

        Raises:
            ValueError: If inconsistent shape is passed compared to earlier ones, or if input shape is not specified.
            UserWarning: If output shape is not provided, a default fallback is used.

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
        Run forward pass of the model.

        Args:
            x (torch.Tensor):
                Input tensor of shape (batch_size, *input_shape) where
                input_shape = (num_channel, channel_length).

        Returns:
            torch.Tensor:
                Output tensor of shape (batch_size, *output_shape)

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
