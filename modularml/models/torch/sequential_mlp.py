import numpy as np
import torch

from modularml.core.models.torch_base_model import TorchBaseModel
from modularml.utils.data.shape_utils import ensure_tuple_shape
from modularml.utils.logging.warnings import warn
from modularml.utils.nn.activations import resolve_activation


class SequentialMLP(TorchBaseModel):
    """
    Configurable multi-layer perceptron (MLP) with lazy shape inference.

    This model wraps a PyTorch sequential MLP, with optional layer count, activation,
    and dropout. It supports lazy building when input/output shapes are unknown.
    """

    def __init__(
        self,
        *,
        input_shape: tuple[int, ...] | None = None,
        output_shape: tuple[int, ...] | None = None,
        n_layers: int = 2,
        hidden_dim: int = 32,
        activation: str = "relu",
        dropout: float = 0.0,
        **kwargs,
    ):
        """
        Configurable multi-layer perceptron (MLP) with lazy shape inference.

        This model wraps a PyTorch sequential MLP, with optional layer count, activation,
        and dropout. It supports lazy building when input/output shapes are unknown.

        Args:
            input_shape (tuple[int, ...], optional):
                Shape of the input excluding batch dim.
            output_shape (tuple[int, ...], optional):
                Shape of the output excluding batch dim.
            n_layers (int):
                Number of fully connected layers.
            hidden_dim (int):
                Number of hidden units per layer.
            activation (str):
                Activation function name (e.g., 'relu', 'gelu').
            dropout (float):
                Dropout rate (0.0 = no dropout).
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
            activation=activation,
            dropout=dropout,
            **kwargs,
        )

        # Ensure shape formatting (if given)
        self._input_shape: tuple[int, ...] | None = ensure_tuple_shape(
            shape=input_shape,
            min_len=2,
            max_len=None,
            allow_null_shape=True,
        )
        self._output_shape: tuple[int, ...] | None = ensure_tuple_shape(
            shape=output_shape,
            min_len=2,
            max_len=None,
            allow_null_shape=True,
        )

        # Init args
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.dropout = dropout

        # Layers (construct in build())
        self.fc = None

        # Build model immediately if given all shapes
        if self._input_shape and self._output_shape:
            self.build(self._input_shape, self._output_shape)

    @property
    def input_shape(self) -> tuple[int, ...] | None:
        """The expected input shape excluding batch dimension."""
        return self._input_shape

    @property
    def output_shape(self) -> tuple[int, ...] | None:
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
        Builds the internal torch.nn.Sequential model.

        Args:
            input_shape (tuple[int], optional):
                Input shape excluding batch dim.
            output_shape (tuple[int], optional):
                Output shape excluding batch dim.
            force (bool):
                If model is already instantiated, `force` determines whether
                to reinstantiate with the new shapes. Defaults to False.

        Raises:
            ValueError: If shape mismatch is detected.
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

        flat_input = int(np.prod(self._input_shape))
        flat_output = int(np.prod(self._output_shape))
        act_fn = resolve_activation(self.activation, backend=self.backend)

        layers = []
        for i in range(self.n_layers - 1):
            in_dim = flat_input if i == 0 else self.hidden_dim
            layers.append(torch.nn.Linear(in_dim, self.hidden_dim))
            layers.append(act_fn)
            if self.dropout > 0:
                layers.append(torch.nn.Dropout(self.dropout))

        final_in = self.hidden_dim if self.n_layers > 1 else flat_input
        layers.append(torch.nn.Linear(final_in, flat_output))

        self.fc = torch.nn.Sequential(*layers)
        self._built = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor):
                Input tensor of shape (batch_size, *input_shape)

        Returns:
            torch.Tensor:
                Output tensor of shape (batch_size, *output_shape)

        """
        # ensure input is 3D
        # if x.ndim == 2:
        #     x = x.unsqueeze(1)  # (batch, 1, length)
        if x.ndim > 2:
            x = x.view(x.size(0), -1)

        if not self.is_built:
            self.build(input_shape=tuple(x.shape[1:]))

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x.view(x.size(0), *self._output_shape)
