"""Wrappers for integrating PyTorch modules with ModularML."""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any

import torch

from modularml.core.models.base_model import BaseModel
from modularml.utils.data.shape_utils import ensure_tuple_shape
from modularml.utils.io.inspection import infer_kwargs_from_init
from modularml.utils.logging.warnings import warn
from modularml.utils.nn.backend import Backend

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np


class TorchModelWrapper(BaseModel, torch.nn.Module):
    """
    Wrap an arbitrary :class:`torch.nn.Module` into :class:`BaseModel`.

    Attributes:
        model (torch.nn.Module | None):
            Wrapped module when eagerly provided.
        model_class (Callable | None):
            Constructor used for lazy builds.
        model_kwargs (dict[str, Any] | None):
            Stored kwargs for lazy instantiation.
        inject_input_shape_as (str):
            Keyword name used when injecting the inferred input shape.
        inject_output_shape_as (str):
            Keyword name used when injecting the inferred output shape.

    """

    def __init__(
        self,
        *,
        model: torch.nn.Module = None,
        model_class: Callable | None = None,
        model_kwargs: dict[str, Any] | None = None,
        inject_input_shape_as: str = "input_shape",
        inject_output_shape_as: str = "output_shape",
        **kwargs,
    ):
        """
        Initialize the wrapper for an eager or lazy PyTorch model.

        Args:
            model (torch.nn.Module | None):
                Already-instantiated module to wrap.
            model_class (Callable | None):
                Module class to instantiate lazily when `model` is not provided.
            model_kwargs (dict[str, Any] | None):
                Keyword arguments cached for lazy instantiation.
            inject_input_shape_as (str):
                Keyword used to inject the inferred input shape into `model_kwargs`
                when missing.
            inject_output_shape_as (str):
                Keyword used to inject the inferred output shape into `model_kwargs`
                when missing.
            **kwargs:
                Additional keyword arguments passed to :class:`BaseModel`.

        Raises:
            ValueError: If neither `model` nor `model_class` is supplied.
            TypeError: If provided arguments are not callable or modules.

        """
        torch.nn.Module.__init__(self)

        # Validation of provided arguments
        if model is None and model_class is None:
            raise ValueError("Must provide either `model` or `model_class`.")
        if model is not None and not isinstance(model, torch.nn.Module):
            raise TypeError("`model` must be a torch.nn.Module instance.")
        if model_class is not None and not callable(model_class):
            raise TypeError("`model_class` must be callable.")

        # Try to cast model to cls + kwargs
        if model is not None:
            model_class = type(model)
            # Infer only if user didn't provide kwargs
            if model_kwargs is None:
                try:
                    model_kwargs = self._infer_init_args(model)
                except RuntimeError:
                    msg = (
                        "Failed to infer `model_kwargs` from the wrapped model "
                        f"instance. This model (`{self.model.__class__.__qualname__}`) "
                        "cannot be serialized because its constructor arguments could "
                        "not be fully reconstructed from the current object state."
                    )
                    hint = (
                        "Explicitly provide `model_kwargs` when constructing this "
                        "wrapper, or modify the model so all required `__init__` "
                        "parameters are stored as instance attributes with the same "
                        "names."
                    )
                    warn(msg, category=RuntimeWarning, stacklevel=2, hints=hint)

        # Pass all init args directly to BaseModel
        # This allows for automatic implementation of get_config/from_config
        super().__init__(
            backend=kwargs.pop("backend", None) or Backend.TORCH,
            model_class=model_class,
            model_kwargs=model_kwargs,
            inject_input_shape_as=inject_input_shape_as,
            inject_output_shape_as=inject_output_shape_as,
        )

        # Init args
        self.model: torch.nn.Module | None = model
        self.model_class = model_class
        self.model_kwargs = model_kwargs
        self.inject_input_shape_as = inject_input_shape_as
        self.inject_output_shape_as = inject_output_shape_as

        # Shape tracking
        self._input_shape: tuple[int, ...] | None = None
        self._output_shape: tuple[int, ...] | None = None

    @property
    def input_shape(self) -> tuple[int, ...] | None:
        """Returns the input shape (if known)."""
        return self._input_shape

    @property
    def output_shape(self) -> tuple[int, ...] | None:
        """Returns the output shape (if known)."""
        return self._output_shape

    # ================================================
    # Build
    # ================================================
    def build(
        self,
        input_shape: tuple[int, ...] | None = None,
        output_shape: tuple[int, ...] | None = None,
        *,
        force: bool = False,
    ):
        """
        Construct or validate the wrapped PyTorch model.

        If a model instance was provided at initialization, this method validates its input
        and output shapes using a dummy forward pass. If a model class and keyword arguments
        were provided instead, the model is instantiated by injecting the inferred input and/or
        output shape if not already present in the provided kwargs.

        This method ensures the model is compatible with the ModularML shape propagation logic
        and supports both eager and lazy model creation strategies.

        Args:
            input_shape (tuple[int, ...] | None):
                Input shape (excluding batch dimension) to use for validation or lazy
                instantiation.
            output_shape (tuple[int, ...] | None):
                Expected output shape (excluding batch dimension) to use for validation or
                lazy instantiation.
            force (bool):
                If model is already instantiated, force determines whether to reinstantiate
                with the new shapes. Defaults to False.

        Raises:
            RuntimeError: If validation fails for a pre-instantiated model, or if model
                construction fails due to invalid arguments or shape mismatches.

        """
        # Skip if don't need to build
        if self.model is not None and self.is_built and not force:
            return

        # Set input shape (check for mismatch)
        if input_shape:
            input_shape = ensure_tuple_shape(
                shape=input_shape,
                min_len=1,
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
                min_len=1,
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

        # Eager-wrapped model instance --> just validate shapes
        if self.model is not None:
            self._validate_model_shapes(
                input_shape=self._input_shape,
                output_shape=self._output_shape,
            )
            self._built = True
            return

        # Lazy construction
        if self.model_class is None:
            raise RuntimeError("Lazy build requested but no model_class provided.")

        kwargs = self._prepare_model_kwargs(
            input_shape=self._input_shape,
            output_shape=self._output_shape,
        )
        try:
            self.model = self.model_class(**kwargs)
        except Exception as e:
            msg = f"Failed to instantiate {self.model_class} with kwargs={kwargs}. {e}"
            raise RuntimeError(msg) from e
        self._validate_model_shapes(
            input_shape=self._input_shape,
            output_shape=self._output_shape,
        )
        self._built = True

    def _validate_model_shapes(
        self,
        input_shape: tuple[int, ...] | None = None,
        output_shape: tuple[int, ...] | None = None,
    ):
        """
        Validate that the wrapped module is compatible with expected shapes.

        Description:
            Runs a dummy forward pass using a random tensor of shape
            `(4, *input_shape)`. If execution fails or the resulting
            shape does not match `output_shape` (when provided), a
            :class:`RuntimeError` is raised.

        Args:
            input_shape (tuple[int, ...] | None): Expected per-sample
                input shape without the batch dimension.
            output_shape (tuple[int, ...] | None): Optional per-sample
                output shape for verification.

        Raises:
            RuntimeError: If the module rejects the input or produces an
                unexpected output shape.

        """
        dummy_in = torch.randn(4, *input_shape)
        # Check that pre-instantiated model accepts expected input_shape
        try:
            with torch.no_grad():
                dummy_out = self.model(dummy_in)
        except Exception as e:
            msg = (
                "Pre-instantiated PyTorch model failed to accept the expected "
                f"input_shape: {input_shape}"
            )
            raise RuntimeError(msg) from e

        # Check that pre-instantiated model returns expected output_shape
        if output_shape is not None and dummy_out.shape[1:] != output_shape:
            msg = (
                "Output shape of pre-instantiated PyTorch model does not match "
                f"expected output_shape: {dummy_out.shape[1:]} != {output_shape}"
            )
            raise RuntimeError(msg)
        self._output_shape = tuple(dummy_out.shape[1:])

    def _prepare_model_kwargs(
        self,
        input_shape: tuple[int, ...] | None,
        output_shape: tuple[int, ...] | None,
    ) -> dict[str, Any]:
        """
        Inject inferred shapes into keyword arguments for lazy builds.

        Args:
            input_shape (tuple[int, ...] | None):
                Input shape to inject via :attr:`inject_input_shape_as`.
            output_shape (tuple[int, ...] | None):
                Output shape to inject via :attr:`inject_output_shape_as`.

        Returns:
            dict[str, Any]:
                Keyword arguments used to instantiate :attr:`model_class`.

        """
        kwargs = dict(self.model_kwargs)

        # Inspect model_class constructor
        sig = inspect.signature(self.model_class.__init__)
        accepted_params = sig.parameters.keys()

        def maybe_inject(name: str, value: Any, error: str = "raise"):
            if value is None or name in kwargs:
                return
            if name in accepted_params:
                kwargs[name] = value
            else:
                msg = f"Model constructor for `{self.model_class.__name__}` does not accept `{name}`. Skipping injection."
                if error == "raise":
                    raise ValueError(msg)
                hint = "Revise keys given in `model_kwargs`."
                warn(msg, category=RuntimeWarning, stacklevel=2, hints=hint)

        # Try to inject input shape
        # If user explicitly changed input key -> raise error if not accepted
        error_mode = (
            "raise" if (self.inject_input_shape_as != "input_shape") else "warn"
        )
        maybe_inject(self.inject_input_shape_as, input_shape, error=error_mode)

        # Try to inject output shape
        # If user explicitly changed output key -> raise error if not accepted
        error_mode = (
            "raise" if (self.inject_output_shape_as != "output_shape") else "warn"
        )
        maybe_inject(self.inject_output_shape_as, output_shape, error=error_mode)

        return kwargs

    def _infer_init_args(self, model: Any) -> dict[str, Any]:
        """Infer constructor arguments from an instantiated module."""
        if hasattr(self, "model_kwargs") and self.model_kwargs is not None:
            raise ValueError("`model_kwargs` are already defined")

        if model is None:
            raise ValueError("Cannot infer kwargs for an uninstantiated model.")

        # Try to extract init parameters from signature
        model_kwargs = infer_kwargs_from_init(
            obj=model,
            strict=True,
        )
        return model_kwargs

    # ================================================
    # Forward Pass
    # ================================================
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass through the wrapped module."""
        if not self.is_built:
            self.build(input_shape=tuple(x.shape[1:]))
        return self.model(x)

    # ================================================
    # Weight Handling
    # ================================================
    def get_weights(self) -> dict[str, np.ndarray]:
        """Return :class:`torch.nn.Module` weights as numpy arrays."""
        if not self.is_built:
            return {}
        return {k: v.detach().cpu().numpy() for k, v in self.model.state_dict().items()}

    def set_weights(self, weights: dict[str, np.ndarray]) -> None:
        """Restore numpy-based weights produced by :meth:`get_weights`."""
        if not weights:
            return
        if self.model is None:
            raise RuntimeError("Cannot set weights before model is built.")

        torch_state = {k: torch.as_tensor(v) for k, v in weights.items()}
        self.model.load_state_dict(torch_state, strict=True)
