from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

from modularml.core.io.protocols import Configurable, Stateful
from modularml.utils.data.comparators import deep_equal
from modularml.utils.environment.optional_imports import ensure_tensorflow, ensure_torch
from modularml.utils.errors.exceptions import (
    BackendNotSupportedError,
    OptimizerError,
    OptimizerNotSetError,
)
from modularml.utils.nn.backend import Backend, infer_backend, normalize_backend

if TYPE_CHECKING:
    from collections.abc import Callable


def _safe_infer_backend(obj_or_cls: Any) -> Backend:
    backend = infer_backend(obj_or_cls=obj_or_cls)

    if backend == Backend.NONE:
        msg = (
            "Could not infer backend from optimizer class. Specify backend explicitly."
        )
        raise ValueError(msg)

    return backend


class Optimizer(Configurable, Stateful):
    """
    Backend-agnostic optimizer wrapper with lazy construction.

    Supported initialization modes:
        1. Class + kwargs (string or class)
        2. Callable factory: Callable[[params], optimizer]
    """

    def __init__(
        self,
        opt: str | type | None = None,
        *,
        opt_kwargs: dict[str, Any] | None = None,
        factory: Callable | None = None,
        backend: Backend | None = None,
    ):
        if opt is not None and factory is not None:
            msg = (
                "Provide either an optimizer (`opt`) or a `factory` callable, not both."
            )
            raise ValueError(msg)

        # Case 1: class / name + kwargs
        if opt is not None:
            if isinstance(opt, str):
                self.name = opt.lower()
                if backend is None:
                    msg = (
                        "Backend must be specified when initializing an optimizer "
                        "with a string-name."
                    )
                    raise ValueError(msg)
                self._backend = normalize_backend(backend)
                self.cls = self._resolve()

            elif isinstance(opt, type):
                self.name = opt.__name__
                self.cls = opt
                self._backend = _safe_infer_backend(self.cls)

            else:
                msg = (
                    "Optimizer (`opt`) must be a string-name or class. "
                    f"Recevied: {type(opt)}"
                )
                raise TypeError(msg)

            self.kwargs = opt_kwargs or {}
            self._factory = None

        # Case 2: factory
        elif factory is not None:
            self._factory = factory
            # don't know name or cls until instantiated
            self.cls = None
            self.name = None
            self.kwargs = opt_kwargs or {}
            if backend is None:
                msg = (
                    "Backend must be specified when initializing an optimizer "
                    "with a factory."
                )
                raise ValueError(msg)
            self._backend = normalize_backend(backend)

        else:
            raise ValueError(
                "Must provide either an optimizer (`opt`) or a `factory` callable.",
            )

        # Runtime state
        self.instance: Any | None = None
        self.parameters: Any | None = None

        # Pending serialized internal state (used after from_state())
        # Structure:
        #   PyTorch: {"state_dict": ...}
        #   TF:      {"weights": [...]}
        self._pending_state: dict[str, Any] | None = None

    @classmethod
    def from_factory(cls, factory: Callable, *, backend: Backend) -> Optimizer:
        return cls(factory=factory, backend=backend)

    def __eq__(self, other):
        if not isinstance(other, Optimizer):
            msg = f"Cannot compare equality between Optimizer and {type(other)}"
            raise TypeError(msg)

        # Compare config
        if not deep_equal(self.get_config(), other.get_config()):
            return False

        # Compare state
        return deep_equal(self.get_state(), other.get_state())

    __hash__ = None

    # ================================================
    # Core Properties
    # ================================================
    @property
    def is_built(self) -> bool:
        return self.instance is not None

    @property
    def backend(self) -> Backend | None:
        return self._backend

    @backend.setter
    def backend(self, value: Backend):
        self._backend = value

    # ================================================
    # Representation
    # ================================================
    def _summary_rows(self) -> list[tuple]:
        return [
            ("name", self.name),
            ("cls", str(self.cls.__name__ if self.cls else None)),
            ("kwargs", [(k, str(v)) for k, v in self.kwargs.items()]),
            ("backend", f"{self.backend!r}"),
        ]

    def __repr__(self):
        msg_kwargs = ""
        for k, v in self.kwargs.items():
            msg_kwargs += f", {k}={v}"
        name = self.name if self.name is not None else "<custom>"
        return f"Optimizer('{name}'{msg_kwargs})"

    # ================================================
    # Internal helpers
    # ================================================
    def _resolve(self) -> Callable:
        """
        Resolve a named optimizer to its backend-specific class by introspection.

        Strategy:
            1. Determine backend-specific optimizer module path
            2. Inspect all classes defined in that module
            3. Match class name case-insensitively against `self.name`
        """
        if not isinstance(self.name, str):
            raise OptimizerError(
                "Optimizer name must be a string to resolve dynamically.",
            )

        name_lc = self.name.lower()

        # Resolve backend optimizer module
        if self.backend == Backend.TORCH:
            torch = ensure_torch()
            module = torch.optim
        elif self.backend == Backend.TENSORFLOW:
            tf = ensure_tensorflow()
            module = tf.keras.optimizers
        else:
            raise BackendNotSupportedError(
                backend=self.backend,
                method="Optimizer._resolve()",
            )

        # Inspect available classes
        candidates: dict[str, type] = {}
        for attr_name in dir(module):
            try:
                attr = getattr(module, attr_name)
            except Exception:  # noqa: BLE001, S112
                continue
            if not isinstance(attr, type):
                continue

            # Match class names ignoring case
            candidates[attr_name.lower()] = attr

        # Resolve optimizer
        opt_cls = candidates.get(name_lc)
        if opt_cls is None:
            available = sorted(candidates.keys())
            msg = f"Unknown optimizer name '{self.name}' for backend '{self.backend}'. Available optimizers: {available}"
            raise OptimizerError(msg)

        return opt_cls

    def _check_optimizer(self):
        if not self.is_built:
            raise OptimizerNotSetError(message="Optimizer has not been built.")

    def _extract_kwargs_from_instance(self):
        if self.instance is None:
            raise ValueError("Instance cannot be None.")

        # If lazy backend, infer from instance
        if self._backend is None:
            self._backend = _safe_infer_backend(self.instance)

        # Extract kwargs, cls, and cls_name
        if self.backend == Backend.TORCH:
            self.kwargs = deepcopy(self.instance.defaults)
            self.cls = self.instance.__class__
            self.name = self.cls.__name__

        elif self.backend == Backend.TENSORFLOW:
            self.kwargs = deepcopy(self.instance.get_config())
            self.cls = self.instance.__class__
            self.name = self.cls.__name__

    # ================================================
    # Build
    # ================================================
    def build(
        self,
        *,
        parameters: Any | None = None,
        backend: Backend | None = None,
        force_rebuild: bool = False,
    ):
        """
        Instantiate the backend optimizer if not already provided.

        Args:
            parameters (any, optional):
                Trainable parameters of the model (required for PyTorch).
            backend (Backend | None):
                Backend to use for optimizer, if not already set.
            force_rebuild (bool, optional):
                Whether to force rebuild optimizer.

        """
        if self.is_built and not force_rebuild:
            msg = (
                "Optimizer.built() is being called on an already instantiated "
                "optimizer. If you want to rebuild the optimizer, set "
                "`force_rebuild=True`."
            )
            raise OptimizerNotSetError(message=msg)

        # Set/validate backend
        if backend is not None:
            if self.backend is not None and backend != self.backend:
                msg = (
                    "Backend passed to Optimizer.build differs from backend "
                    f"at init: {backend} != {self.backend}"
                )
                raise ValueError(msg)
            self.backend = backend
        if self.backend is None:
            raise ValueError("Backend must be set before building optimizer.")

        # Instantiate backend-specific optimizer
        # Case 1: class + kwargs
        if self.cls is not None:
            if self.backend == Backend.TORCH:
                if parameters is None:
                    raise ValueError(
                        "Torch Optimizer requires model parameters during build.",
                    )
                self.parameters = parameters
                self.instance = self.cls(self.parameters, **self.kwargs)

            elif self.backend == Backend.TENSORFLOW:
                self.parameters = None  # TF doesn't need parameters at construction
                self.instance = self.cls(**self.kwargs)

            else:
                raise BackendNotSupportedError(
                    backend=self._backend,
                    method="Optimizer.build()",
                )

        # Case 2: factory
        elif self._factory is not None:
            if self.backend == Backend.TORCH:
                if parameters is None:
                    raise ValueError("Torch optimizer factory requires parameters.")
                self.instance = self._factory(parameters)

            elif self.backend == Backend.TENSORFLOW:
                self.instance = self._factory(None)

            else:
                raise BackendNotSupportedError(
                    backend=self._backend,
                    method="Optimizer.build()",
                )

            # Extract self.cls, self.name, and self.kwargs from instance
            self._extract_kwargs_from_instance()

        else:
            raise RuntimeError("Unsupported initiatization state.")

        # If we have a pending internal state (from from_state), restore it now
        if self._pending_state is not None:
            self._restore_internal_state(self._pending_state)
            self._pending_state = None

    # ================================================
    # Backprop methods
    # ================================================
    def step(self, grads=None, variables=None):
        """
        Perform optimizer step.

        For PyTorch: calls `optimizer.step()`.
        For TensorFlow: requires `grads` and `variables` and applies gradients.
        """
        self._check_optimizer()

        if self._backend == Backend.TORCH:
            self.instance.step()

        elif self._backend == Backend.TENSORFLOW:
            if grads is None or variables is None:
                msg = (
                    "TensorFlow backend requires both `grads` and `variables` to be "
                    "set in Optimizer.step()."
                )
                raise ValueError(msg)
            self.instance.apply_gradients(zip(grads, variables, strict=True))

        else:
            raise BackendNotSupportedError(
                backend=self._backend,
                method="Optimizer.step()",
            )

    def zero_grad(self):
        """Resets the optimizer gradients."""
        self._check_optimizer()

        if self._backend == Backend.TORCH:
            self.instance.zero_grad()

        elif self._backend == Backend.TENSORFLOW:
            tf = ensure_tensorflow()
            for var in self.instance.variables():
                var.assign(tf.zeros_like(var))

        else:
            raise BackendNotSupportedError(
                backend=self._backend,
                method="Optimizer.zero_grad()",
            )

    # ================================================
    # Configurable
    # ================================================
    def get_config(self) -> dict[str, Any]:
        """Get config of this optimizer."""
        # Prefer to return only cls (str name) + kwargs
        if self.is_built:
            return {
                "opt": str(self.name).lower(),
                "opt_kwargs": self.kwargs,
                "backend": None
                if self.backend is None
                else str(self.backend.value).lower(),
            }

        return {
            "opt": None if self.name is None else str(self.name).lower(),
            "opt_kwargs": self.kwargs,
            "backend": None
            if self.backend is None
            else str(self.backend.value).lower(),
            "factory": self._factory,
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Optimizer:
        return cls(**config)

    # ================================================
    # Stateful
    # ================================================
    def get_state(self) -> dict[str, Any]:
        state = {"is_built": self.is_built}
        if self.is_built:
            state["internal"] = self._capture_internal_state()
        return state

    def set_state(self, state: dict[str, Any]) -> None:
        # Stash pending optimizer internal state; will be applied in build()
        if state.get("is_built"):
            self._pending_state = state.get("internal")

    # ================================================
    # Internal state handling
    # ================================================
    def _capture_internal_state(self) -> dict[str, Any] | None:
        """
        Capture backend-specific internal optimizer state.

        Returns:
            dict or None

        """
        if not self.is_built:
            return None

        if self._backend == Backend.TORCH:
            return {"state_dict": self.instance.state_dict()}

        if self._backend == Backend.TENSORFLOW:
            try:
                return {"weights": self.instance.get_weights()}
            except AttributeError:
                # Optimizer not yet initialized with variables
                return {"weights": None}

        return None

    def _restore_internal_state(self, state: dict[str, Any]) -> None:
        """
        Restore backend-specific internal optimizer state.

        Assumes the backend optimizer (`self.instance`) has already been
        constructed and attached to the correct parameters/variables.
        """
        if not self.is_built or state is None:
            return

        if self._backend == Backend.TORCH:
            d_state = state.get("state_dict")
            if d_state is not None:
                self.instance.load_state_dict(d_state)

        elif self._backend == Backend.TENSORFLOW:
            weights = state.get("weights")
            if weights is not None:
                self.instance.set_weights(weights)

        else:
            raise BackendNotSupportedError(
                backend=self._backend,
                method="Optimizer._restore_internal_state()",
            )

    # ================================================
    # Serialization
    # ================================================
    def save(self, filepath: Path, *, overwrite: bool = False) -> Path:
        """
        Serializes this Optimizer to the specified filepath.

        Args:
            filepath (Path):
                File location to save to. Note that the suffix may be overwritten
                to enforce the ModularML file extension schema.
            overwrite (bool, optional):
                Whether to overwrite any existing file at the save location.
                Defaults to False.

        Returns:
            Path: The actual filepath to write the Optimizer is saved.

        """
        from modularml.core.io.serialization_policy import SerializationPolicy
        from modularml.core.io.serializer import serializer

        return serializer.save(
            self,
            filepath,
            policy=SerializationPolicy.BUILTIN,
            overwrite=overwrite,
        )

    @classmethod
    def load(cls, filepath: Path, *, allow_packaged_code: bool = False) -> Optimizer:
        """
        Load a Optimizer from file.

        Args:
            filepath (Path):
                File location of a previously saved Optimizer.
            allow_packaged_code : bool
                Whether bundled code execution is allowed.

        Returns:
            Optimizer: The reloaded optimizer.

        """
        from modularml.core.io.serializer import _enforce_file_suffix, serializer

        # Append proper sufficx only if no suffix is given
        if Path(filepath).suffix == "":
            filepath = _enforce_file_suffix(path=filepath, cls=cls)

        return serializer.load(filepath, allow_packaged_code=allow_packaged_code)
