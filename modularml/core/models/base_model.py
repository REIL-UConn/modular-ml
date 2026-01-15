from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from modularml.core.io.protocols import Configurable, Stateful
from modularml.core.io.symbol_registry import symbol_registry
from modularml.utils.data.comparators import deep_equal
from modularml.utils.nn.backend import Backend, normalize_backend


class BaseModel(Configurable, Stateful, ABC):
    def __init__(self, backend: str | Backend, **init_args: Any):
        super().__init__()
        self._backend = normalize_backend(backend)
        self._built = False

        # Store init args for config round-trip
        self._init_args = dict(init_args)

    def __eq__(self, other):
        if not isinstance(other, BaseModel):
            msg = f"Cannot compare equality between BaseModel and {type(other)}"
            raise TypeError(msg)

        # Compare config (replace class object with name)
        def grab_clean_cfg(x: Configurable):
            cfg = x.get_config()
            # Replace class with qualname (str)
            if "model_class" in cfg:
                cls_or_obj = cfg["model_class"]
                cls = (
                    cls_or_obj if isinstance(cls_or_obj, type) else cls_or_obj.__class__
                )
                cfg["model_class"] = cls.__qualname__

        if not deep_equal(grab_clean_cfg(self), grab_clean_cfg(other)):
            return False

        # Compare state
        return deep_equal(self.get_state(), other.get_state())

    def __hash__(self) -> int:
        return id(self)

    def __repr__(self) -> str:
        return f"<{self.backend.value} model>"

    # ================================================
    # Properties
    # ================================================
    @property
    @abstractmethod
    def input_shape(self) -> tuple[int, ...] | None:
        """
        Shape of the expected input to this mode.

        Returns:
            tuple[int, ...] | None: Input shape tuple (e.g., (32, 64)) or None if unknown.

        """

    @property
    @abstractmethod
    def output_shape(self) -> tuple[int, ...] | None:
        """
        Shape of the output produced by the model.

        Returns:
            tuple[int, ...] | None: Output shape tuple (e.g., (32, 10)) or None if unknown.

        """

    @property
    def backend(self) -> Backend:
        return self._backend

    @property
    def is_built(self) -> bool:
        return (
            self._built
            and (self.input_shape is not None)
            and (self.output_shape is not None)
        )

    # ================================================
    # Methods
    # ================================================
    @abstractmethod
    def build(
        self,
        input_shape: tuple[int, ...] | None = None,
        output_shape: tuple[int, ...] | None = None,
        *,
        force: bool = False,
    ):
        """Build the internal model layers given an input shape."""

    @abstractmethod
    def forward(self, *args, **kwargs):
        """Forward pass."""

    def __call__(self, *args, **kwargs):
        """Run a forward pass."""
        return self.forward(*args, **kwargs)

    # ================================================
    # Model Weights & Initiallize Params
    # ================================================
    @abstractmethod
    def get_weights(self) -> dict[str, Any]:
        """
        Return all backend weights in a pure-python dict form.

        Must return pure-Python or numpy types only.
        """

    @abstractmethod
    def set_weights(self, weights: dict[str, Any]) -> None:
        """Restore weights from `get_weights()` dict."""

    def get_init_args(self) -> dict[str, Any]:
        """
        Return keyword arguments needed to reconstruct this model.

        Subclasses may override this if they want stricter control,
        but storing init_args automatically works well for most models.
        """
        return dict(self._init_args)

    # ================================================
    # Configurable
    # ================================================
    def get_config(self) -> dict[str, Any]:
        """
        Return configuration required to re-instantiate this model.

        Returns:
            dict[str, Any]: Model configuration.

        """
        return {
            "model_class": type(self),
            "init_args": self.get_init_args(),
            "backend": self.backend.value,
            "is_built": self._built,
            "input_shape": self.input_shape,
            "output_shape": self.output_shape,
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> BaseModel:
        """
        Construct a model from configuration data.

        Description:
            This method instantiates a new model with the given config
            data. It *does not* restore the model state/learned weights.

        Args:
            config (dict[str, Any]): Model configuration.

        Returns:
            BaseModel: New BaseModel instance.

        """
        model_cls = config["model_class"]
        model: BaseModel = model_cls(backend=config["backend"], **config["init_args"])

        if config["is_built"]:
            model.build(
                input_shape=config.get("input_shape"),
                output_shape=config.get("output_shape"),
                force=True,
            )
        return model

    # ================================================
    # Stateful
    # ================================================
    def get_state(self) -> dict[str, Any]:
        """
        Return learned state/weights of the model.

        Returns:
            dict[str, Any]: Learned state.

        """
        state: dict[str, Any] = {
            "weights": self.get_weights(),
        }
        return state

    def set_state(self, state: dict[str, Any]) -> None:
        """
        Restore learned state of the scaler.

        Args:
            state (dict[str, Any]):
                State produced by get_state().

        """
        self.set_weights(state["weights"])

    # ================================================
    # Serialization
    # ================================================
    def save(self, filepath: Path, *, overwrite: bool = False) -> Path:
        """
        Serializes this model to the specified filepath.

        Args:
            filepath (Path):
                File location to save to. Note that the suffix may be overwritten
                to enforce the ModularML file extension schema.
            overwrite (bool, optional):
                Whether to overwrite any existing file at the save location.
                Defaults to False.

        Returns:
            Path: The actual filepath where the model artifact is saved.

        """
        from modularml.core.io.serialization_policy import SerializationPolicy
        from modularml.core.io.serializer import serializer

        policy = SerializationPolicy.PACKAGED
        if symbol_registry.obj_is_a_builtin_class(self):
            policy = SerializationPolicy.BUILTIN
        elif symbol_registry.obj_in_a_builtin_registry(
            obj_or_cls=self,
            registry_name="model_registry",
        ):
            policy = SerializationPolicy.REGISTERED

        return serializer.save(
            self,
            filepath,
            policy=policy,
            overwrite=overwrite,
        )

    @classmethod
    def load(cls, filepath: Path, *, allow_packaged_code: bool = False) -> BaseModel:
        """
        Load a BaseModel from file.

        Args:
            filepath (Path):
                File location of a previously saved BaseModel.
            allow_packaged_code : bool
                Whether bundled code execution is allowed.

        Returns:
            BaseModel: The reloaded model.

        """
        from modularml.core.io.serializer import _enforce_file_suffix, serializer

        # Append proper sufficx only if no suffix is given
        if Path(filepath).suffix == "":
            filepath = _enforce_file_suffix(path=filepath, cls=cls)

        return serializer.load(filepath, allow_packaged_code=allow_packaged_code)
