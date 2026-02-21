"""Abstract base class definitions for ModularML model backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from modularml.core.io.protocols import Configurable, Stateful
from modularml.core.io.symbol_registry import symbol_registry
from modularml.utils.data.comparators import deep_equal
from modularml.utils.nn.backend import Backend, normalize_backend


class BaseModel(Configurable, Stateful, ABC):
    """
    Abstract base class for backend-agnostic ModularML models.

    Attributes:
        _backend (Backend): Normalized backend enum powering the model.
        _built (bool): Flag indicating whether :meth:`build` completed.
        _init_args (dict[str, Any]): Keyword arguments stored for config
            round-trips via :meth:`get_config`.

    """

    def __init__(self, backend: str | Backend, **init_args: Any):
        """
        Initialize the model with a backend and keyword arguments.

        Args:
            backend (str | Backend): Backend identifier or enum value used
                to normalize :attr:`_backend`.
            **init_args (Any): Keyword arguments cached for config
                serialization so subclasses can reconstruct themselves.

        """
        super().__init__()
        self._backend = normalize_backend(backend)
        self._built = False
        self._init_args = dict(init_args)

    def __eq__(self, other):
        """Return True when configs and states match another model."""
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
        """Return identity-based hash so models remain hashable."""
        return id(self)

    def __repr__(self) -> str:
        """Return string representation including backend value."""
        return f"<{self.backend.value} model>"

    # ================================================
    # Properties
    # ================================================
    @property
    @abstractmethod
    def input_shape(self) -> tuple[int, ...] | None:
        """
        Return the expected per-sample input shape for the model.

        Returns:
            tuple[int, ...] | None: Shape tuple without the batch dimension,
                or None when the shape is not yet known.

        """

    @property
    @abstractmethod
    def output_shape(self) -> tuple[int, ...] | None:
        """
        Return the per-sample output shape produced by the model.

        Returns:
            tuple[int, ...] | None: Shape tuple without the batch dimension,
                or None when not yet known.

        """

    @property
    def backend(self) -> Backend:
        """Return the :class:`Backend` powering this model."""
        return self._backend

    @property
    def is_built(self) -> bool:
        """Return True when the model has been built and shapes resolved."""
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
        """
        Build backend layers for the model implementation.

        Args:
            input_shape (tuple[int, ...] | None): Per-sample input shape,
                excluding batch dimension, if known.
            output_shape (tuple[int, ...] | None): Per-sample output shape,
                excluding batch dimension, if known.
            force (bool): Whether to rebuild even if the model is already
                constructed.

        """

    @abstractmethod
    def forward(self, *args, **kwargs):
        """Run a forward pass for the underlying backend."""

    def __call__(self, *args, **kwargs):
        """Execute :meth:`forward` to make the instance callable."""
        return self.forward(*args, **kwargs)

    # ================================================
    # Model Weights & Initiallize Params
    # ================================================
    @abstractmethod
    def get_weights(self) -> dict[str, Any]:
        """Return backend weights as pure-Python or numpy data."""

    @abstractmethod
    def set_weights(self, weights: dict[str, Any]) -> None:
        """
        Restore the weights previously produced by :meth:`get_weights`.

        Args:
            weights (dict[str, Any]): Dictionary returned by
                :meth:`get_weights`.

        """

    def get_init_args(self) -> dict[str, Any]:
        """
        Return keyword arguments needed to reconstruct this model.

        Description:
            Subclasses may override this to control config serialization,
            but caching `init_args` automatically works for most models.

        Returns:
            dict[str, Any]: Copy of initialization keyword arguments.

        """
        return dict(self._init_args)

    # ================================================
    # Configurable
    # ================================================
    def get_config(self) -> dict[str, Any]:
        """
        Return configuration required to re-instantiate this model.

        Returns:
            dict[str, Any]: Serializable configuration capturing the model
                class, backend, and build metadata.

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
        Construct a :class:`BaseModel` from serialized configuration.

        Description:
            Instantiates a new model with the provided config. Learned
            weights are not restored; call :meth:`set_state` afterwards.

        Args:
            config (dict[str, Any]): Configuration emitted by
                :meth:`get_config`.

        Returns:
            BaseModel: Newly constructed model instance.

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
        """Return the learned state for serialization."""
        state: dict[str, Any] = {
            "weights": self.get_weights(),
        }
        return state

    def set_state(self, state: dict[str, Any]) -> None:
        """
        Restore learned state produced by :meth:`get_state`.

        Args:
            state (dict[str, Any]): State dictionary containing weights.

        """
        self.set_weights(state["weights"])

    # ================================================
    # Serialization
    # ================================================
    def save(self, filepath: Path, *, overwrite: bool = False) -> Path:
        """
        Serialize this model to disk using the serializer.

        Args:
            filepath (Path): Destination path. The suffix may be adjusted to
                follow ModularML naming conventions.
            overwrite (bool): Whether to overwrite an existing artifact.

        Returns:
            Path: Actual file path written by the serializer.

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
        Load a serialized :class:`BaseModel` from disk.

        Args:
            filepath (Path): Path pointing to a saved model.
            allow_packaged_code (bool): Whether packaged code execution is
                permitted for user-defined artifacts.

        Returns:
            BaseModel: Reloaded model instance.

        """
        from modularml.core.io.serializer import _enforce_file_suffix, serializer

        # Append proper sufficx only if no suffix is given
        if Path(filepath).suffix == "":
            filepath = _enforce_file_suffix(path=filepath, cls=cls)

        return serializer.load(filepath, allow_packaged_code=allow_packaged_code)
