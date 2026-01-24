from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Generic, TypeVar

from modularml.core.io.packaged_code_loaders.default_loader import (
    default_packaged_code_loader,
)
from modularml.core.io.protocols import Configurable, Stateful
from modularml.core.io.serialization_policy import SerializationPolicy
from modularml.core.io.symbol_registry import symbol_registry
from modularml.core.io.symbol_spec import SymbolSpec

if TYPE_CHECKING:
    from modularml.core.io.serializer import LoadContext, SaveContext

T = TypeVar("T")


class BaseHandler(Generic[T]):
    """Base handler for encoding/decoding config and state for a family of objects."""

    object_version: ClassVar[str] = "1.0"

    # ================================================
    # Object encoding
    # ================================================
    def encode(
        self,
        obj: T,
        save_dir: Path,
        *,
        ctx: SaveContext,
    ) -> dict[str, str | None]:
        """
        Encodes state and config to files.

        Args:
            obj (T):
                Object instance to encode.
            save_dir (Path):
                Directory to write encodings to.
            ctx (SaveContext, optional):
                Additional serialization context.

        Returns:
            dict[str, str | None]: Mapping of "config" and "state" keys to saved files.

        """
        file_mapping = self.encode_config(obj=obj, save_dir=save_dir, ctx=ctx)
        file_mapping.update(self.encode_state(obj=obj, save_dir=save_dir, ctx=ctx))
        return file_mapping

    def encode_config(
        self,
        obj: T,
        save_dir: Path,
        *,
        ctx: SaveContext,
        config_rel_path: str = "config.json",
    ) -> dict[str, str]:
        """
        Encodes config to a json file.

        Args:
            obj (T):
                Object to encode config for.
            save_dir (Path):
                Directory to write encodings to.
            ctx (SaveContext, optional):
                Additional serialization context.
            config_rel_path (str):
                Relative path to saved config file.
                Defaults to "config.json"

        Returns:
            dict[str, str]: Mapping of config to saved json file

        """
        if not isinstance(obj, Configurable):
            return {"config": None}

        config = obj.get_config()

        # Ensure config is JSON-safe
        json_data = self._json_safe_cfg(config=config, ctx=ctx)

        # Save config to file
        path = self._write_json(json_data, Path(save_dir) / config_rel_path)
        return {"config": path.name}

    def encode_state(
        self,
        obj: T,
        save_dir: Path,
        *,
        ctx: SaveContext,  # noqa: ARG002
        state_rel_path: str = "state.pkl",
    ) -> dict[str, str]:
        """
        Encodes object state to a pickle file.

        Args:
            obj (T):
                Object to encode state for.
            save_dir (Path):
                Directory to write encodings to.
            ctx (SaveContext, optional):
                Additional serialization context.
            state_rel_path (str):
                Relative path to save state to.
                Defaults to "state.pkl"

        Returns:
            dict[str, str]: Mapping of state to saved pkl file

        """
        import pickle

        if not isinstance(obj, Stateful):
            return {"state": None}

        state = obj.get_state()

        # Save state to file
        file_state = Path(save_dir) / state_rel_path
        with Path.open(file_state, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

        return {"state": state_rel_path}

    # ================================================
    # Object decoding
    # ================================================
    def decode(
        self,
        cls: type[T],
        load_dir: Path,
        *,
        ctx: LoadContext,
    ) -> T:
        """
        Decodes an object from a saved artifact.

        Description:
            Instantiates an object (instantiates from config and sets state).

        Args:
            cls (type[T]):
                Load config for class.
            load_dir (Path):
                Directory to decode from.
            ctx (LoadContext, optional):
                Additional de-serialization context.

        Returns:
            T: The re-instantiated object.

        """
        obj = None

        # Instantiate from config (if defined)
        config = {"config": None}
        if isinstance(cls, Configurable):
            json_cfg = self.decode_config(load_dir=load_dir, ctx=ctx)

            # Restore any non-JSON elements
            config = self._restore_json_cfg(
                data=json_cfg,
                ctx=ctx,
            )
            obj = cls.from_config(config)
        else:
            obj = cls()

        # Restore state (if defined)
        if isinstance(obj, Stateful):
            state = self.decode_state(load_dir=load_dir, ctx=ctx)
            obj.set_state(state)

        return obj

    def decode_config(
        self,
        load_dir: Path,
        *,
        ctx: LoadContext,  # noqa: ARG002
        config_rel_path: str = "config.json",
    ) -> dict[str, Any] | None:
        """
        Decodes config from a json file.

        Args:
            load_dir (Path):
                Directory to decode from.
            ctx (LoadContext, optional):
                Additional de-serialization context.
            config_rel_path (str):
                Relative path to saved config file.
                Defaults to "config.json"

        Returns:
            dict[str, Any] | None: The decoded config data.

        """
        # Check that config.json exists
        file_config = Path(load_dir) / config_rel_path
        if not file_config.exists():
            msg = f"Could not find config file in directory: '{file_config}'."
            raise FileNotFoundError(msg)

        # Read config
        config = self._read_json(file_config)
        return config

    def decode_state(
        self,
        load_dir: Path,
        *,
        ctx: LoadContext,  # noqa: ARG002
        state_rel_path: str = "state.pkl",
    ) -> dict[str, Any]:
        """
        Decodes state from a pkl file.

        Args:
            load_dir (Path):
                Directory to decode from.
            ctx (LoadContext, optional):
                Additional de-serialization context.
            state_rel_path (str):
                Relative path to save state to.
                Defaults to "state.pkl"

        Returns:
            dict[str, Any]: The decoded state data.

        """
        import pickle

        # Check that config.json exists
        file_state = Path(load_dir) / state_rel_path
        if not file_state.exists():
            msg = f"Could not find state file in directory: '{file_state}'."
            raise FileNotFoundError(msg)

        # Read config
        with Path.open(file_state, "rb") as f:
            state: dict[str, Any] = pickle.load(f)

        return state

    # ================================================
    # Convenience Methods
    # ================================================
    def has_config(self, obj: T) -> bool:
        """
        Return True if object has config that should be serialized.

        Args:
            obj (T): Object to inspect.

        Returns:
            bool: True if Configurable.

        """
        return isinstance(obj, Configurable)

    def has_state(self, obj: T) -> bool:
        """
        Return True if object has state that should be serialized.

        Args:
            obj (T): Object to inspect.

        Returns:
            bool: True if Stateful.

        """
        return isinstance(obj, Stateful)

    def get_symbol_spec(self, obj_or_cls: Any, *, ctx: SaveContext) -> SymbolSpec:
        """
        Gets the SymbolSpec instance for a given object and SaveContext.

        Description:
            If the object is a BUILTIN or REGISTERED class (or an instance of one), a
            SymbolSpec is returned mapping this object to its known class.
            If the object is created using unknown source code (e.g., a custom Torch
            model), the source code is packaged and saved to file under the directory
            specified in `ctx`. A SymbolSpec is then created mapping this objects class
            to the saved file for later deserialization.

            The return SymbolSpec can be cast to a JSON-safe dict using
            `SymbolSpec.to_dict()`.

        """
        # Default to packaged code (safe fallback)
        policy = SerializationPolicy.PACKAGED
        if symbol_registry.obj_is_a_builtin_class(obj_or_cls):
            policy = SerializationPolicy.BUILTIN
        elif symbol_registry.obj_in_a_builtin_registry(
            obj_or_cls=obj_or_cls,
            registry_name=None,
        ):
            policy = SerializationPolicy.REGISTERED

        # Create spec (internally packages code only if needed)
        return ctx.make_symbol_spec(symbol=obj_or_cls, policy=policy)

    # ================================================
    # JSON-based config encode/decode
    # ================================================
    def _json_safe_cfg(
        self,
        config: dict[Any, Any],
        *,
        ctx: SaveContext,
    ) -> dict[str, Any]:
        """
        Parses raw config data, returning a JSON-safe dictionary.

        Args:
            config (dict[Any, Any]):
                A dictionary returned from `get_config()` of Configurable objects.
                The dictionary can contain non-JSON items like classes, instances,
                etc. Note that the keys must already be JSON safe (i.e. strings).

            ctx (SaveContext):
                The SaveContext to used during this safe casting. Any non-JSON
                elements may need to be packaged into source code and will be
                saved under the directory specified in `ctx`.

        Returns:
            dict[str, Any]:
                A JSON safe dict of the original config data.

        """

        def _json_safe(obj: Any) -> Any:
            """Returns JSON safe version of some value."""
            # Primitives
            if obj is None or isinstance(obj, (bool, int, float, str)):
                return obj

            # Configurable instances
            if not isinstance(obj, type) and isinstance(obj, Configurable):
                return {
                    "__mml_object__": True,
                    "class": self.get_symbol_spec(obj_or_cls=obj, ctx=ctx).to_dict(),
                    "config": self._json_safe_cfg(obj.get_config(), ctx=ctx),
                }

            # Dicts
            if isinstance(obj, Mapping):
                return {str(k): _json_safe(x) for k, x in obj.items()}

            # Tuples -> need to be able to ensure tuple after reload
            if isinstance(obj, tuple):
                return {
                    "__type__": "tuple",
                    "items": [_json_safe(x) for x in obj],
                }

            # Lists
            if isinstance(obj, list):
                return [_json_safe(x) for x in obj]

            # Dataclasses
            if dataclasses.is_dataclass(obj):
                if hasattr(obj, "to_dict"):
                    return _json_safe(obj.to_dict())
                return _json_safe(dataclasses.asdict(obj))

            # Classes or functions
            spec = self.get_symbol_spec(obj_or_cls=obj, ctx=ctx)
            return spec.to_dict()

        json_data: dict[str, Any] = {}
        for k, v in config.items():
            # Enforce string keys
            if not isinstance(k, str):
                msg = f"Config keys must be strings. Received: {k} of type {type(k)}."
                raise TypeError(msg)
            json_data[k] = _json_safe(v)

        return json_data

    def _restore_json_cfg(
        self,
        data: dict[str, Any],
        *,
        ctx: LoadContext,
    ):
        """Restores any non-JSON items in the config file."""

        def _restore_val(val: Any) -> Any:
            # If a SymbolSpec dict
            if SymbolSpec.is_symbol_spec_dict(val):
                return symbol_registry.resolve_symbol(
                    spec=SymbolSpec.from_dict(val),
                    allow_packaged_code=ctx.allow_packaged_code,
                    packaged_code_loader=lambda source_ref: default_packaged_code_loader(
                        artifact_path=ctx.artifact_path,
                        source_ref=source_ref,
                        allow_packaged=ctx.allow_packaged_code,
                    ),
                )

            # If a configurable instance
            if isinstance(val, dict) and val.get("__mml_object__"):
                cls = symbol_registry.resolve_symbol(
                    spec=SymbolSpec.from_dict(val["class"]),
                    allow_packaged_code=ctx.allow_packaged_code,
                    packaged_code_loader=lambda source_ref: default_packaged_code_loader(
                        artifact_path=ctx.artifact_path,
                        source_ref=source_ref,
                        allow_packaged=ctx.allow_packaged_code,
                    ),
                )
                config = self._restore_json_cfg(data=val["config"], ctx=ctx)
                return cls.from_config(config)

            # If a general mapping instance
            if isinstance(val, Mapping):
                if (
                    isinstance(val, dict)
                    and "__type__" in val
                    and val["__type__"] == "tuple"
                ):
                    return tuple(_restore_val(x) for x in val["items"])

                return {k: _restore_val(x) for k, x in val.items()}

            # If a list
            if isinstance(val, (tuple, list)):
                return [_restore_val(x) for x in val]

            return val

        restored_cfg: dict[str, Any] = {}
        for k, v in data.items():
            restored_cfg[k] = _restore_val(v)

        return restored_cfg

    def _write_json(self, data: dict[str, Any], save_path: Path) -> Path:
        """Saves `data` to `path` as json."""
        path = Path(save_path).with_suffix(".json")
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True)
        return path

    def _read_json(self, data_path: Path) -> Any:
        """Reads `data` from `data_path` as json."""
        with Path(data_path).open("r", encoding="utf-8") as f:
            config = json.load(f)
        return config


class HandlerRegistry:
    """Registry for mapping base classes to BaseHandlers with MRO resolution."""

    def __init__(self):
        self._handlers: dict[type, BaseHandler] = {}

    def register(
        self,
        cls: type,
        handler: BaseHandler,
        *,
        overwrite: bool = False,
    ) -> None:
        """
        Register a handler for a class (typically a base class).

        Args:
            cls (type): Class to register the handler for.
            handler (BaseHandler): Handler instance.
            overwrite (bool): Overwrite existing mapping if True.

        """
        if not overwrite and cls in self._handlers:
            msg = f"Handler already registered for {cls.__name__}."
            raise ValueError(msg)
        self._handlers[cls] = handler

    def resolve(self, cls: type) -> BaseHandler:
        """
        Resolve a handler using MRO search.

        Args:
            cls (type): Class to resolve.

        Returns:
            BaseHandler: Matching handler.

        """
        for base in cls.__mro__:
            if base in self._handlers:
                return self._handlers[base]
        # Default handler fallback
        return BaseHandler()
