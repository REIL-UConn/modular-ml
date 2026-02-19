"""Base handler implementations for encoding and decoding IO artifacts."""

from __future__ import annotations

import dataclasses
import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Generic, TypeVar

from modularml.core.experiment.experiment_node import ExperimentNode
from modularml.core.io.packaged_code_loaders.default_loader import (
    default_packaged_code_loader,
)
from modularml.core.io.protocols import Configurable, Stateful
from modularml.core.io.serialization_policy import SerializationPolicy
from modularml.core.io.symbol_registry import symbol_registry
from modularml.core.io.symbol_spec import SymbolSpec
from modularml.utils.logging.logger import get_logger

if TYPE_CHECKING:
    from modularml.core.io.serializer import LoadContext, SaveContext

T = TypeVar("T")

logger = get_logger(level=logging.INFO)


class BaseHandler(Generic[T]):
    """
    Serialize and deserialize configuration and state for related object types.

    Attributes:
        object_version (str): Semantic version of artifacts emitted by the handler.

    """

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
        Encode configuration and state artifacts for `obj`.

        Args:
            obj (T): Object instance to encode.
            save_dir (Path): Directory where files will be written.
            ctx (SaveContext): Active :class:`SaveContext` supplied by the serializer.

        Returns:
            dict[str, str | None]: Mapping from logical keys (config/state) to saved file names.

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
        Encode a configuration dictionary to JSON.

        Args:
            obj (T): Configurable object providing :meth:`get_config`.
            save_dir (Path): Directory where files will be written.
            ctx (SaveContext): Active :class:`SaveContext`.
            config_rel_path (str): Relative path for the saved JSON file.

        Returns:
            dict[str, str]: Mapping with a `config` key referencing the JSON file.

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
        Encode :class:`Stateful` object state into a pickle file.

        Args:
            obj (T): Object implementing :class:`Stateful`.
            save_dir (Path): Directory where files will be written.
            ctx (SaveContext): Active :class:`SaveContext`.
            state_rel_path (str): Relative path for the serialized state file.

        Returns:
            dict[str, str]: Mapping with a `state` key referencing the pickle file.

        Raises:
            NotImplementedError: If `obj` does not implement :class:`Stateful`.

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
        Reconstruct an object from serialized config and state files.

        Args:
            cls (type[T]): Class or factory providing :meth:`from_config`.
            load_dir (Path): Directory containing the saved artifact.
            ctx (LoadContext): Active :class:`LoadContext`.

        Returns:
            T: Re-instantiated object populated with decoded state.

        """
        obj = None

        # Instantiate from config (if defined)
        config = {"config": None}
        if isinstance(cls, Configurable):
            json_cfg = self.decode_config(load_dir=load_dir, ctx=ctx)

            # Restore any non-JSON elements
            config = self._restore_json_cfg(data=json_cfg, ctx=ctx)

            # Delay node registration, if supported
            # Registration will be handled by collision checking later
            if isinstance(cls, ExperimentNode):
                obj = cls.from_config(config, register=False)
            else:
                obj = cls.from_config(config)
        else:
            obj = cls()

        # Restore state (if defined)
        if isinstance(obj, Stateful):
            state = self.decode_state(load_dir=load_dir, ctx=ctx)
            obj.set_state(state)

        # Collision checking
        obj = self.handle_node_collision(obj=obj, ctx=ctx)

        return obj

    def decode_config(
        self,
        load_dir: Path,
        *,
        ctx: LoadContext,  # noqa: ARG002
        config_rel_path: str = "config.json",
    ) -> dict[str, Any] | None:
        """
        Load configuration JSON from `load_dir`.

        Args:
            load_dir (Path): Directory containing the saved artifact.
            ctx (LoadContext): Active :class:`LoadContext`.
            config_rel_path (str): Relative path to the config JSON file.

        Returns:
            dict[str, Any] | None: Parsed configuration dictionary.

        Raises:
            FileNotFoundError: If the configuration file is missing.

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
        Load serialized state data from `load_dir`.

        Args:
            load_dir (Path): Directory containing the saved artifact.
            ctx (LoadContext): Active :class:`LoadContext`.
            state_rel_path (str): Relative path to the pickle file.

        Returns:
            dict[str, Any]: Deserialized state mapping.

        Raises:
            FileNotFoundError: If the state file does not exist.

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
    def get_symbol_spec(self, obj_or_cls: Any, *, ctx: SaveContext) -> SymbolSpec:
        """
        Create a :class:`SymbolSpec` describing how to reload `obj_or_cls`.

        Description:
            Builtin and registry-backed classes are recorded by reference. Unregistered
            symbols are packaged via the provided :class:`SaveContext` to allow
            future reloading.

        Args:
            obj_or_cls (Any): Object instance or class requiring serialization metadata.
            ctx (SaveContext): Context used for packaging custom code.

        Returns:
            SymbolSpec: Metadata describing symbol provenance.

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

    def handle_node_collision(self, obj: Any, *, ctx: LoadContext) -> Any:
        """
        Handle Experiment node ID collisions during deserialization.

        Description:
            - If a node with the same ID and state exists, reuse it.
            - If :attr:`LoadContext.overwrite_collision` is True, replace the existing node.
            - Otherwise, assign a new node ID before registering the reloaded node.

        Args:
            obj (Any): Object potentially representing an :class:`ExperimentNode`.
            ctx (LoadContext): Active :class:`LoadContext`.

        Returns:
            Any: Either the existing registered node or the adjusted object.

        """
        from modularml.core.experiment.experiment_context import ExperimentContext
        from modularml.core.experiment.experiment_node import (
            ExperimentNode,
            generate_node_id,
        )

        # If not ExperimentNode, return object
        if not isinstance(obj, ExperimentNode):
            return obj

        # ------------------------------------------------
        # Collision Checking
        #  If same node_id exists, perform the following checks:
        #  1. If same label + same state -> reuse existing
        #  2. If different label or different state -> override or fork
        #     - Override = replace existing node_id reference in ExperimentContext
        #       with new object
        #     - Fork = generate new node_id for reloaded object and register
        # ------------------------------------------------
        exp_ctx: ExperimentContext = ExperimentContext.get_active()
        if exp_ctx.has_node(node_id=obj.node_id):
            cls_name = type(obj).__qualname__
            existing = exp_ctx.get_node(node_id=obj.node_id)

            # Case 1
            if isinstance(existing, type(obj)) and existing == obj:
                # Early-return existing node
                msg = (
                    f"Loaded {cls_name} is identical to '{existing.label}' in "
                    f"the existing ExperimentContext. Returning '{existing}'."
                )
                logger.info(
                    msg=msg,
                    extra={
                        "title_desc": "Node ID Collision",
                        "omit_origin": True,
                    },
                )
                return existing

            # Case 2
            if ctx.overwrite_collision:
                # Remove existing from ExperimentContext
                msg = (
                    f"The loaded {cls_name} has an overlapping node ID with existing "
                    f"{cls_name} '{existing.label}'. '{existing.label}' will be "
                    "overwritten in the active ExperimentContext."
                )
                logger.info(
                    msg=msg,
                    extra={
                        "title_desc": "Node ID Collision",
                        "omit_origin": True,
                    },
                )
                _ = exp_ctx.remove_node(
                    node_id=existing.node_id,
                    error_if_missing=True,
                )
            else:
                msg = (
                    f"The loaded {cls_name} has an overlapping node ID with existing "
                    f"{cls_name} '{existing.label}'. A new node ID will be assigned "
                    f"to the loaded {cls_name}."
                )
                logger.info(
                    msg=msg,
                    extra={
                        "title_desc": "Node ID Collision",
                        "omit_origin": True,
                    },
                )
                # Update node_id of new node
                obj._node_id = generate_node_id()

        # Register new node (we allow same labels)
        exp_ctx.register_experiment_node(
            node=obj,
            check_label_collision=False,
        )

        return obj

    def handle_model_graph_collision(
        self,
        obj: Any,
        *,
        ctx: LoadContext,
    ) -> Any:
        """
        Handle collisions for nodes associated with a :class:`ModelGraph`.

        Description:
            Reuses identical nodes, overwrites existing registrations when allowed,
            or assigns a fresh node ID when necessary.

        Args:
            obj (Any): Object potentially representing an :class:`ExperimentNode`.
            ctx (LoadContext): Active :class:`LoadContext`.

        Returns:
            Any: Either the existing registered node or the adjusted object.

        """
        from modularml.core.experiment.experiment_context import ExperimentContext
        from modularml.core.experiment.experiment_node import (
            ExperimentNode,
            generate_node_id,
        )

        # If not ExperimentNode, return object
        if not isinstance(obj, ExperimentNode):
            return obj

        # ------------------------------------------------
        # Collision Checking
        #  If same node_id exists, perform the following checks:
        #  1. If same label + same state -> reuse existing
        #  2. If different label or different state -> override or fork
        #     - Override = replace existing node_id reference in ExperimentContext
        #       with new object
        #     - Fork = generate new node_id for reloaded object and register
        # ------------------------------------------------
        exp_ctx: ExperimentContext = ExperimentContext.get_active()
        if exp_ctx.has_node(node_id=obj.node_id):
            cls_name = type(obj).__qualname__
            existing = exp_ctx.get_node(node_id=obj.node_id)

            # Case 1
            if isinstance(existing, type(obj)) and existing == obj:
                # Early-return existing node
                msg = (
                    f"Loaded {cls_name} is identical to '{existing.label}' in "
                    f"the existing ExperimentContext. Returning '{existing}'."
                )
                logger.info(
                    msg=msg,
                    extra={
                        "title_desc": "Node ID Collision",
                        "omit_origin": True,
                    },
                )
                return existing

            # Case 2
            if ctx.overwrite_collision:
                # Remove existing from ExperimentContext
                msg = (
                    f"The loaded {cls_name} has an overlapping ID with existing "
                    f"{cls_name} '{existing.label}'. '{existing.label}' will be "
                    "overwritten in the active ExperimentContext."
                )
                logger.info(
                    msg=msg,
                    extra={
                        "title_desc": "Node ID Collision",
                        "omit_origin": True,
                    },
                )
                _ = exp_ctx.remove_node(
                    node_id=existing.node_id,
                    error_if_missing=True,
                )
            else:
                msg = (
                    f"The loaded {cls_name} has an overlapping node ID with existing "
                    f"{cls_name} '{existing.label}'. A new node ID will be assigned "
                    f"to the loaded {cls_name}."
                )
                logger.info(
                    msg=msg,
                    extra={
                        "title_desc": "Node ID Collision",
                        "omit_origin": True,
                    },
                )
                # Update node_id of new node
                obj._node_id = generate_node_id()

        # Register new node (we allow same labels)
        exp_ctx.register_experiment_node(
            node=obj,
            check_label_collision=False,
        )

        return obj

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
        register: bool = True,
    ):
        """
        Restore non-JSON objects encoded by :meth:`_json_safe_cfg`.

        Args:
            data (dict[str, Any]): JSON-safe configuration to restore.
            ctx (LoadContext): Active :class:`LoadContext`.
            register (bool): Whether newly constructed nodes should register with the context.

        Returns:
            dict[str, Any]: Configuration with live objects reconstructed.

        """

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
                cls: Configurable = symbol_registry.resolve_symbol(
                    spec=SymbolSpec.from_dict(val["class"]),
                    allow_packaged_code=ctx.allow_packaged_code,
                    packaged_code_loader=lambda source_ref: default_packaged_code_loader(
                        artifact_path=ctx.artifact_path,
                        source_ref=source_ref,
                        allow_packaged=ctx.allow_packaged_code,
                    ),
                )
                config = self._restore_json_cfg(data=val["config"], ctx=ctx)

                if isinstance(cls, ExperimentNode):
                    obj = cls.from_config(config, register=register)
                else:
                    obj = cls.from_config(config)

                return obj

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
        """
        Write `data` to `save_path` as JSON.

        Args:
            data (dict[str, Any]): JSON-serializable payload.
            save_path (Path): Output file path (suffix enforced to `.json`).

        Returns:
            Path: Path to the written JSON file.

        """
        path = Path(save_path).with_suffix(".json")
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True)
        return path

    def _read_json(self, data_path: Path) -> Any:
        """
        Read JSON content from `data_path`.

        Args:
            data_path (Path): File path containing JSON data.

        Returns:
            Any: Parsed JSON payload.

        """
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
        Associate a :class:`BaseHandler` with a class (usually a base class).

        Args:
            cls (type): Class the handler should manage.
            handler (BaseHandler): Handler instance.
            overwrite (bool): Overwrite an existing mapping when True.

        Raises:
            ValueError: If a handler is already registered and `overwrite` is False.

        """
        if not overwrite and cls in self._handlers:
            msg = f"Handler already registered for {cls.__name__}."
            raise ValueError(msg)
        self._handlers[cls] = handler

    def resolve(self, cls: type) -> BaseHandler:
        """
        Return the most specific handler for `cls` using MRO traversal.

        Args:
            cls (type): Class requiring a handler.

        Returns:
            BaseHandler: Resolved handler instance, or a default :class:`BaseHandler`.

        """
        for base in cls.__mro__:
            if base in self._handlers:
                return self._handlers[base]
        # Default handler fallback
        return BaseHandler()
