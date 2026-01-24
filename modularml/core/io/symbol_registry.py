from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from modularml.core.io.serialization_policy import SerializationPolicy
from modularml.utils.io.importing import import_object

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType

    from modularml.core.io.symbol_spec import SymbolSpec


class SymbolResolutionError(RuntimeError):
    """Raised when a SymbolSpec cannot be resolved into a class."""


@dataclass(frozen=True)
class _RegistryEntry:
    registry: dict[str, type]
    naming_fn: Callable[[type], str]


class SymbolRegistry:
    """
    Central registry for resolving class identity during (de)serialization.

    Responsibilities:
    - Convert runtime classes to SymbolSpec (identify)
    - Convert SymbolSpec to runtime classes (resolve)
    - Enforce SerializationPolicy semantics
    """

    # ================================================
    # Construction
    # ================================================
    def __init__(self):
        self._builtin_classes: dict[str, type] = {}
        self._builtin_registries: dict[str, _RegistryEntry] = {}

    # ================================================
    # Registration APIs
    # ================================================
    def register_builtin_class(self, key: str, cls: type) -> None:
        """
        Register a built-in ModularML class.

        Args:
            key: Stable registry key (e.g. "FeatureSet").
            cls: Class object.

        """
        if key in self._builtin_classes:
            msg = f"Builtin class key '{key}' already registered."
            raise ValueError(msg)
        self._builtin_classes[key] = cls

    def register_builtin_registry(
        self,
        import_path: str,
        registry: dict[str, type],
        *,
        naming_fn: Callable[[type], str],
    ) -> None:
        """
        Register a registry whose contents should be treated as REGISTERED symbols.

        Args:
            import_path:
                Import path to registry object
                (e.g. 'modularml.models.MODEL_REGISTRY')
            registry:
                Mapping of keys -> classes
            naming_fn:
                Function that maps a class to its registry key

        """
        if import_path in self._builtin_registries:
            msg = f"Builtin registry path '{import_path}' already registered."
            raise ValueError(msg)
        self._builtin_registries[import_path] = _RegistryEntry(
            registry=registry,
            naming_fn=naming_fn,
        )

    # ================================================
    # Resolve: SymbolSpec to class or function
    # ================================================
    def resolve_symbol(
        self,
        spec: SymbolSpec,
        *,
        allow_packaged_code: bool = False,
        packaged_code_loader=None,
    ) -> type:
        """
        Resolve a SymbolSpec into a runtime class or function.

        Args:
            spec: SymbolSpec to resolve.
            allow_packaged_code: Whether executing bundled code is allowed.
            packaged_code_loader: Callable to load bundled source if needed.

        Returns:
            Resolved class.

        Raises:
            SymbolResolutionError

        """
        spec.validate()

        policy = spec.policy

        if policy is SerializationPolicy.STATE_ONLY:
            raise SymbolResolutionError(
                "STATE_ONLY artifacts require user-supplied class.",
            )

        if policy is SerializationPolicy.BUILTIN:
            if spec.key in self._builtin_classes:
                return self._builtin_classes[spec.key]
            # Try importing missing class
            try:
                return import_object(spec.key.replace(":", "."))
            except Exception as exc:
                msg = f"Failed to find BUILTIN class '{spec.key}'. {exc}"
                raise SymbolResolutionError(msg) from exc

        if policy is SerializationPolicy.REGISTERED:
            registry = import_object(spec.registry_path)
            try:
                return registry[spec.registry_key]
            except KeyError as exc:
                msg = f"REGISTERED class '{spec.registry_key}' not found in runtime registry."
                raise SymbolResolutionError(msg) from exc

        if policy is SerializationPolicy.PACKAGED:
            if not allow_packaged_code:
                msg = (
                    "The serialized file contains packaged code. Do not load this until "
                    "all code has been inspected. Then use `allow_packaged_code=True`."
                )
                raise RuntimeError(msg)
            if spec.source_ref:
                if not packaged_code_loader:
                    raise SymbolResolutionError("Bundled code loader not provided.")
                module = packaged_code_loader(spec.source_ref)
                return self._get_attr(module, spec.qualname)

            msg = f"PACKAGED class '{spec.import_path}' could not be resolved."
            raise SymbolResolutionError(msg)

        msg = f"Unsupported SerializationPolicy: {policy}"
        raise TypeError(msg)

    # ================================================
    # Helpers & Convenience
    # ================================================
    def obj_is_a_builtin_class(self, obj_or_cls: Any) -> bool:
        """Checks whether an instance is a registered built-in."""
        cls = obj_or_cls if isinstance(obj_or_cls, type) else obj_or_cls.__class__
        return cls in set(self._builtin_classes.values())

    def obj_in_a_builtin_registry(
        self,
        obj_or_cls: Any,
        registry_name: str | None,
    ) -> bool:
        """
        Checks whether an instance is in a built-in registry.

        Args:
            obj_or_cls:
                Instance or class to check.
            registry_name (str, optional):
                Name of built-in registry to search. E.g., "scaler_registry".
                If None, searches all registries.

        """
        # Ensure is class, not isntance
        cls = obj_or_cls if isinstance(obj_or_cls, type) else obj_or_cls.__class__

        # Restrict registries to search (if given registry name)
        ks_to_search = list(self._builtin_registries.keys())
        if registry_name:
            valid_ks = [k for k in ks_to_search if k.endswith(registry_name)]
            ks_to_search = valid_ks

        # Search registries
        for k in ks_to_search:
            entry = self._builtin_registries[k]
            try:
                key = entry.naming_fn(cls)
                if key in entry.registry:
                    return True
            except Exception:  # noqa: BLE001, S112
                continue

        return False

    def key_for(self, obj_or_cls: Any) -> str:
        """Returns the registered key for an object or class."""
        cls = obj_or_cls if isinstance(obj_or_cls, type) else obj_or_cls.__class__
        if not self.obj_is_a_builtin_class(obj_or_cls=obj_or_cls):
            msg = f"Class '{cls.__qualname__}' is not registered. It has no key."
            raise ValueError(msg)
        return self._make_runtime_key(cls=cls)

    def registered_location_for(self, symbol: object) -> tuple[str, str]:
        """Return (registry_import_path, key) if symbol is found in a registered registry."""
        cls = symbol if isinstance(symbol, type) else symbol.__class__

        for registry_path, entry in self._builtin_registries.items():
            key = entry.naming_fn(cls)
            if key in entry.registry and entry.registry[key] is cls:
                return registry_path, key

        msg = f"Symbol {symbol} is not registered in any known registry."
        raise KeyError(msg)

    def _make_runtime_key(self, cls: type) -> str:
        return f"{cls.__module__}:{cls.__qualname__}"

    def _get_attr(self, module: ModuleType, qualname: str | None) -> type:
        if not qualname:
            raise SymbolResolutionError("Missing qualname for resolution.")

        obj = module
        for attr in qualname.split("."):
            if isinstance(obj, dict):
                try:
                    obj = obj[attr]
                except KeyError as exc:
                    msg = f"Name '{attr}' not found in packaged code namespace."
                    raise SymbolResolutionError(msg) from exc
            else:
                try:
                    obj = getattr(obj, attr)
                except AttributeError as exc:
                    msg = f"Attribute '{attr}' not found while resolving '{qualname}'."
                    raise SymbolResolutionError(msg) from exc
        return obj


symbol_registry = SymbolRegistry()
