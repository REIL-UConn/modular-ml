"""Serialization kind conventions for ModularML artifacts."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar

MML_FILE_EXTENSION = "mml"


@dataclass(frozen=True)
class SerializationKind:
    """
    Defines the on-disk identity for a serializable object category.

    Attributes:
        name: Human-readable kind name (e.g., "FeatureSet").
        kind: Short suffix identifier (e.g., "fs").

    """

    name: str
    kind: str

    @property
    def file_suffix(self) -> str:
        """
        Build the filename suffix for this kind.

        Returns:
            str: Suffix formatted as `.kind.mml`.

        """
        return f".{self.kind}.{MML_FILE_EXTENSION}"


class KindRegistry:
    """
    Central registry mapping base classes to serialization kinds.

    Resolution is MRO-based: subclasses inherit the first matching base class kind.
    """

    _registry: ClassVar[dict[type, SerializationKind]] = {}
    _rev_registry: ClassVar[dict[SerializationKind, type]] = {}

    def register(
        self,
        cls: type,
        kind: SerializationKind,
        *,
        overwrite: bool = False,
    ) -> None:
        """
        Register a class as the base for a serialization kind.

        Args:
            cls (type): Base class to associate with this serialization kind.
            kind (SerializationKind): Kind definition to register.
            overwrite (bool, optional): Whether to replace existing entries.

        Raises:
            ValueError: If registration conflicts occur.

        """
        if "." in kind.kind:
            msg = f"Serialization kind cannot contain '.': '{kind.kind}'"
            raise ValueError(msg)

        if not overwrite and cls in self._registry:
            msg = f"Class {cls.__name__} already registered as '{self._registry[cls].kind}'."
            raise ValueError(msg)

        if not overwrite and kind in self._rev_registry:
            msg = f"Serialization kind '{kind}' already registered to {self._rev_registry[kind].__name__}."
            raise ValueError(msg)

        self._registry[cls] = kind
        self._rev_registry[kind] = cls

    def register_kind(
        self,
        *,
        name: str,
        kind: str,
        overwrite: bool = False,
    ) -> Callable[[type], type]:
        """
        Decorator for registering a class as a serialization base.

        Args:
            name (str): Human-readable kind name.
            kind (str): Short kind suffix (e.g., `fs`).
            overwrite (bool, optional): Whether to allow overwriting.

        Returns:
            Callable[[type], type]: Decorator that registers the class.

        Raises:
            ValueError: If the kind name is already registered and `overwrite` is False.

        """
        serialization_kind = SerializationKind(name=name, kind=kind)

        def decorator(cls: type) -> type:
            self.register(cls, serialization_kind, overwrite=overwrite)
            return cls

        return decorator

    def get_kind(self, cls: type) -> SerializationKind:
        """
        Resolve the serialization kind for a class using MRO lookup.

        Args:
            cls (type): Class to resolve.

        Returns:
            SerializationKind: Kind associated with the nearest registered base class.

        Raises:
            KeyError: If no serialization kind is registered for the hierarchy.

        """
        for base in cls.__mro__:
            if base in self._registry:
                return self._registry[base]
        msg = f"No serialization kind registered for class hierarchy of {cls.__name__}"
        raise KeyError(msg)

    def clear(self):
        """Clear all registered items."""
        self._registry.clear()
        self._rev_registry.clear()


kind_registry = KindRegistry()
