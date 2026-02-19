"""Symbol specification data structure for referencing classes/functions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import ClassVar

from modularml.core.io.serialization_policy import SerializationPolicy, normalize_policy


@dataclass(frozen=True)
class SymbolSpec:
    """
    Serializable specification describing how to identify and resolve a Python symbol.

    Description:
        A :class:`SymbolSpec` records the information required to locate a class or
        function during serialization and to resolve it during deserialization.
        Behaviour, configuration, and runtime state are handled elsewhere.

    Attributes:
        policy (SerializationPolicy): Serialization policy controlling resolution strategy.
        version (int): SymbolSpec schema version.
        key (str | None): Builtin key for :class:`SerializationPolicy.BUILTIN`.
        registry_path (str | None): Import path for lookup registries.
        registry_key (str | None): Key within the referenced registry.
        module (str | None): Module path for directly importable symbols.
        qualname (str | None): Qualified name within `module`.
        source_ref (str | None): Packaged source reference `code/<file>.py:<qualname>`.

    """

    # Structural metadata
    TAG: ClassVar[str] = "__mml_symbol_spec__"
    VERSION: ClassVar[int] = 1

    # Core fields
    policy: SerializationPolicy
    version: int = VERSION

    # BUILTIN
    #  - key into symbol_registry._builtin_registry
    key: str | None = None

    # REGISTERED
    #  - Fully-qualified import path of registry object and registry key
    #  - e.g. "modularml.core.models.MODEL_REGISTRY"
    registry_path: str | None = None
    registry_key: str | None = None

    # PACKAGED
    #  - Format: "code/<file>.py:<qualname>"
    source_ref: str | None = None

    # Fallback import metadata (optional)
    module: str | None = None
    qualname: str | None = None

    def validate(self) -> None:
        """
        Ensure the current :class:`SerializationPolicy` is coherent with the fields.

        Raises:
            ValueError: If required identity fields are missing or extra data is provided.
            TypeError: If the policy is unsupported.

        """
        object.__setattr__(self, "policy", normalize_policy(self.policy))

        if self.policy is SerializationPolicy.BUILTIN:
            if not self.key:
                raise ValueError("BUILTIN requires key")

        elif self.policy is SerializationPolicy.STATE_ONLY:
            if any([self.key, self.module, self.qualname, self.source_ref]):
                raise ValueError(
                    "STATE_ONLY SymbolSpec must not define class identity fields.",
                )

        elif self.policy is SerializationPolicy.REGISTERED:
            if not (self.registry_path and self.registry_key):
                raise ValueError("REGISTERED requires registry_path and registry_key")

        elif self.policy is SerializationPolicy.PACKAGED:
            if not self.source_ref:
                raise ValueError("PACKAGED policy requires a source_ref.")

        else:
            msg = f"Unsupported policy: {self.policy}"
            raise TypeError(msg)

    def to_dict(self) -> dict:
        """Return a JSON-safe dictionary representation of this :class:`SymbolSpec`."""
        data = asdict(self)
        data[self.TAG] = True
        return data

    @classmethod
    def from_dict(cls, data: dict) -> SymbolSpec:
        """
        Create a :class:`SymbolSpec` from serialized dictionary data.

        Args:
            data (dict): Dictionary previously produced by :meth:`to_dict`.

        Returns:
            SymbolSpec: Reconstructed specification.

        Raises:
            ValueError: If `data` does not represent a serialized :class:`SymbolSpec`.

        """
        data = dict(data)
        if cls.TAG not in data or not data[cls.TAG]:
            raise ValueError("Not a SymbolSpec dict")
        data.pop(cls.TAG)
        return cls(**data)

    @classmethod
    def is_symbol_spec_dict(cls, data: dict) -> bool:
        """
        Return True if `data` originated from :meth:`to_dict`.

        Args:
            data (dict): Dictionary to inspect.

        Returns:
            bool: True when `data` contains the SymbolSpec tag.

        """
        return isinstance(data, dict) and data.get(cls.TAG) is True
