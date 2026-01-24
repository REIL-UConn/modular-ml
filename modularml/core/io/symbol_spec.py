from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import ClassVar

from modularml.core.io.serialization_policy import SerializationPolicy, normalize_policy


@dataclass(frozen=True)
class SymbolSpec:
    """
    Serializable specification describing how to identify and resolve a Python symbol (class or function).

    It defines how a class/function is identified at save time and how it
    should be resolved at load time, such that:
    - A runtime class/fnc can be converted into a SymbolSpec
    - A SymbolSpec can be resolved back into a runtime class/fnc

    It *does not* describe behaviour, configuration, or state.
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
        """JSON-safe dict representation of this SymbolSpec."""
        data = asdict(self)
        data[self.TAG] = True
        return data

    @classmethod
    def from_dict(cls, data: dict) -> SymbolSpec:
        data = dict(data)
        if cls.TAG not in data or not data[cls.TAG]:
            raise ValueError("Not a SymbolSpec dict")
        data.pop(cls.TAG)
        return cls(**data)

    @classmethod
    def is_symbol_spec_dict(cls, data: dict) -> bool:
        return isinstance(data, dict) and data.get(cls.TAG) is True
