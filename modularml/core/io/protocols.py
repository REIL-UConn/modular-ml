"""Protocols describing configurable and stateful objects."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Configurable(Protocol):
    """Protocol for objects that have configurable instantiation."""

    def get_config(self) -> dict[str, Any]:
        """
        Return the configuration needed by :meth:`from_config`.

        Returns:
            dict[str, Any]: JSON-serializable configuration mapping.

        """
        ...

    @classmethod
    def from_config(cls, config: dict):
        """
        Construct an object from the configuration produced by :meth:`get_config`.

        Args:
            config (dict[str, Any]): Serialized configuration mapping.

        Returns:
            Configurable: Instance of the implementing class.

        """
        ...


@runtime_checkable
class Stateful(Protocol):
    """Protocol for objects that can export and restore runtime state."""

    def get_state(self) -> Any:
        """
        Return runtime state used to fully reproduce the object.

        Returns:
            Any: State payload that :meth:`set_state` can consume.

        """
        ...

    def set_state(self, state: Any) -> None:
        """
        Restore runtime state previously captured by :meth:`get_state`.

        Args:
            state (Any): State payload to restore.

        """
        ...
