"""Serialization policy semantics for saving ModularML objects."""

from enum import Enum


class SerializationPolicy(str, Enum):
    """
    Defines how an object's class and state are serialized and restored.

    Attributes:
        BUILTIN (str):
            Object uses a built in class from ModularML.
            Object is fully serializable.

        REGISTERED (str):
            Object is user-created but registered.
            This is only serializable (i.e. reload-able) within the
            same runtime environment.

        PACKAGED (str):
            Fully serializable, user-created class. Object class code is
            copied to a file and saved with the saved weights.

        STATE_ONLY (str):
            Only the object weights (internal states) are saved.
            User must supply the same object class on reload.

    """

    BUILTIN = "builtin"
    REGISTERED = "registered"
    PACKAGED = "packaged"
    STATE_ONLY = "state_only"

    def __repr__(self) -> str:
        """Return the policy value for debugging."""
        return self.value

    # ================================================
    # Semantic properties
    # ================================================
    @property
    def requires_class_identity(self) -> bool:
        """
        Whether this policy requires an accompanying class definition.

        Returns:
            bool: True if class identity must be provided.

        """
        return self is not SerializationPolicy.STATE_ONLY

    @property
    def is_portable(self) -> bool:
        """
        Whether this policy can be ported to external users/devices.

        Returns:
            bool: True if the saved artifact is portable.

        """
        return self in {
            SerializationPolicy.BUILTIN,
            SerializationPolicy.PACKAGED,
        }

    @property
    def uses_registry(self) -> bool:
        """
        Whether this policy utilizes registries to deserialize objects.

        Returns:
            bool: True if the registry is required for deserialization.

        """
        return self in {
            SerializationPolicy.BUILTIN,
            SerializationPolicy.REGISTERED,
        }

    @property
    def bundles_code(self) -> bool:
        """
        Whether this policy exports the class/artifact as executable code.

        Returns:
            bool: True if class code is packaged with the artifact.

        """
        return self is SerializationPolicy.PACKAGED


def normalize_policy(value: str | SerializationPolicy):
    """
    Normalize a string or enum to :class:`SerializationPolicy`.

    Args:
        value (str | SerializationPolicy): Value to normalize.

    Returns:
        SerializationPolicy: Canonical policy enum.

    Raises:
        ValueError: If `value` is neither a string nor a policy enum.

    """
    if isinstance(value, SerializationPolicy):
        return value

    if isinstance(value, str):
        value = value.lower().strip()
        return SerializationPolicy(value=value)

    msg = f"Unsupported SerializationPolicy value: {value}"
    raise ValueError(msg)
