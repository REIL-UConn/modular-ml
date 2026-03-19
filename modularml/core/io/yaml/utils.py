"""Shared YAML serialization utilities."""

from __future__ import annotations

import dataclasses
import inspect
from typing import Any

from modularml.core.io.yaml.class_resolver import class_to_path


def callable_to_path(fn: Any) -> str:
    """
    Return a dotted import path for a class or callable.

    Args:
        fn (Any): A class, function, or other callable.

    Returns:
        str: Fully qualified dotted import path.

    Raises:
        ImportError: If ``fn`` is defined in ``__main__`` (not importable by name).

    """
    if inspect.isclass(fn):
        path = class_to_path(fn)
    else:
        path = f"{fn.__module__}.{fn.__qualname__}"

    module = path.rsplit(".", 1)[0]
    if module == "__main__":
        msg = (
            f"Cannot serialize callable '{path}': "
            "module '__main__' is not importable. "
            "Define the callable in a proper module for YAML round-trips."
        )
        raise ImportError(msg)

    return path


def yaml_safe_cfg(value: Any) -> Any:
    """
    Recursively convert a config value to a YAML-safe primitive.

    This function mirrors the logic of ``BaseHandler._json_safe_cfg`` but operates
    without a ``SaveContext`` or temporary directory.  It is used as a fallback
    for object types that do not have a dedicated YAML translator.

    Rules applied in order:

    * ``None`` / ``bool`` / ``int`` / ``float`` / ``str`` — returned as-is.
    * :class:`~modularml.core.io.protocols.Configurable` instances (non-type) —
      ``yaml_safe_cfg(instance.get_config())``.
    * ``dict`` — keys coerced to ``str``, values recursively processed.
    * ``tuple`` — converted to a ``list`` with each element recursively processed.
    * ``list`` — each element recursively processed.
    * dataclass instances — converted via ``dataclasses.asdict`` then recursively
      processed.
    * classes / callables — converted to a dotted import path string via
      :func:`callable_to_path`.

    Args:
        value (Any): Any Python value that may appear in a ``get_config()`` dict.

    Returns:
        Any: A YAML-safe representation of ``value``.

    """
    from modularml.core.io.protocols import Configurable

    # Primitives — returned as-is
    if value is None or isinstance(value, (bool, int, float, str)):
        return value

    # Configurable instances (not classes) — recurse into get_config()
    if not isinstance(value, type) and isinstance(value, Configurable):
        return yaml_safe_cfg(value.get_config())

    # Dicts
    if isinstance(value, dict):
        return {str(k): yaml_safe_cfg(v) for k, v in value.items()}

    # Tuples → list
    if isinstance(value, tuple):
        return [yaml_safe_cfg(x) for x in value]

    # Lists
    if isinstance(value, list):
        return [yaml_safe_cfg(x) for x in value]

    # Dataclass instances
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        if hasattr(value, "to_dict"):
            return yaml_safe_cfg(value.to_dict())
        return yaml_safe_cfg(dataclasses.asdict(value))

    # Classes and callables → dotted path string
    if isinstance(value, type) or callable(value):
        return callable_to_path(value)

    return value
