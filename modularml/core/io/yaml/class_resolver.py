"""Utilities for converting between Python classes and dotted-path strings."""

from __future__ import annotations

import importlib
from typing import Any


def resolve_class(dotted_path: str) -> type:
    """
    Import and return the class identified by a dotted module path.

    Args:
        dotted_path (str): Fully qualified class path, e.g. ``"modularml.models.torch.MLP"``.

    Returns:
        type: The resolved class.

    Raises:
        ImportError: If the module cannot be imported, or if the module path
            is ``"__main__"`` (not importable by name).
        AttributeError: If the class name does not exist on the module.

    """
    module_path, cls_name = dotted_path.rsplit(".", 1)
    if module_path == "__main__":
        msg = (
            f"Cannot resolve '{dotted_path}': module '__main__' is not importable. "
            "Define the callable in a proper module so it can be round-tripped via YAML."
        )
        raise ImportError(msg)
    module = importlib.import_module(module_path)
    return getattr(module, cls_name)


def class_to_path(cls: type) -> str:
    """
    Return the fully qualified dotted path for a class.

    Args:
        cls (type): A Python class object.

    Returns:
        str: Dotted path string, e.g. ``"modularml.models.torch.MLP"``.

    """
    return f"{cls.__module__}.{cls.__qualname__}"


def obj_class_to_path(obj: Any) -> str:
    """
    Return the fully qualified dotted path for the class of an instance.

    Args:
        obj (Any): An instance of any class.

    Returns:
        str: Dotted path string of the instance's class.

    """
    return class_to_path(type(obj))
