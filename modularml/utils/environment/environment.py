"""Environment detection helpers and dynamic import utilities."""

from __future__ import annotations

import importlib
from typing import Any


def _detect_notebook() -> bool:
    """
    Return True if running inside an IPython notebook-like shell.

    Returns:
        bool: True when executing in a notebook environment.

    """
    try:
        from IPython import get_ipython

        ip = get_ipython()
        if ip is None:
            return False

        # Jupyter notebooks and qtconsole
        return ip.__class__.__name__ in {  # noqa: TRY300
            "ZMQInteractiveShell",
        }

    except Exception:  # noqa: BLE001
        return False


def import_from_path(path: str) -> Any:
    """
    Dynamically import a class, function, or variable from a full module path.

    Example:
        ```python
        cls = import_from_path("torch.nn.Linear")
        ```

    Args:
        path (str):
            Fully-qualified import path of the form
            "<module>.<name>", e.g. "modularml.models.SequentialMLP".

    Returns:
        Any: The imported class / function / attribute.

    Raises:
        ValueError: If the path is malformed or target object cannot be resolved.
        ImportError: If the module cannot be imported.
        AttributeError: If the object is not found in the module.

    """
    if not isinstance(path, str) or "." not in path:
        msg = f"import_from_path() expected a string 'module.submodule.Class', but received: {path!r}"
        raise ValueError(msg)

    module_path, attr_name = path.rsplit(".", 1)

    # Import module
    try:
        module = importlib.import_module(module_path)
    except Exception as e:
        msg = f"Failed to import module '{module_path}' when resolving '{path}'."
        raise ImportError(msg) from e

    # Get attribute (class, function, variable, etc.)
    try:
        attr = getattr(module, attr_name)
    except Exception as e:
        msg = f"Module '{module_path}' does not define attribute '{attr_name}'. Full path: '{path}'."
        raise AttributeError(msg) from e

    return attr


IN_NOTEBOOK: bool = _detect_notebook()
"""Whether current environment is a Jupyter Notebook"""
