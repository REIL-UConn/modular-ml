"""Default loader for executing packaged source code within artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class PackagedCodeError(RuntimeError):
    """Raised when bundled code cannot be safely loaded."""


def default_packaged_code_loader(
    *,
    artifact_path: Path,
    source_ref: str,
    allow_packaged: bool,
) -> Any:
    """
    Load bundled source referenced by `source_ref` from an artifact.

    Args:
        artifact_path (Path): Root directory of the artifact.
        source_ref (str): Reference of the form `code/<file>.py:<qualname>`.
        allow_packaged (bool): Whether executing bundled code is allowed.

    Returns:
        Any: Execution namespace containing the loaded symbols.

    Raises:
        PackagedCodeError: If execution is blocked or fails.
        FileNotFoundError: If the referenced source file cannot be found.

    """
    if not allow_packaged:
        msg = (
            "The artifact contains packaged executable code, which is blocked by default. "
            "Use `inspect_packaged_code(path=...)` to review the code. "
            "If approved, reload with `allow_packaged=True`."
        )
        raise PackagedCodeError(msg)

    # ================================================
    # Get path to source
    # ================================================
    try:
        rel_path, _ = source_ref.split(":", 1)
    except ValueError as exc:
        msg = f"Invalid source_ref format: {source_ref!r}"
        raise PackagedCodeError(msg) from exc

    src_path = Path(artifact_path) / rel_path
    if not src_path.exists():
        msg = f"Packaged source not found: {src_path}"
        raise FileNotFoundError(msg)

    source = src_path.read_text(encoding="utf-8")

    # ================================================
    # Isolated execution namespace
    # ================================================
    ns: dict[str, Any] = {
        "__builtins__": __builtins__,
        "__file__": str(src_path),
        "__name__": "__mml_packaged__",
    }

    try:
        exec(compile(source, str(src_path), "exec"), ns, ns)  # noqa: S102
    except Exception as exc:
        msg = f"Failed to execute bundled code: {src_path}"
        raise PackagedCodeError(msg) from exc

    for obj in ns.values():
        if isinstance(obj, type):
            obj.__mml_source_ref__ = source_ref
            obj.__mml_source_text__ = source

    return ns
