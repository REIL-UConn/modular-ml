import inspect
import json
import pathlib
import zipfile
from typing import Any

from modularml.core.io.artifacts import Artifact
from modularml.core.io.conventions import MML_FILE_EXTENSION
from modularml.utils.logging.warnings import warn


def inspect_packaged_code(path: pathlib.Path) -> dict[str, str]:
    """
    Inspect a ModularML .mml artifact and extract all packaged source code.

    Description:
        Recursively scans the top-level artifact for any nested `.mml` files,
        then inspects each artifact independently to extract packaged Python
        source files without executing them.

        This function is safe to call on untrusted artifacts.

    Args:
        path (Path):
            Path to a `.mml` file.

    Returns:
        Dict[str, str]:
            Mapping from source_ref (e.g. "code/my_scaler.py:MyScaler")
            to the corresponding source code text.

    Raises:
        FileNotFoundError:
            If the artifact file does not exist.
        ValueError:
            If the file is not a valid ModularML artifact.

    """
    path = pathlib.Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    all_sources: dict[str, str] = {}
    with zipfile.ZipFile(path, "r") as root_zip:
        # 1. Inspect root artifact
        all_sources.update(
            _inspect_mml_zip(
                mml_zip=root_zip,
                root_zip=root_zip,
                label="<root>",
            ),
        )

        # 2. Find and inspect any nested artifacts (.mml) files
        for name in root_zip.namelist():
            if not name.endswith(MML_FILE_EXTENSION):
                continue

            with root_zip.open(name) as nested_bytes, zipfile.ZipFile(nested_bytes) as nested_zip:
                all_sources.update(
                    _inspect_mml_zip(
                        mml_zip=nested_zip,
                        root_zip=root_zip,
                        label=name,
                    ),
                )

    return all_sources


def _inspect_mml_zip(
    *,
    mml_zip: zipfile.ZipFile,
    root_zip: zipfile.ZipFile,
    label: str,
) -> dict[str, str]:
    """Inspect a single opened `.mml` zip archive."""
    sources: dict[str, str] = {}

    # Validate artifact.json
    try:
        with mml_zip.open("artifact.json") as f:
            artifact = Artifact.from_json(json.load(f))
    except KeyError as exc:
        msg = f"Invalid ModularML artifact ({label}): missing artifact.json"
        raise ValueError(msg) from exc

    # Extract config if present
    config_rel_path = artifact.files.get("config")
    if not config_rel_path or not config_rel_path.endswith(".json"):
        return sources

    try:
        with mml_zip.open(config_rel_path) as f:
            config = json.load(f)
    except Exception as exc:
        msg = f"Failed to read config in {label}"
        raise RuntimeError(msg) from exc

    # Recursively scan config for packaged symbols
    def collect(obj: object) -> None:
        if isinstance(obj, dict):
            if obj.get("policy") == "packaged" and "source_ref" in obj:
                source_ref = obj["source_ref"]
                file_path, _ = source_ref.split(":", 1)

                try:
                    # Code paths are relative to root_zip, not artifact zip
                    with root_zip.open(file_path) as src:
                        sources[source_ref] = src.read().decode("utf-8")
                except KeyError as exc:
                    msg = f"Packaged source '{file_path}' not found in {label}"
                    raise FileNotFoundError(msg) from exc

            for v in obj.values():
                collect(v)

        elif isinstance(obj, list):
            for v in obj:
                collect(v)

    collect(config)
    return sources


def infer_kwargs_from_init(
    obj: Any,
    *,
    strict: bool = False,
) -> dict[str, Any]:
    """
    Infer constructor keyword arguments from an existing object instance.

    Description:
        Inspects the `__init__` signature of the object's class and attempts to
        reconstruct a dictionary of keyword arguments by reading attributes on
        the instance with matching names.

        For each parameter in `obj.__class__.__init__`, excluding `self`, this
        function checks whether:
          - the parameter has a default value, OR
          - the instance has an attribute with the same name

        Parameters that are required (no default) but cannot be recovered from
        instance attributes are treated as reconstruction failures.

    Behavior:
        - If all required (non-default) parameters can be inferred, the inferred
          keyword dictionary is returned.
        - If one or more required parameters cannot be inferred:
            * a warning is emitted by default
            * a ValueError is raised if `strict=True`

    Notes:
        - This function reflects the *current state* of the object, not necessarily
          the original arguments passed at construction time.
        - Only parameters whose names exactly match instance attribute names can
          be inferred.
        - Positional-only parameters, `*args`, and `**kwargs` cannot be reliably
          reconstructed and are ignored for strictness checks.
        - Default-valued parameters are allowed to be missing.

    Limitations:
        - Cannot recover arguments that are transformed, renamed, or discarded
          inside `__init__`.
        - Cannot distinguish between default values and explicitly provided values.
        - Properties or dynamically computed attributes may give misleading results.

    Args:
        obj (Any):
            The object instance from which to infer constructor keyword arguments.
        strict (bool, optional):
            If True, raise a ValueError when required constructor arguments cannot
            be inferred. If False, emit a warning instead. Defaults to False.

    Returns:
        dict[str, Any]:
            A dictionary mapping inferred constructor parameter names to their
            current attribute values on the object.

    Raises:
        ValueError:
            If `strict=True` and one or more required constructor parameters
            cannot be inferred.

    """
    kwargs: dict[str, Any] = {}
    missing_required: list[str] = []

    sig = inspect.signature(obj.__class__.__init__)

    for name, param in sig.parameters.items():
        if name == "self":
            continue

        # Skip *args / **kwargs — cannot reconstruct meaningfully
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue

        if hasattr(obj, name):
            kwargs[name] = getattr(obj, name)
        elif param.default is inspect.Parameter.empty:
            missing_required.append(name)

    if missing_required:
        msg = f"Cannot fully infer constructor arguments for {obj.__class__.__qualname__}. "
        hint = f"Provide required parameters: {missing_required}"
        if strict:
            raise ValueError(msg)
        warn(msg, RuntimeWarning, stacklevel=2, hints=hint)

    return kwargs
