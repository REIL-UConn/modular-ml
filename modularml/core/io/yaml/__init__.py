from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


# ================================================
# Internal helpers
# ================================================
def _detect_kind(d: dict[str, Any]) -> str:
    """Detect the object kind from the top-level key of a YAML dict."""
    keys = set(d.keys())
    if "model_graph" in keys:
        return "model_graph"
    if "train_phase" in keys:
        return "train_phase"
    if "eval_phase" in keys:
        return "eval_phase"
    if "fit_phase" in keys:
        return "fit_phase"
    if "experiment" in keys:
        return "experiment"
    if "sampler" in keys:
        return "sampler"
    if "splitter" in keys:
        return "splitter"
    msg = (
        f"Cannot detect object kind from YAML top-level keys: {sorted(keys)}. "
        "Expected one of: model_graph, train_phase, eval_phase, experiment, sampler, splitter."
    )
    raise ValueError(msg)


# ================================================
# Public API
# ================================================
def obj_to_yaml_dict(obj: Any) -> dict[str, Any]:
    """
    Convert a ModularML object to a YAML-friendly dictionary.

    Supported types: :class:`ModelGraph`, :class:`ExperimentPhase` subclasses
    (:class:`TrainPhase`, :class:`EvalPhase`), :class:`BaseSampler` subclasses,
    :class:`BaseSplitter` subclasses, :class:`Experiment`.

    Args:
        obj (Any): Object to convert.

    Returns:
        dict[str, Any]: YAML-friendly dict with a single descriptive top-level key.

    Raises:
        TypeError: If ``obj`` is not a supported type.

    """
    from modularml.core.experiment.experiment import Experiment
    from modularml.core.experiment.phases.phase import ExperimentPhase
    from modularml.core.io.yaml.translators.experiment import experiment_to_yaml_dict
    from modularml.core.io.yaml.translators.model_graph import model_graph_to_yaml_dict
    from modularml.core.io.yaml.translators.phase import phase_to_yaml_dict
    from modularml.core.io.yaml.translators.sampler import sampler_to_yaml_dict
    from modularml.core.io.yaml.translators.splitter import splitter_to_yaml_dict
    from modularml.core.sampling.base_sampler import BaseSampler
    from modularml.core.splitting.base_splitter import BaseSplitter
    from modularml.core.topology.model_graph import ModelGraph

    if isinstance(obj, ModelGraph):
        return model_graph_to_yaml_dict(obj)
    if isinstance(obj, ExperimentPhase):
        return phase_to_yaml_dict(obj)
    if isinstance(obj, BaseSampler):
        d = sampler_to_yaml_dict(obj)
        return {"sampler": d}
    if isinstance(obj, BaseSplitter):
        d = splitter_to_yaml_dict(obj)
        return {"splitter": d}
    if isinstance(obj, Experiment):
        return experiment_to_yaml_dict(obj)

    msg = f"YAML export is not supported for type: {type(obj).__name__}"
    raise TypeError(msg)


def yaml_dict_to_obj(
    d: dict[str, Any],
    kind: str | None = None,
    *,
    overwrite: bool = False,
) -> Any:
    """
    Reconstruct a ModularML object from a YAML-friendly dictionary.

    Args:
        d (dict[str, Any]): Dict produced by :func:`obj_to_yaml_dict`.
        kind (str | None): Optional override for auto-detection. Supported
            values: ``"model_graph"``, ``"train_phase"``, ``"eval_phase"``,
            ``"experiment"``, ``"sampler"``, ``"splitter"``.
        overwrite (bool, optional): Passed to node-registering translators.
            See :func:`from_yaml` for semantics. Defaults to False.

    Returns:
        Any: Reconstructed object.

    """
    from modularml.core.io.yaml.translators.experiment import experiment_from_yaml_dict
    from modularml.core.io.yaml.translators.model_graph import (
        model_graph_from_yaml_dict,
    )
    from modularml.core.io.yaml.translators.phase import phase_from_yaml_dict
    from modularml.core.io.yaml.translators.sampler import sampler_from_yaml_dict
    from modularml.core.io.yaml.translators.splitter import splitter_from_yaml_dict

    kind = kind or _detect_kind(d)

    if kind == "model_graph":
        return model_graph_from_yaml_dict(d, overwrite=overwrite)
    if kind in ("train_phase", "eval_phase", "fit_phase"):
        return phase_from_yaml_dict(d)
    if kind == "experiment":
        return experiment_from_yaml_dict(d, overwrite=overwrite)
    if kind == "sampler":
        inner = d.get("sampler", d)
        return sampler_from_yaml_dict(inner)
    if kind == "splitter":
        inner = d.get("splitter", d)
        return splitter_from_yaml_dict(inner)

    msg = f"Unknown YAML kind: {kind!r}"
    raise ValueError(msg)


def to_yaml(obj: Any, path: str | Path, *, overwrite: bool = False) -> Path:
    """
    Export a ModularML object to a YAML file.

    Args:
        obj (Any): Object to export. See :func:`obj_to_yaml_dict` for supported types.
        path (str | Path): Destination file path. Parent directories are created
            automatically. The ``.yaml`` extension is added if not present.
        overwrite (bool, optional): Whether to overwrite an existing file at
            ``path``. Defaults to False.

    Returns:
        Path: The resolved path the file was written to.

    Raises:
        FileExistsError: If ``path`` already exists and ``overwrite`` is False.

    """
    path = Path(path)
    if path.suffix not in (".yaml", ".yml"):
        path = path.with_suffix(".yaml")
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists() and not overwrite:
        msg = f"File already exists at '{path}'. Pass overwrite=True to replace it."
        raise FileExistsError(msg)

    d = obj_to_yaml_dict(obj)
    with path.open("w", encoding="utf-8") as f:
        yaml.dump(d, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    return path


def from_yaml(
    path: str | Path,
    kind: str | None = None,
    *,
    overwrite: bool = False,
) -> Any:
    """
    Load a ModularML object from a YAML file.

    Args:
        path (str | Path): Path to the YAML file.
        kind (str | None): Optional hint for the object type. When ``None``
            the kind is auto-detected from the top-level key. Supported
            values: ``"model_graph"``, ``"train_phase"``, ``"eval_phase"``,
            ``"experiment"``, ``"sampler"``, ``"splitter"``.
        overwrite (bool, optional): Whether to overwrite conflicting node
            registrations already present in the active
            :class:`ExperimentContext`. When ``False`` (default) a
            :exc:`ValueError` is raised if any reconstructed node label
            collides with an existing registration. When ``True`` the
            existing registration is replaced. Defaults to False.

    Returns:
        Any: Reconstructed object.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If a node label conflict is detected and
            ``overwrite`` is False.

    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8") as f:
        d = yaml.safe_load(f)

    return yaml_dict_to_obj(d, kind=kind, overwrite=overwrite)
