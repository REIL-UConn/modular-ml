"""Translator between Experiment instances and YAML-friendly dicts."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from modularml.core.io.yaml.translators.model_graph import (
    model_graph_from_yaml_dict,
    model_graph_to_yaml_dict,
)
from modularml.core.io.yaml.translators.phase import (
    phase_from_yaml_dict,
    phase_to_yaml_dict,
)

if TYPE_CHECKING:
    from modularml.core.experiment.experiment import Experiment
    from modularml.core.experiment.phases.phase import ExperimentPhase
    from modularml.core.experiment.phases.phase_group import PhaseGroup


# ================================================
# Helpers
# ================================================
def _item_to_dict(item: ExperimentPhase | PhaseGroup) -> dict[str, Any]:
    """Convert an ExperimentPhase or PhaseGroup to a YAML-friendly dict."""
    from modularml.core.experiment.phases.phase import ExperimentPhase
    from modularml.core.experiment.phases.phase_group import PhaseGroup

    if isinstance(item, PhaseGroup):
        return {
            "phase_type": "phase_group",
            "label": item.label,
            "items": [_item_to_dict(child) for child in item.all],
        }

    if isinstance(item, ExperimentPhase):
        phase_dict = phase_to_yaml_dict(item)
        phase_type, phase_inner = next(iter(phase_dict.items()))
        result = dict(phase_inner)
        result["phase_type"] = phase_type
        return result

    msg = f"Unsupported execution plan item type: {type(item)!r}"
    raise TypeError(msg)


def _item_from_dict(cfg: dict[str, Any]) -> ExperimentPhase | PhaseGroup:
    """Reconstruct an ExperimentPhase or PhaseGroup from a YAML dict."""
    from modularml.core.experiment.phases.phase_group import PhaseGroup

    cfg = dict(cfg)
    phase_type = cfg.pop("phase_type", None)

    if phase_type == "phase_group":
        label = cfg.get("label", "group")
        group = PhaseGroup(label=label)
        for child_cfg in cfg.get("items", []):
            group.add_item(_item_from_dict(child_cfg))
        return group

    if phase_type in ("train_phase", "TrainPhase"):
        return phase_from_yaml_dict({"train_phase": cfg})

    if phase_type in ("eval_phase", "EvalPhase"):
        return phase_from_yaml_dict({"eval_phase": cfg})

    if phase_type in ("fit_phase", "FitPhase"):
        return phase_from_yaml_dict({"fit_phase": cfg})

    msg = f"Unsupported phase_type in experiment YAML: {phase_type!r}"
    raise ValueError(msg)


# ================================================
# Export
# ================================================
def experiment_to_yaml_dict(experiment: Experiment) -> dict[str, Any]:
    """
    Export an :class:`Experiment` to a unified YAML-friendly dictionary.

    The resulting dict has a single ``experiment`` top-level key containing
    the experiment label, model graph, and all registered phases (including
    nested :class:`PhaseGroup` items).

    Args:
        experiment (Experiment): Experiment instance to export.

    Returns:
        dict[str, Any]: YAML-friendly dict.

    """
    exp_dict: dict[str, Any] = {"label": experiment.label}

    # Export model graph
    if experiment.model_graph is not None:
        mg_dict = model_graph_to_yaml_dict(experiment.model_graph)
        exp_dict["model_graph"] = mg_dict["model_graph"]

    # Export phases / phase groups from execution plan
    phases_yaml = [_item_to_dict(item) for item in experiment.execution_plan.all]
    if phases_yaml:
        exp_dict["phases"] = phases_yaml

    return {"experiment": exp_dict}


# ================================================
# Import
# ================================================
def experiment_from_yaml_dict(
    d: dict[str, Any],
    *,
    overwrite: bool = False,
) -> Experiment:
    """
    Reconstruct an :class:`Experiment` from a YAML-friendly dictionary.

    The model graph (if present) is built and registered first. All phase
    input_sources must reference nodes or FeatureSets already in the active
    :class:`ExperimentContext`. :class:`PhaseGroup` items are reconstructed
    recursively.

    Args:
        d (dict[str, Any]): Dict with an ``experiment`` top-level key, as
            produced by :func:`experiment_to_yaml_dict`.
        overwrite (bool, optional): Passed to :func:`model_graph_from_yaml_dict`.
            See :func:`~modularml.core.io.yaml.from_yaml` for semantics.
            Defaults to False.

    Returns:
        Experiment: Reconstructed experiment.

    """
    from modularml.core.experiment.experiment import Experiment
    from modularml.core.experiment.experiment_context import ExperimentContext

    if "experiment" in d:
        d = d["experiment"]

    label = d.get("label", "experiment")

    # Reuse the active context so already-registered nodes/FeatureSets are preserved
    # Passing ctx= prevents Experiment.__init__ from creating a new context and
    # overwriting the active one via ExperimentContext._set_active()
    try:
        active_ctx = ExperimentContext.get_active()
    except RuntimeError:
        active_ctx = None
    experiment = Experiment(label=label, ctx=active_ctx)

    # Build model graph first (registers nodes into active context)
    if "model_graph" in d:
        mg = model_graph_from_yaml_dict(
            {"model_graph": d["model_graph"]},
            overwrite=overwrite,
        )
        experiment.ctx.register_model_graph(mg)

    # Build phases / phase groups
    for item_cfg in d.get("phases", []):
        experiment.execution_plan.add_item(_item_from_dict(item_cfg))

    return experiment
