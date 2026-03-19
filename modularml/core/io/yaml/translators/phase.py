"""Translator between ExperimentPhase instances and YAML-friendly dicts."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from modularml.core.io.yaml.class_resolver import resolve_class
from modularml.core.io.yaml.translators.references import (
    ref_from_yaml_dict,
    ref_to_yaml_dict,
)
from modularml.core.io.yaml.translators.sampler import (
    sampler_from_yaml_dict,
    sampler_to_yaml_dict,
)
from modularml.core.io.yaml.utils import callable_to_path

if TYPE_CHECKING:
    from modularml.core.experiment.checkpointing import Checkpointing
    from modularml.core.experiment.phases.eval_phase import EvalPhase
    from modularml.core.experiment.phases.fit_phase import FitPhase
    from modularml.core.experiment.phases.phase import ExperimentPhase, InputBinding
    from modularml.core.experiment.phases.train_phase import TrainPhase
    from modularml.core.training.applied_loss import AppliedLoss
    from modularml.core.training.loss import Loss


def _get_node_label(node_id: str) -> str:
    from modularml.core.experiment.experiment_context import ExperimentContext

    ctx = ExperimentContext.get_active()
    return ctx.get_node(node_id=node_id).label


# ================================================
# Export
# ================================================
def phase_to_yaml_dict(phase: ExperimentPhase) -> dict[str, Any]:
    """
    Export an :class:`ExperimentPhase` to a YAML-friendly dictionary.

    The returned dict has a single top-level key identifying the phase type
    (``train_phase``, ``eval_phase``, or ``fit_phase``).

    Args:
        phase (ExperimentPhase): Phase instance to export.

    Returns:
        dict[str, Any]: YAML-friendly dict.

    Raises:
        TypeError: If the phase type is not supported.

    """
    from modularml.core.experiment.phases.eval_phase import EvalPhase
    from modularml.core.experiment.phases.fit_phase import FitPhase
    from modularml.core.experiment.phases.train_phase import TrainPhase

    if isinstance(phase, TrainPhase):
        phase_dict = _train_phase_to_dict(phase)
        return {"train_phase": phase_dict}

    if isinstance(phase, EvalPhase):
        phase_dict = _eval_phase_to_dict(phase)
        return {"eval_phase": phase_dict}

    if isinstance(phase, FitPhase):
        phase_dict = _fit_phase_to_dict(phase)
        return {"fit_phase": phase_dict}

    msg = f"Unsupported phase type for YAML export: {type(phase).__name__}"
    raise TypeError(msg)


def _train_phase_to_dict(phase: TrainPhase) -> dict[str, Any]:
    """Convert a TrainPhase to a YAML dict."""
    d: dict[str, Any] = {
        "label": phase.label,
        "n_epochs": phase.n_epochs,
        "input_sources": [_input_binding_to_dict(b) for b in phase.input_sources],
        "losses": [_applied_loss_to_dict(ls) for ls in phase.losses],
        "active_nodes": [_get_node_label(n_id) for n_id in phase.active_nodes]
        if phase.active_nodes
        else None,
        "batch_schedule": phase.batch_schedule.value,
        "callbacks": [_callback_to_dict(cb) for cb in phase.callbacks]
        if phase.callbacks
        else None,
        "checkpointing": _checkpointing_to_dict(phase.checkpointing)
        if phase.checkpointing is not None
        else None,
        "result_recording": phase.result_recording.value,
        "accelerator": phase.accelerator.device
        if phase.accelerator is not None
        else None,
    }
    return d


def _eval_phase_to_dict(phase: EvalPhase) -> dict[str, Any]:
    """Convert an EvalPhase to a YAML dict."""
    d: dict[str, Any] = {
        "label": phase.label,
        "input_sources": [_input_binding_to_dict(b) for b in phase.input_sources],
        "losses": [_applied_loss_to_dict(ls) for ls in phase.losses]
        if phase.losses
        else None,
        "active_nodes": [_get_node_label(n_id) for n_id in phase.active_nodes]
        if phase.active_nodes
        else None,
        "batch_size": getattr(phase, "batch_size", None),
        "callbacks": [_callback_to_dict(cb) for cb in phase.callbacks]
        if phase.callbacks
        else None,
        "accelerator": phase.accelerator.device
        if phase.accelerator is not None
        else None,
    }
    return d


def _fit_phase_to_dict(phase: FitPhase) -> dict[str, Any]:
    """Convert a FitPhase to a YAML dict."""
    d: dict[str, Any] = {
        "label": phase.label,
        "input_sources": [_input_binding_to_dict(b) for b in phase.input_sources],
        "losses": [_applied_loss_to_dict(ls) for ls in phase.losses]
        if phase.losses
        else None,
        "active_nodes": [_get_node_label(n_id) for n_id in phase.active_nodes]
        if phase.active_nodes
        else None,
        "freeze_after_fit": phase.freeze_after_fit,
        "callbacks": [_callback_to_dict(cb) for cb in phase.callbacks]
        if phase.callbacks
        else None,
        "accelerator": phase.accelerator.device
        if phase.accelerator is not None
        else None,
    }
    return d


def _input_binding_to_dict(binding: InputBinding) -> dict[str, Any]:
    """Convert an InputBinding to a YAML dict."""
    from modularml.core.data.schema_constants import STREAM_DEFAULT
    from modularml.core.experiment.experiment_context import ExperimentContext

    ctx = ExperimentContext.get_active()

    # Resolve node label from node_id
    try:
        node = ctx.get_node(node_id=binding.node_id)
        node_label = node.label
    except Exception:  # noqa: BLE001
        node_label = binding.node_id  # fallback to ID if label not found

    d: dict[str, Any] = {
        "node": node_label,
        "featureset": ref_to_yaml_dict(binding.upstream_ref),
        "split": binding.split,
    }
    if binding.sampler is not None:
        d["sampler"] = sampler_to_yaml_dict(binding.sampler)

    stream = getattr(binding, "stream", None)
    if stream is not None and stream != STREAM_DEFAULT:
        d["stream"] = stream

    return d


def _applied_loss_to_dict(applied_loss: AppliedLoss) -> dict[str, Any]:
    """Convert an AppliedLoss to a YAML dict."""
    from modularml.core.experiment.experiment_context import ExperimentContext

    ctx = ExperimentContext.get_active()

    # Resolve node label from node_id
    try:
        node = ctx.get_node(node_id=applied_loss.node_id)
        node_label = node.label
    except Exception:  # noqa: BLE001
        node_label = applied_loss.node_id

    # Serialize the Loss object
    loss_cfg = _loss_to_dict(applied_loss.loss)

    d: dict[str, Any] = {
        "loss": loss_cfg,
        "node": node_label,
        "inputs": dict(applied_loss.inputs),
        "weight": applied_loss.weight,
    }
    if applied_loss.label is not None:
        d["label"] = applied_loss.label

    return d


def _callback_to_dict(callback: Any) -> dict[str, Any]:
    """Serialize a Callback to a YAML-friendly dict."""
    from modularml.core.io.yaml.translators.callback import callback_to_yaml_dict

    return callback_to_yaml_dict(callback)


def _checkpointing_to_dict(checkpointing: Checkpointing) -> dict[str, Any]:
    """Serialize a Checkpointing instance to a YAML-friendly dict via its get_config."""
    return checkpointing.get_config()


def _loss_to_dict(loss: Loss) -> dict[str, Any]:
    """
    Serialize a Loss object to a YAML-friendly dict.

    String-named losses (e.g. ``Loss("mse", backend="torch")``) are preserved
    as their string name plus backend. This is the preferred form for standard
    backend losses.
    Custom callables and factories are serialized as their fully
    qualified dotted import path so they can be resolved on import.
    """
    from modularml.core.io.yaml.utils import yaml_safe_cfg

    cfg = loss.get_config()

    # Convert callable "loss" field (Loss.fn, Case 3) to a dotted path
    # String names (Cases 1 & 2) are already the correct YAML representation
    fn = cfg.get("loss")
    if fn is not None and not isinstance(fn, str):
        cfg["loss"] = callable_to_path(fn)

    # Convert factory callable to dotted path if present
    factory = cfg.get("factory")
    if factory is not None and callable(factory):
        cfg["factory"] = callable_to_path(factory)

    # yaml_safe_cfg rebuilds all nested containers as fresh copies, preventing
    # PyYAML from emitting anchors/aliases when the same Loss instance is
    # serialized more than once (e.g. shared between eval_phase.losses and
    # an EvalLossMetric).
    return yaml_safe_cfg({k: v for k, v in cfg.items() if v is not None})


# ================================================
# Import
# ================================================
def phase_from_yaml_dict(d: dict[str, Any]) -> ExperimentPhase:
    """
    Reconstruct an :class:`ExperimentPhase` from a YAML-friendly dictionary.

    The dict must have exactly one top-level key identifying the phase type
    (``train_phase`` or ``eval_phase``).  All referenced graph nodes must
    already be registered in the active :class:`ExperimentContext`.

    Args:
        d (dict[str, Any]): Dict produced by :func:`phase_to_yaml_dict`, or
            its inner dict when ``kind`` is supplied.

    Returns:
        ExperimentPhase: Reconstructed phase.

    Raises:
        KeyError: If the phase type key is missing.

    """
    if "train_phase" in d:
        return _train_phase_from_dict(d["train_phase"])
    if "eval_phase" in d:
        return _eval_phase_from_dict(d["eval_phase"])
    if "fit_phase" in d:
        return _fit_phase_from_dict(d["fit_phase"])

    msg = "YAML dict must have a 'train_phase', 'eval_phase', or 'fit_phase' top-level key."
    raise KeyError(msg)


def _train_phase_from_dict(d: dict[str, Any]) -> TrainPhase:
    """Reconstruct a TrainPhase from its inner YAML dict."""
    from modularml.core.experiment.phases.train_phase import TrainPhase

    input_sources = [
        _input_binding_from_dict(b, for_training=True)
        for b in d.get("input_sources", [])
    ]

    losses = None
    if d.get("losses"):
        losses = [_applied_loss_from_dict(ls) for ls in d["losses"]]

    callbacks = None
    if d.get("callbacks"):
        from modularml.core.io.yaml.translators.callback import callback_from_yaml_dict

        callbacks = [callback_from_yaml_dict(cfg) for cfg in d["callbacks"]]

    checkpointing = None
    if d.get("checkpointing"):
        from modularml.core.experiment.checkpointing import Checkpointing

        checkpointing = Checkpointing.from_config(d["checkpointing"])

    return TrainPhase(
        label=d["label"],
        input_sources=input_sources,
        losses=losses,
        n_epochs=d.get("n_epochs", 1),
        active_nodes=d.get("active_nodes"),
        batch_schedule=d.get("batch_schedule", "zip_strict"),
        callbacks=callbacks,
        checkpointing=checkpointing,
        result_recording=d.get("result_recording", "all"),
        accelerator=d.get("accelerator"),
    )


def _eval_phase_from_dict(d: dict[str, Any]) -> EvalPhase:
    """Reconstruct an EvalPhase from its inner YAML dict."""
    from modularml.core.experiment.phases.eval_phase import EvalPhase

    input_sources = [
        _input_binding_from_dict(b, for_training=False)
        for b in d.get("input_sources", [])
    ]

    losses = None
    if d.get("losses"):
        losses = [_applied_loss_from_dict(ls) for ls in d["losses"]]

    callbacks = None
    if d.get("callbacks"):
        from modularml.core.io.yaml.translators.callback import callback_from_yaml_dict

        callbacks = [callback_from_yaml_dict(cfg) for cfg in d["callbacks"]]

    return EvalPhase(
        label=d["label"],
        input_sources=input_sources,
        losses=losses,
        active_nodes=d.get("active_nodes"),
        batch_size=d.get("batch_size"),
        callbacks=callbacks,
        accelerator=d.get("accelerator"),
    )


def _fit_phase_from_dict(d: dict[str, Any]) -> FitPhase:
    """Reconstruct a FitPhase from its inner YAML dict."""
    from modularml.core.experiment.phases.fit_phase import FitPhase

    input_sources = [
        _input_binding_from_dict(b, for_training=False)
        for b in d.get("input_sources", [])
    ]

    losses = None
    if d.get("losses"):
        losses = [_applied_loss_from_dict(ls) for ls in d["losses"]]

    callbacks = None
    if d.get("callbacks"):
        from modularml.core.io.yaml.translators.callback import callback_from_yaml_dict

        callbacks = [callback_from_yaml_dict(cfg) for cfg in d["callbacks"]]

    return FitPhase(
        label=d["label"],
        input_sources=input_sources,
        losses=losses,
        active_nodes=d.get("active_nodes"),
        freeze_after_fit=d.get("freeze_after_fit", True),
        callbacks=callbacks,
        accelerator=d.get("accelerator"),
    )


def _input_binding_from_dict(d: dict[str, Any], *, for_training: bool) -> InputBinding:
    """Reconstruct an InputBinding from its YAML dict."""
    from modularml.core.experiment.phases.phase import InputBinding

    node_label = d["node"]
    upstream_ref = ref_from_yaml_dict(d["featureset"])
    split = d.get("split")
    stream_val = d.get("stream")

    if for_training:
        sampler_d = d.get("sampler")
        if sampler_d is None:
            msg = f"Training input_source for node '{node_label}' is missing 'sampler'."
            raise ValueError(msg)
        sampler = sampler_from_yaml_dict(sampler_d)

        kwargs: dict[str, Any] = {
            "node": node_label,
            "sampler": sampler,
            "upstream": upstream_ref,
            "split": split,
        }
        if stream_val is not None:
            kwargs["stream"] = stream_val
        return InputBinding.for_training(**kwargs)

    return InputBinding.for_evaluation(
        node=node_label,
        upstream=upstream_ref,
        split=split,
    )


def _applied_loss_from_dict(d: dict[str, Any]) -> AppliedLoss:
    """Reconstruct an AppliedLoss from its YAML dict."""
    from modularml.core.training.applied_loss import AppliedLoss

    loss = _loss_from_dict(d["loss"])
    inputs = d.get("inputs", {})
    # inputs stored as {"0": "targets.soh"} -> pass as list or dict
    # AppliedLoss.__init__ normalizes both list and dict
    return AppliedLoss(
        loss=loss,
        on=d["node"],
        inputs=inputs,
        weight=d.get("weight", 1.0),
        label=d.get("label"),
    )


def _loss_from_dict(d: dict[str, Any]) -> Loss:
    """Reconstruct a Loss object from its YAML dict."""
    import contextlib

    from modularml.core.training.loss import Loss

    d = dict(d)

    # Resolve dotted paths back to callable/class; keep as string if not resolvable
    for key in ("loss", "factory"):
        val = d.get(key)
        if isinstance(val, str) and "." in val:
            with contextlib.suppress(Exception):
                d[key] = resolve_class(val)

    return Loss(**d)
