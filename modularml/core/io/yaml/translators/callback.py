"""Translator between Callback instances and YAML-friendly dicts."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from modularml.core.io.yaml.utils import yaml_safe_cfg

if TYPE_CHECKING:
    from modularml.callbacks.early_stopping import EarlyStopping
    from modularml.callbacks.eval_loss_metric import EvalLossMetric
    from modularml.callbacks.evaluation import Evaluation
    from modularml.core.experiment.callbacks.callback import Callback


# ================================================
# Export
# ================================================
def callback_to_yaml_dict(callback: Callback) -> dict[str, Any]:
    """
    Export a :class:`Callback` to a YAML-friendly dictionary.

    Args:
        callback (Callback): Callback instance to export.

    Returns:
        dict[str, Any]: YAML-friendly dict with a ``callback_type`` key.

    """
    from modularml.callbacks.early_stopping import EarlyStopping
    from modularml.callbacks.eval_loss_metric import EvalLossMetric
    from modularml.callbacks.evaluation import Evaluation

    if isinstance(callback, Evaluation):
        return _evaluation_to_dict(callback)

    if isinstance(callback, EvalLossMetric):
        return _eval_loss_metric_to_dict(callback)

    if isinstance(callback, EarlyStopping):
        return _early_stopping_to_dict(callback)

    # Fallback for unknown callback types
    return yaml_safe_cfg(callback.get_config())


def _evaluation_to_dict(cb: Evaluation) -> dict[str, Any]:
    """Serialize an Evaluation callback to a YAML dict."""
    from modularml.core.io.yaml.translators.phase import _eval_phase_to_dict

    return {
        "callback_type": cb.__class__.__qualname__,
        "label": cb.label,
        "eval_phase": _eval_phase_to_dict(cb.eval_phase),
        "every_n_epochs": cb.every_n_epochs,
        "run_on_start": cb.run_on_start,
        "metrics": [callback_to_yaml_dict(m) for m in cb._metrics],
    }


def _eval_loss_metric_to_dict(cb: EvalLossMetric) -> dict[str, Any]:
    """Serialize an EvalLossMetric callback to a YAML dict."""
    from modularml.core.io.yaml.translators.phase import _applied_loss_to_dict

    return {
        "callback_type": cb.__class__.__qualname__,
        "name": cb.metric_name,
        "loss": _applied_loss_to_dict(cb._loss),
        "reducer": cb._reducer,
    }


def _early_stopping_to_dict(cb: EarlyStopping) -> dict[str, Any]:
    """Serialize an EarlyStopping callback to a YAML dict (config is already JSON-safe)."""
    return cb.get_config()


# ================================================
# Import
# ================================================
def callback_from_yaml_dict(d: dict[str, Any]) -> Callback:
    """
    Reconstruct a :class:`Callback` from a YAML-friendly dictionary.

    Args:
        d (dict[str, Any]): Dict produced by :func:`callback_to_yaml_dict`.

    Returns:
        Callback: Reconstructed callback.

    Raises:
        KeyError: If ``callback_type`` is missing from ``d``.

    """
    from modularml.core.experiment.callbacks.callback import Callback

    callback_type = d.get("callback_type")

    if callback_type == "Evaluation":
        return _evaluation_from_dict(d)

    if callback_type == "EvalLossMetric":
        return _eval_loss_metric_from_dict(d)

    if callback_type == "EarlyStopping":
        return _early_stopping_from_dict(d)

    # Fallback for unknown callback types
    return Callback.from_config(d)


def _evaluation_from_dict(d: dict[str, Any]) -> Evaluation:
    """Reconstruct an Evaluation callback from its YAML dict."""
    from modularml.callbacks.evaluation import Evaluation
    from modularml.core.io.yaml.translators.phase import _eval_phase_from_dict

    eval_phase = _eval_phase_from_dict(d["eval_phase"])
    metrics = [callback_from_yaml_dict(m) for m in d.get("metrics", [])]

    return Evaluation(
        eval_phase=eval_phase,
        every_n_epochs=d.get("every_n_epochs", 1),
        run_on_start=d.get("run_on_start", False),
        label=d.get("label"),
        metrics=metrics or None,
    )


def _eval_loss_metric_from_dict(d: dict[str, Any]) -> EvalLossMetric:
    """Reconstruct an EvalLossMetric callback from its YAML dict."""
    from modularml.callbacks.eval_loss_metric import EvalLossMetric
    from modularml.core.io.yaml.translators.phase import _applied_loss_from_dict

    loss = _applied_loss_from_dict(d["loss"])

    return EvalLossMetric(
        loss=loss,
        reducer=d.get("reducer", "mean"),
        name=d.get("name", "val_loss"),
    )


def _early_stopping_from_dict(d: dict[str, Any]) -> EarlyStopping:
    """Reconstruct an EarlyStopping callback from its YAML dict."""
    from modularml.callbacks.early_stopping import EarlyStopping

    return EarlyStopping.from_config(d)
