"""Translator between SimilarityCondition instances and YAML-friendly dicts."""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any

from modularml.core.io.yaml.class_resolver import resolve_class

if TYPE_CHECKING:
    from modularml.core.sampling.similiarity_condition import SimilarityCondition


def similarity_condition_to_yaml_dict(cond: SimilarityCondition) -> dict[str, Any]:
    """
    Export a :class:`SimilarityCondition` to a YAML-friendly dictionary.

    All scalar fields are preserved as-is.  A custom ``metric`` callable is
    serialized as its fully-qualified dotted import path so it can be
    round-tripped via YAML.  ``None`` metrics are omitted.

    Args:
        cond (SimilarityCondition): Condition instance to export.

    Returns:
        dict[str, Any]: YAML-friendly dict.

    Raises:
        ImportError: If ``metric`` is defined in ``__main__`` and therefore
            cannot be imported by name.

    """
    cfg = cond.get_config()

    metric = cfg.get("metric")
    if metric is None:
        cfg.pop("metric", None)
    else:
        import inspect

        from modularml.core.io.yaml.class_resolver import class_to_path

        if inspect.isclass(metric):
            metric_path = class_to_path(metric)
        else:
            metric_path = f"{metric.__module__}.{metric.__qualname__}"

        module = metric_path.rsplit(".", 1)[0]
        if module == "__main__":
            msg = (
                f"Cannot serialize metric callable '{metric_path}': "
                "module '__main__' is not importable. "
                "Define the metric in a proper module for YAML round-trips."
            )
            raise ImportError(msg)

        cfg["metric"] = metric_path

    return cfg


def similarity_condition_from_yaml_dict(d: dict[str, Any]) -> SimilarityCondition:
    """
    Reconstruct a :class:`SimilarityCondition` from a YAML-friendly dictionary.

    Dotted paths stored under ``metric`` are resolved back to callables.
    Missing ``metric`` keys default to ``None``.

    Args:
        d (dict[str, Any]): Dict produced by
            :func:`similarity_condition_to_yaml_dict`.

    Returns:
        SimilarityCondition: Reconstructed condition.

    """
    from modularml.core.sampling.similiarity_condition import SimilarityCondition

    d = dict(d)

    metric_val = d.get("metric")
    if isinstance(metric_val, str):
        with contextlib.suppress(Exception):
            d["metric"] = resolve_class(metric_val)
    else:
        d.setdefault("metric", None)

    return SimilarityCondition(**d)
