"""Translator between BaseSampler instances and YAML-friendly dicts."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from modularml.core.io.yaml.class_resolver import class_to_path
from modularml.core.io.yaml.translators.similarity_condition import (
    similarity_condition_from_yaml_dict,
    similarity_condition_to_yaml_dict,
)

if TYPE_CHECKING:
    from modularml.core.sampling.base_sampler import BaseSampler
    from modularml.core.sampling.similiarity_condition import SimilarityCondition


def _condition_mapping_to_yaml(
    condition_mapping: dict[str, dict[str, SimilarityCondition]],
) -> dict[str, dict[str, Any]]:
    """Convert nested SimilarityCondition objects to YAML-friendly dicts."""
    return {
        role: {
            col: similarity_condition_to_yaml_dict(cond) for col, cond in conds.items()
        }
        for role, conds in condition_mapping.items()
    }


def _condition_mapping_from_yaml(
    raw: dict[str, dict[str, Any]],
) -> dict[str, dict[str, SimilarityCondition]]:
    """Reconstruct nested SimilarityCondition objects from YAML dicts."""
    return {
        role: {
            col: similarity_condition_from_yaml_dict(cond)
            for col, cond in conds.items()
        }
        for role, conds in raw.items()
    }


def sampler_to_yaml_dict(sampler: BaseSampler) -> dict[str, Any]:
    """
    Export a sampler instance to a YAML-friendly dictionary.

    The output includes a ``sampler_class`` key with the fully qualified class
    path, plus all constructor hyperparameters from :meth:`get_config`.
    Any ``condition_mapping`` values containing :class:`SimilarityCondition`
    objects are converted to plain dicts.

    Args:
        sampler (BaseSampler): Sampler instance to export.

    Returns:
        dict[str, Any]: YAML-friendly representation.

    """
    cfg = sampler.get_config()
    cfg["sampler_class"] = class_to_path(type(sampler))

    if "condition_mapping" in cfg:
        cfg["condition_mapping"] = _condition_mapping_to_yaml(cfg["condition_mapping"])

    return cfg


def sampler_from_yaml_dict(d: dict[str, Any]) -> BaseSampler:
    """
    Reconstruct a sampler from a YAML-friendly dictionary.

    Args:
        d (dict[str, Any]): Dict previously produced by :func:`sampler_to_yaml_dict`.
            Must contain a ``sampler_name`` key (set by the subclass ``get_config``).

    Returns:
        BaseSampler: Reconstructed sampler instance.

    Raises:
        KeyError: If ``sampler_name`` is missing from ``d``.

    """
    from modularml.core.sampling.base_sampler import BaseSampler

    d = dict(d)
    d.pop("sampler_class", None)

    if "condition_mapping" in d and isinstance(d["condition_mapping"], dict):
        d["condition_mapping"] = _condition_mapping_from_yaml(d["condition_mapping"])

    return BaseSampler.from_config(d)
