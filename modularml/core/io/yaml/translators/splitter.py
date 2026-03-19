"""Translator between BaseSplitter instances and YAML-friendly dicts."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from modularml.core.io.yaml.class_resolver import class_to_path

if TYPE_CHECKING:
    from modularml.core.splitting.base_splitter import BaseSplitter


def splitter_to_yaml_dict(splitter: BaseSplitter) -> dict[str, Any]:
    """
    Export a splitter instance to a YAML-friendly dictionary.

    The output includes a ``splitter_class`` key with the fully qualified class
    path, plus all constructor hyperparameters from :meth:`get_config`.

    Args:
        splitter (BaseSplitter): Splitter instance to export.

    Returns:
        dict[str, Any]: YAML-friendly representation.

    """
    cfg = splitter.get_config()
    cfg["splitter_class"] = class_to_path(type(splitter))
    return cfg


def splitter_from_yaml_dict(d: dict[str, Any]) -> BaseSplitter:
    """
    Reconstruct a splitter from a YAML-friendly dictionary.

    Args:
        d (dict[str, Any]): Dict previously produced by :func:`splitter_to_yaml_dict`.
            Must contain a ``splitter_name`` key (set by the subclass ``get_config``).

    Returns:
        BaseSplitter: Reconstructed splitter instance.

    Raises:
        KeyError: If ``splitter_name`` is missing from ``d``.

    """
    from modularml.core.splitting.base_splitter import BaseSplitter

    d = dict(d)
    d.pop("splitter_class", None)
    return BaseSplitter.from_config(d)
