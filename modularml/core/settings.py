"""Package-wide configuration for ModularML."""

from __future__ import annotations

import os
from dataclasses import dataclass

_TRUE = ("1", "true", "yes", "on")
_FALSE = ("0", "false", "no", "off")


@dataclass
class ModularMLSettings:
    """
    Package-wide settings for ModularML.

    Attributes:
        require_unique_labels (bool):
            When True, ExperimentContext enforces unique node labels using the
            ERROR registration policy, regardless of per-context policy.
            Configurable via the ``MODULARML_REQUIRE_UNIQUE_LABELS`` env var
            or :func:`configure`.
        show_progress (bool):
            When False, no progress bar or spinner is displayed, whatever the
            per-call ``show_*_progress`` arguments say. Intended for batch jobs
            and for programs that run many short phases, where starting and
            stopping the live display costs more than the phase itself.
            Configurable via the ``MODULARML_SHOW_PROGRESS`` env var or
            :func:`configure`. Defaults to True.

    """

    require_unique_labels: bool = False
    show_progress: bool = True


def _env_flag(name: str, *, default: bool) -> bool:
    val = os.getenv(name, "").strip().lower()
    if val in _TRUE:
        return True
    if val in _FALSE:
        return False
    return default


def _init_defaults() -> ModularMLSettings:
    return ModularMLSettings(
        require_unique_labels=_env_flag(
            "MODULARML_REQUIRE_UNIQUE_LABELS",
            default=False,
        ),
        show_progress=_env_flag("MODULARML_SHOW_PROGRESS", default=True),
    )


settings: ModularMLSettings = _init_defaults()
"""Module-level settings singleton. Mutate directly or use :func:`configure`."""


def configure(
    *,
    require_unique_labels: bool | None = None,
    show_progress: bool | None = None,
) -> None:
    """
    Programmatically configure package-wide ModularML settings.

    Args:
        require_unique_labels (bool | None, optional):
            When True, all new ExperimentContexts enforce unique node labels
            (ERROR registration policy). Defaults to no change.
        show_progress (bool | None, optional):
            When False, disables every progress bar and spinner. Defaults to
            no change.

    """
    if require_unique_labels is not None:
        settings.require_unique_labels = require_unique_labels
    if show_progress is not None:
        settings.show_progress = show_progress
