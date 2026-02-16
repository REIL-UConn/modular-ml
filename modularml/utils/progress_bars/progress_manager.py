from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, ClassVar

from rich.console import Console, Group
from rich.live import Live
from rich.padding import Padding
from rich.progress import Progress

from modularml.utils.environment.environment import IN_NOTEBOOK

from .progress_styles import (
    style_cv,
    style_sampling,
    style_training,
    style_training_loss,
)

if TYPE_CHECKING:
    from .progress_styles import ProgressStyle
    from .progress_task import ProgressTask


def _register_ipython_hooks():
    if not IN_NOTEBOOK:
        return

    from IPython import get_ipython

    ip = get_ipython()
    if ip is None:
        return

    # Avoid re-registering
    if getattr(ip, "_mml_progress_hooks_registered", False):
        return
    ip._mml_progress_hooks_registered = True

    def _pre_run_cell(*args, **kwargs):  # noqa: ARG001
        mgr = ProgressManager.get_active()
        mgr._reset_for_new_cell()

    def _post_run_cell(*args, **kwargs):  # noqa: ARG001
        mgr = ProgressManager.get_active()
        mgr._shutdown()

    ip.events.register("pre_run_cell", _pre_run_cell)
    ip.events.register("post_run_cell", _post_run_cell)


class ProgressManager:
    _ACTIVE: ClassVar[ProgressManager | None] = None

    def __init__(self):
        self._console = Console(force_jupyter=IN_NOTEBOOK)
        self._live: Live | None = None

        self._active_tasks: set[ProgressTask] = set()
        self._progress: dict[int, Progress] = {}
        self._task_counter: int = 0

        # Indentation: style-based nesting
        self._style_indent: dict[str, int] = {}
        self._progress_indent: dict[int, int] = {}
        self._next_indent_level: int = 0

        self._styles: dict[str, ProgressStyle] = {
            style_sampling.name: style_sampling,
            style_training.name: style_training,
            style_training_loss.name: style_training_loss,
            style_cv.name: style_cv,
        }

    # ================================================
    # Scope Managemenet
    # ================================================
    @classmethod
    def activate(cls) -> ProgressManager:
        mgr = cls()
        cls._ACTIVE = mgr
        return mgr

    @classmethod
    def get_active(cls) -> ProgressManager:
        if cls._ACTIVE is None:
            cls._ACTIVE = cls()
            _register_ipython_hooks()
        return cls._ACTIVE

    @classmethod
    def deactivate(cls):
        if cls._ACTIVE is not None:
            cls._ACTIVE._shutdown()
            cls._ACTIVE = None

    # ================================================
    # Style Registration
    # ================================================
    def register_style(self, style: ProgressStyle):
        if style.name in self._styles:
            msg = f"Style name '{style.name}' already registered."
            raise ValueError(msg)
        self._styles[style.name] = style

    # ================================================
    # Rich.live Control
    # ================================================
    def _render_group(self):
        if not self._progress:
            return None
        renderables = []
        for key, progress in self._progress.items():
            indent = self._progress_indent.get(key, 0)
            if indent > 0:
                renderables.append(Padding(progress, (0, 0, 0, indent * 2)))
            else:
                renderables.append(progress)
        return Group(*renderables)

    def _ensure_live(self):
        if self._live is not None:
            return
        if not self._progress:
            return

        self._live = Live(
            self._render_group(),
            console=self._console,
            refresh_per_second=10,
            transient=not IN_NOTEBOOK,
        )
        self._live.start()

    def _refresh_layout(self):
        self._ensure_live()
        if self._live is not None:
            self._live.update(self._render_group(), refresh=True)

    def _shutdown(self):
        if self._live is None:
            return

        self._live.stop()
        self._live = None

        if not IN_NOTEBOOK:
            self._progress.clear()
            self._active_tasks.clear()

    # ================================================
    # Task Registration
    # ================================================
    def _reset_for_new_cell(self):
        # Stop any existing Live display
        if self._live is not None:
            with contextlib.suppress(Exception):
                self._live.stop()
            self._live = None

        # Clear task & progress state
        self._active_tasks.clear()
        self._progress.clear()
        self._task_counter = 0
        self._style_indent.clear()
        self._progress_indent.clear()
        self._next_indent_level = 0

    def _attach_task(self, task: ProgressTask):
        style = self._styles[task.style_name]
        base_fields = dict(style.default_fields or {})
        base_fields.update(task.fields or {})

        # Each task gets its own Progress instance, keyed by insertion order
        key = self._task_counter
        self._task_counter += 1

        # Assign indent level based on first-seen style order
        if task.style_name not in self._style_indent:
            self._style_indent[task.style_name] = self._next_indent_level
            self._next_indent_level += 1
        self._progress_indent[key] = self._style_indent[task.style_name]

        self._progress[key] = Progress(
            *style.columns,
            auto_refresh=False,
        )

        self._ensure_live()

        progress = self._progress[key]
        task._task_id = progress.add_task(
            task.description,
            total=task.total,
            **base_fields,
        )
        task._progress_key = key

        self._active_tasks.add(task)
        self._refresh_layout()

    def _mark_task_finished(self, task: ProgressTask):
        self._active_tasks.discard(task)

        progress = self._progress[task._progress_key]

        if not task.persist:
            progress.remove_task(task._task_id)
            del self._progress[task._progress_key]
            self._progress_indent.pop(task._progress_key, None)

        self._refresh_layout()

        # If nothing is running anymore, shut down
        if not self._active_tasks and not IN_NOTEBOOK:
            self._shutdown()
