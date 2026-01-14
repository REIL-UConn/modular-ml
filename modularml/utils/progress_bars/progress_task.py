from __future__ import annotations

from .progress_manager import ProgressManager


class ProgressTask:
    """A lazily-started task bound to a ProgressContext and style."""

    def __init__(
        self,
        *,
        style: str,
        description: str,
        total: int | None,
        enabled: bool = True,
        persist: bool = True,
    ):
        self.style_name = style
        self.description = description
        self.total = total
        self.enabled = enabled
        self.persist = persist

        self.fields: dict[str, object] = {}

        self._started = False
        self._finished = False
        self._task_id = None

    def start(self):
        """Start this task in the Live display."""
        if not self.enabled or self._started:
            return
        mgr = ProgressManager.get_active()
        mgr._attach_task(self)
        self._started = True

    def tick(self, n: int = 1, **fields):
        """Increment this tasks progress in the Live display."""
        if not self.enabled:
            return
        if not self._started:
            self.start()

        # Update stored fields for final renderable
        self.fields.update(fields)

        # Advance task from global manager
        mgr = ProgressManager.get_active()
        progress = mgr._progress[self.style_name]

        progress.advance(self._task_id, n)
        if fields:
            progress.update(self._task_id, fields=fields)

        # Must refresh layout manually
        mgr._refresh_layout()

    def set_total(self, total: int):
        # Update task total
        self.total = total
        # Update display only if already being displayed
        if not self._started:
            return

        # Update task from global manager
        mgr = ProgressManager.get_active()
        progress = mgr._progress[self.style_name]
        progress.update(self._task_id, total=total)
        mgr._refresh_layout()

    def finish(self, **fields):
        if not self._started or self._finished:
            return
        self._finished = True

        # Update task from global manager
        mgr = ProgressManager.get_active()
        progress = mgr._progress[self.style_name]

        # Update fields one last time
        if fields:
            self.fields.update(fields)
            progress.update(self._task_id, fields=fields)

        task = progress._tasks.get(self._task_id)
        if task is not None and task.total is not None:
            progress.update(self._task_id, completed=task.total)

        # Tell manager this task is no longer active
        mgr._mark_task_finished(self)

    def reset(self):
        """Reset internal state so this task can be reused after a cell reset."""
        self._started = False
        self._finished = False
        self._task_id = None
        self.fields.clear()
