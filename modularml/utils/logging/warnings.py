"""Warning emission utilities with consistent ModularML formatting."""

from __future__ import annotations

import inspect
import warnings
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING

from .logger import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

_logger = get_logger("warnings")


WarningPayload = dict[str, object]
_WARNING_HOOK: ContextVar[Callable[[WarningPayload], bool] | None] = ContextVar(
    "_WARNING_HOOK",
    default=None,
)


def _resolve_caller_frame(stacklevel: int) -> inspect.FrameInfo | None:
    """
    Safely resolve the caller frame based on stacklevel.

    Falls back to the outermost available frame if the requested
    depth exceeds the call stack.

    Args:
        stacklevel (int): Stack depth relative to :func:`warn`.

    Returns:
        inspect.FrameInfo | None: Frame info for the resolved caller, if any.

    """
    frame = inspect.currentframe()
    try:
        # Move up the stack: currentframe -> warn -> caller (+ stacklevel)
        for _ in range(stacklevel + 1):
            if frame is None or frame.f_back is None:
                break
            frame = frame.f_back
        return frame
    finally:
        # Prevent reference cycles
        del frame


def warn(
    message: str,
    *,
    category: type[Warning] = UserWarning,
    hints: str | Iterable[str] | None = None,
    stacklevel: int = 1,
) -> None:
    """
    Emit a formatted ModularML warning with optional hints and source context.

    This function should be used instead of :func:`warnings.warn` inside ModularML
    modules. It automatically captures the calling location, formats the
    warning, and routes it through the ModularML logging system.

    Args:
        message (str): Warning message text.
        category (type[Warning]): Warning category class.
        hints (str | Iterable[str] | None): Optional hint or list of hints suggesting corrective actions.
        stacklevel (int): Stack level adjustment for locating the user call site.

    """
    # Resolve caller frame
    frame = _resolve_caller_frame(stacklevel)
    if frame is not None:
        filename = frame.f_code.co_filename
        lineno = frame.f_lineno
    else:
        filename = "<unknown>"
        lineno = 0

    payload = {
        "category": category,
        "filename": filename,
        "lineno": lineno,
        "message": message,
        "hints": hints,
    }

    # Optional hook to intercept warning prior to logging
    hook = _WARNING_HOOK.get()
    if hook is not None:
        handled = hook(payload)
        if handled:
            return

    _logger.warning(
        message,
        extra={f"warning_{k}": v for k, v in payload.items()},  # preprend "warning_"
    )

    # Register warning via stdlib but don't print
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category)
        warnings.warn(
            message,
            category=category,
            stacklevel=stacklevel + 2,
        )


@contextmanager
def catch_warnings():
    """
    Catch ModularML warnings within the current execution context.

    Allows inspection, suppression, or replacement of warnings before
    they are logged.

    Yields:
        WarningInterceptor: Captured warning accessor for the scoped block.

    """
    captured: list[WarningPayload] = []

    def hook(payload: WarningPayload) -> bool:
        captured.append(payload)
        return True  # suppress original emission

    token = _WARNING_HOOK.set(hook)
    try:
        yield WarningInterceptor(captured)
    finally:
        _WARNING_HOOK.reset(token)


class WarningInterceptor:
    """
    Container that exposes captured warning payloads during interception.

    Attributes:
        _captured (list[WarningPayload]): Stored warning payloads in emission order.

    """

    def __init__(self, captured: list[dict]):
        """
        Initialize the interceptor with captured payloads.

        Args:
            captured (list[dict]): Warning payloads collected by :func:`catch_warnings`.

        """
        self._captured = captured

    def match(self, text: str) -> bool:
        """
        Return True if any captured warning message contains `text`.

        Args:
            text (str): Substring to search within captured messages.

        Returns:
            bool: True if at least one message contains `text`.

        """
        return any(text in w["message"] for w in self._captured)
