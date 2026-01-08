import logging
import textwrap
from datetime import datetime
from pathlib import Path

from modularml.utils.environment.environment import IN_NOTEBOOK


class ModularMLFormatter(logging.Formatter):
    """Custom log formatter with consistent timestamp, level, and source context."""

    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
        level = record.levelname.ljust(8)
        location = f"{record.module}:{record.lineno}"
        message = record.getMessage()

        return f"[{timestamp}] {level} {location} | {message}"


class WarningFormatter(logging.Formatter):
    """
    Formatter for ModularML warnings using a banner-style layout.

    Warnings are rendered as visually distinct blocks with top and bottom
    separators. Separator lines are colored red when supported by the
    terminal.
    """

    def __init__(self, *, max_width: int = 88) -> None:
        """
        Initialize the warning formatter.

        Args:
            max_width (int):
                Maximum line width for wrapped text.

        """
        super().__init__()
        self.max_width = max_width

    @staticmethod
    def _short_location(filename: str) -> str:
        return Path(filename).name

    def _supports_color(self) -> bool:
        """Return True if the current stdout supports ANSI color codes."""
        import sys

        if not sys.stdout.isatty():
            return False
        try:
            import os

            return os.environ.get("TERM") not in (None, "dumb")
        except Exception:  # noqa: BLE001
            return False

    def _red(self, text: str) -> str:
        """Color text red if supported."""
        if not self._supports_color():
            return text
        return f"\033[31m{text}\033[0m"

    def _wrap(self, text: str, *, indent: int = 2) -> list[str]:
        pad = " " * indent
        return [
            pad + line
            for line in textwrap.wrap(
                text,
                width=self.max_width - 2 * indent,
                replace_whitespace=True,
                drop_whitespace=True,
            )
        ]

    def _separator(self, label: str | None = None) -> str:
        """Create a separator line, optionally with a centered label."""
        if label:
            core = f" {label} "
            side = (self.max_width - len(core)) // 2
            line = "─" * side + core + "─" * (self.max_width - side - len(core))
        else:
            line = "─" * self.max_width

        return self._red(line)

    def format(self, record: logging.LogRecord) -> str:
        """
        Format a warning log record.

        Expected ``record.extra`` fields:
        - warning_category
        - warning_filename
        - warning_lineno
        - warning_message
        - warning_hints
        """
        category = record.warning_category
        filename = record.warning_filename
        lineno = record.warning_lineno
        message = record.warning_message
        hints = record.warning_hints

        lines: list[str] = []

        # Top separator
        lines.append(self._separator(category.__name__))

        # Header
        if not IN_NOTEBOOK:
            lines.append(f" Location: {self._short_location(filename)}:{lineno}")
            lines.append("")
        # Message body spacing
        lines.extend(self._wrap(message, indent=1))

        # Hints
        if hints:
            lines.append("")
            # lines.append(" Hint:")
            hint_list = [hints] if isinstance(hints, str) else list(hints)
            for hint in hint_list:
                lines.extend(self._wrap(hint, indent=1))

        # Bottom separator
        lines.append(self._separator())

        return "\n".join(lines)
