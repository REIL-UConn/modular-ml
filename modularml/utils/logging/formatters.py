"""Logging formatter implementations for ModularML outputs."""

import logging
import textwrap
from datetime import datetime
from pathlib import Path

from modularml.utils.environment.environment import IN_NOTEBOOK


class ModularMLFormatter(logging.Formatter):
    """Custom log formatter with consistent timestamp, level, and source context."""

    def format(self, record: logging.LogRecord) -> str:
        """
        Format a :class:`logging.LogRecord` into a single-line message.

        Args:
            record (logging.LogRecord): Log record to format.

        Returns:
            str: Formatted log line.

        """
        timestamp = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
        level = record.levelname.ljust(8)
        location = f"{record.module}:{record.lineno}"
        message = record.getMessage()

        return f"[{timestamp}] {level} {location} | {message}"


class _BannerMixin:
    """Shared helpers for banner-style formatter implementations."""

    max_width: int = 88

    def _wrap(self, text: str, *, indent: int = 2) -> list[str]:
        """
        Wrap text to the configured banner width, preserving explicit newlines.

        Args:
            text (str): Message to wrap.
            indent (int): Spaces to indent each wrapped line.

        Returns:
            list[str]: Wrapped and indented lines.

        """
        pad = " " * indent
        width = self.max_width - 2 * indent
        result: list[str] = []
        for line in text.split("\n"):
            if not line.strip():
                result.append(pad)
            else:
                # Preserve the line's own leading whitespace
                leading = line[: len(line) - len(line.lstrip())]
                line_pad = pad + leading
                result.extend(
                    line_pad + part
                    for part in textwrap.wrap(
                        line.strip(),
                        width=width - len(leading),
                        replace_whitespace=True,
                        drop_whitespace=True,
                    )
                )
        return result

    def _supports_color(self) -> bool:
        """Return True if ANSI color output is supported."""
        import os
        import sys

        return sys.stdout.isatty() and os.environ.get("TERM") not in (None, "dumb")

    def _color(self, text: str, *, code: str) -> str:
        """Wrap text using ANSI codes when available."""
        if not self._supports_color():
            return text
        return f"\033[{code}m{text}\033[0m"

    def _separator(self, label: str | None = None) -> str:
        """Return a banner separator line optionally labeled with `label`."""
        if label:
            core = f" {label} "
            side = (self.max_width - len(core)) // 2
            line = "─" * side + core + "─" * (self.max_width - side - len(core))
        else:
            line = "─" * self.max_width
        return line


class ModularMLBannerFormatter(ModularMLFormatter, _BannerMixin):
    """
    Banner-style formatter for standard ModularML logs.

    Label format:
        "{LEVEL}" or "{LEVEL} - {custom_title_desc}"

    Body format:
        [timestamp] module:lineno   # optional
        message
    """

    def __init__(self, *, max_width: int = 88) -> None:
        """
        Initialize the formatter with an optional maximum width.

        Args:
            max_width (int): Maximum banner width in characters.

        """
        super().__init__()
        self.max_width = max_width

    def format(self, record: logging.LogRecord) -> str:
        """
        Format a :class:`logging.LogRecord` into a banner block.

        Args:
            record (logging.LogRecord): Log record to format.

        Returns:
            str: Multi-line banner string.

        """
        level = record.levelname

        # Optional custom title_desc:
        #   logger.info("msg", extra={"title_desc": "FeatureSet"})
        custom = getattr(record, "title_desc", None)
        title_desc = f"{level} - {custom}" if custom else level

        # Optional timestamp + location line
        #   logger.info("msg", extra={"omit_origin": True})
        omit_origin = getattr(record, "omit_origin", False)
        timestamp = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
        location = f"{record.module}:{record.lineno}"
        origin = f"[{timestamp}] {location}"

        message = record.getMessage()

        lines: list[str] = []
        lines.append(self._separator(title_desc))
        if not omit_origin:
            lines.extend(self._wrap(origin, indent=1))
        lines.extend(self._wrap(message, indent=1))
        lines.append(self._separator())
        return "\n".join(lines)


class WarningFormatter(ModularMLFormatter, _BannerMixin):
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
        """Return a shortened filename for display."""
        return Path(filename).name

    def _red(self, text: str) -> str:
        """
        Color text red if supported.

        Args:
            text (str): Message text to wrap.

        Returns:
            str: Colored text when supported, otherwise the original text.

        """
        return self._color(text=text, code=31)

    def format(self, record: logging.LogRecord) -> str:
        """
        Format a warning log record.

        Expected `record.extra` fields:
            - warning_category
            - warning_filename
            - warning_lineno
            - warning_message
            - warning_hints

        Args:
            record (logging.LogRecord): Log record to format.

        Returns:
            str: Banner-rendered warning string.

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
