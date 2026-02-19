"""Helpers for rendering nested summaries into ASCII boxes."""

from __future__ import annotations

from collections.abc import Iterable

SummaryRow = tuple[str, str | Iterable["SummaryRow"]]


def _truncate(line: str, max_width: int) -> str:
    """Truncate `line` to `max_width`, adding ellipses when necessary."""
    if len(line) <= max_width:
        return line
    if max_width <= 3:
        return "." * max_width
    return line[: max_width - 3] + "..."


def _try_inline(key: str, rows: Iterable[SummaryRow]) -> str | None:
    """
    Try to render iterable inline.

    Rules:
        - `(k, "")`  → render as `k`
        - `(k, v)`   → render as `k=v`
        - Nested iterables → not inlineable

    Args:
        key (str): Parent label.
        rows (Iterable[SummaryRow]): Child rows to flatten.

    Returns:
        str | None: Inline string or None if expansion is required.

    """
    parts: list[str] = []

    for row in rows:
        if not isinstance(row, tuple) or len(row) != 2:
            return None

        k, v = row

        if (not isinstance(v, str)) and hasattr(v, "__len__"):
            return None

        parts.append(k if v == "" else f"{k}={v}")

    return f"{key} : [{', '.join(parts)}]"


def _flatten_rows(
    rows: Iterable[SummaryRow],
    *,
    max_width: int,
    indent: int = 0,
    indent_str: str = "  ",
) -> list[str]:
    """
    Flatten nested summary rows into formatted strings.

    Args:
        rows (Iterable[SummaryRow]): Rows to flatten.
        max_width (int): Maximum width for each line.
        indent (int): Current indentation level.
        indent_str (str): String used per indent level.

    Returns:
        list[str]: Flattened, width-restricted lines.

    """
    out: list[str] = []
    prefix = indent_str * indent

    for row in rows:
        if not isinstance(row, tuple) or len(row) != 2:
            msg = f"Invalid SummaryRow: {row!r}"
            raise ValueError(msg)

        key, value = row

        # Case 1: Leaf (string value)
        if (not hasattr(value, "__len__")) or isinstance(value, str):
            # Empty string -> label only
            if value == "":
                line = f"{key}"
                out.append(prefix + _truncate(line, max_width))
                continue

            inline = f"{key} : {value}"
            if len(prefix + inline) <= max_width:
                out.append(prefix + inline)
            else:
                # Vertical expansion
                out.append(prefix + f"{key} :")
                child = prefix + indent_str + value
                out.append(_truncate(child, max_width))
            continue

        # Case 2: Iterable (nested)
        vs = list(value)

        inline = _try_inline(key, vs)
        if inline and len(prefix + inline) <= max_width:
            out.append(prefix + inline)
            continue

        # Vertical expansion
        out.append(prefix + f"{key} :")
        out.extend(
            _flatten_rows(
                vs,
                max_width=max_width,
                indent=indent + 1,
                indent_str=indent_str,
            ),
        )

    return out


def format_summary_box(
    *,
    title: str,
    rows: Iterable[SummaryRow],
    max_width: int = 88,
) -> str:
    """
    Format nested summary rows into a bordered, width-limited summary box.

    Args:
        title (str): Box title shown in the header.
        rows (Iterable[SummaryRow]): Rows to render.
        max_width (int): Maximum line width including borders.

    Returns:
        str: Rendered summary box.

    """
    flat = _flatten_rows(rows, max_width=max_width)

    if not flat:
        flat = ["(no data)"]

    # Clamp & pad rows
    content_width = min(
        max(len(r) for r in flat),
        max_width - 4,
    )

    def fmt_line(line: str) -> str:
        line = _truncate(line, content_width)
        return f"│ {line.ljust(content_width)} │"

    top = f"┌─ {title} " + "─" * max(0, content_width - len(title) - 1) + "┐"
    body = "\n".join(fmt_line(r) for r in flat)
    bottom = "└" + "─" * (content_width + 2) + "┘"

    return f"{top}\n{body}\n{bottom}"


def safe_cast_to_summary_rows(obj: object) -> str | list[tuple[str, str]]:
    """
    Normalize an object for SummaryRow usage.

    Uses `_summary_rows` if the object defines it, otherwise falls
    back to `repr` and safely casts multi-line representations.

    Args:
        obj (object): Object to normalize.

    Returns:
        str | list[tuple[str, str]]: Normalized summary content.

    """
    if hasattr(obj, "_summary_rows"):
        return obj._summary_rows()

    s = repr(obj)
    if "\n" not in s:
        return s
    lines = s.splitlines()
    return [(str(i), line) for i, line in enumerate(lines)]


class Summarizable:
    """Mixin that provides a summary box rendering helper."""

    def _summary_rows(self) -> list[SummaryRow]:  # pragma: no cover
        """Return rows used by :meth:`summary`."""

    def summary(self, max_width: int = 88) -> str:
        """
        Render a formatted summary box for this object.

        Args:
            max_width (int): Maximum width for the rendered box.

        Returns:
            str: Summary box string.

        """
        return format_summary_box(
            title=self.__class__.__name__,
            rows=self._summary_rows(),
            max_width=max_width,
        )
