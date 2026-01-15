from __future__ import annotations

from collections.abc import Iterable

SummaryRow = tuple[str, str | Iterable["SummaryRow"]]


def _truncate(line: str, max_width: int) -> str:
    if len(line) <= max_width:
        return line
    if max_width <= 3:
        return "." * max_width
    return line[: max_width - 3] + "..."


def _try_inline(key: str, rows: Iterable[SummaryRow]) -> str | None:
    """
    Try to render iterable inline.

    Rules:
    - (k, "")  → render as `k`
    - (k, v)   → render as `k=v`
    - Nested iterables → not inlineable
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
    """Format nested summary rows into a bordered, width-limited summary box."""
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

    Uses _format_summary_rows if object has that attribute, otherwise
    uses repr and safely casts any multi-line repr strings.
    """
    if hasattr(obj, "_summary_rows"):
        return obj._summary_rows()

    s = repr(obj)
    if "\n" not in s:
        return s
    lines = s.splitlines()
    return [(str(i), line) for i, line in enumerate(lines)]


class Summarizable:
    def _summary_rows(self) -> list[SummaryRow]: ...
    def summary(self, max_width: int = 88) -> str:
        return format_summary_box(
            title=self.__class__.__name__,
            rows=self._summary_rows(),
            max_width=max_width,
        )
