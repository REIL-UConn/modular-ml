"""Progress bar column definitions and reusable styles."""

from __future__ import annotations

from dataclasses import dataclass

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    ProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.text import Text


class OptionalTextColumn(TextColumn):
    """A TextColumn that only renders when a tracked field value is not None."""

    def __init__(self, text_format: str, *, field_name: str, **kwargs):
        super().__init__(text_format, **kwargs)
        self.field_name = field_name

    def render(self, task):
        """Render the column only when the configured field has a value."""
        if task.fields.get(self.field_name) is None:
            return Text("")
        return super().render(task)


@dataclass
class ProgressStyle:
    """
    Container describing a named progress style.

    Attributes:
        name (str): Unique style identifier.
        columns (list[ProgressColumn]): Rich column layout for the progress bar.
        default_fields (dict[str, object]): Optional default task fields.

    """

    name: str
    columns: list[ProgressColumn]
    indent_group: int = 0
    default_fields: dict[str, object] = None


style_sampling = ProgressStyle(
    name="sampling",
    indent_group=1,
    columns=(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("|"),
        TimeElapsedColumn(),
        TextColumn("|"),
        TimeRemainingColumn(),
    ),
)

style_training = ProgressStyle(
    name="training",
    indent_group=1,
    columns=(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("|"),
        TimeElapsedColumn(),
        TextColumn("|"),
        TimeRemainingColumn(),
    ),
)

style_training_loss = ProgressStyle(
    name="training_loss",
    indent_group=2,
    columns=(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("|"),
        TextColumn(
            ("loss={task.fields[loss_total]:.6f}"),
            justify="right",
        ),
        OptionalTextColumn(
            "| val_loss={task.fields[val_loss]:.6f}",
            field_name="val_loss",
            justify="right",
        ),
    ),
    default_fields={
        "loss_total": 0.0,
        "loss_train": 0.0,
        "loss_aux": 0.0,
        "val_loss": None,
    },
)

style_evaluation = ProgressStyle(
    name="evaluation",
    indent_group=1,
    columns=(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("|"),
        TimeElapsedColumn(),
        TextColumn("|"),
        TimeRemainingColumn(),
    ),
)

style_cv = ProgressStyle(
    name="cross_validation",
    columns=(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("|"),
        TimeElapsedColumn(),
        TextColumn("|"),
        TimeRemainingColumn(),
    ),
)
