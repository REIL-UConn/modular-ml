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


@dataclass
class ProgressStyle:
    name: str
    columns: list[ProgressColumn]
    default_fields: dict[str, object] = None


style_sampling = ProgressStyle(
    name="sampling",
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
    columns=(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("|"),
        TextColumn(
            (
                "loss={task.fields[loss_total]:.4f} "
                "({task.fields[loss_train]:.3f}/"
                "{task.fields[loss_aux]:.3f})"
            ),
            justify="right",
        ),
    ),
    default_fields={
        "loss_total": 0.0,
        "loss_train": 0.0,
        "loss_aux": 0.0,
    },
)
