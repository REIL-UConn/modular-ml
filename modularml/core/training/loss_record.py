from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from modularml.utils.data.conversion import to_python


@dataclass
class LossRecord:
    """
    A container for storing the result of computing a loss value.

    Attributes:
        label (str): Identifier for the loss (e.g., "mse_loss", "triplet_margin").
        value (Any): Computed loss value (typically a backend-specific tensor or scalar).

    """

    value: Any  # raw value output by AppliedLoss.compute
    label: str  # AppliedLoss.label
    # True = loss is a trainable loss, False = auxillary loss
    contributes_to_update: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "label": self.label,
            "contributes_to_update": self.contributes_to_update,
        }


class LossCollection:
    def __init__(self, records: list[LossRecord]):
        self._records = records

    @property
    def trainable(self) -> Any:
        vals = [r.value for r in self._records if r.contributes_to_update]
        return sum(vals) if vals else 0.0

    @property
    def auxiliary(self) -> Any:
        vals = [r.value for r in self._records if not r.contributes_to_update]
        return sum(vals) if vals else 0.0

    @property
    def total(self) -> Any:
        return self.trainable + self.auxiliary

    def to_float(self) -> LossCollection:
        lrs = []
        for rec in self._records:
            lr = LossRecord(
                value=to_python(rec.value),
                label=rec.label,
                contributes_to_update=rec.contributes_to_update,
            )
            lrs.append(lr)

        return LossCollection(records=lrs)

    def by_label(self, *, as_float: bool = True) -> dict[str, float | Any]:
        """Group losses by label, return either floats or backend values."""
        grouped: dict[str, list[LossRecord]] = {}
        for rec in self._records:
            grouped.setdefault(rec.label, []).append(rec)

        def _sum(vals):
            return sum(vals) if vals else 0.0

        out = {}
        for label, recs in grouped.items():
            vals = [r.value for r in recs]
            agg = _sum(vals)
            out[label] = (
                float(agg.item())
                if (as_float and hasattr(agg, "item"))
                else float(agg)
                if as_float
                else agg
            )
        return out

    def __repr__(self) -> str:
        lc_copy = self.to_float()
        return (
            f"LossCollection(total={lc_copy.total:.4f}, "
            f"trainable={lc_copy.trainable:.4f}, "
            f"auxiliary={lc_copy.auxiliary:.4f})"
        )

    def merge(self, other: LossCollection) -> LossCollection:
        """Merge two LossCollections into a new one."""
        if not isinstance(other, LossCollection):
            msg = f"Can only merge with LossCollection, got {type(other)}"
            raise TypeError(msg)

        return LossCollection(records=self._records + other._records)

    def __add__(self, other: LossCollection | int) -> LossCollection:
        """Support `lc1 + lc2` and `sum([lc1, lc2, ...])`."""
        if other == 0:  # identity element for sum()
            return self
        return self.merge(other)

    def __radd__(self, other: LossCollection | int) -> LossCollection:
        """Support `0 + lc1` at the start of sum()."""
        if other == 0:  # identity element for sum()
            return self
        return self.merge(other)

    def as_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "trainable": self.trainable,
            "auxiliary": self.auxiliary,
            "all_records": [rec.as_dict() for rec in self._records],
        }
