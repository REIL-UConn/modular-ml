from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from modularml.utils.data.conversion import to_python

if TYPE_CHECKING:
    from collections.abc import Iterable


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

    # ================================================
    # Properties
    # ================================================
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

    @property
    def loss_labels(self) -> list[str]:
        lbls = {rec.label for rec in self._records}
        return list(lbls)

    # ================================================
    # Data Formatting and Casting
    # ================================================
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

    def as_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "trainable": self.trainable,
            "auxiliary": self.auxiliary,
            "all_records": [rec.as_dict() for rec in self._records],
        }

    # ================================================
    # Merging
    # ================================================
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

    def by_label(self, *, as_float: bool = True) -> dict[str, float | Any]:
        """Group losses by loss label, return either floats or backend values."""
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

    def merge(self, *others: LossCollection) -> LossCollection:
        """Merge other LossCollections into this one."""

        def merge_all(*collections: LossCollection) -> LossCollection:
            all_recs: list[LossRecord] = []
            for c in collections:
                if not isinstance(c, LossCollection):
                    msg = f"Expected LossCollection, got {type(c)}"
                    raise TypeError(msg)
                all_recs.extend(c._records)
            return LossCollection(records=all_recs)

        if isinstance(self, type):
            return merge_all(*others)
        return merge_all(self, *others)

    # ================================================
    # Reduction / Aggregation
    # ================================================
    @classmethod
    def mean(cls, collections: Iterable[LossCollection]) -> LossCollection:
        """
        Compute the elementwise mean of multiple LossCollections.

        Description:
            All LossCollections must:
              - have the same number of LossRecords
              - have matching (label, contributes_to_update) per record index

            Values are cast to Python floats before averaging.

        Args:
            collections (Iterable[LossCollection]):
                LossCollections to average.

        Returns:
            LossCollection:
                A new LossCollection with averaged loss values.

        Raises:
            ValueError:
                If collections are empty or structurally incompatible.

        """
        lcs = list(collections)
        if not lcs:
            raise ValueError("Cannot average an empty list of LossCollections.")

        # Convert all to float
        lcs = [c.to_float() for c in lcs]

        # Use first as reference for n_records and labels
        ref = lcs[0]
        n_records = len(ref._records)
        for lc in lcs[1:]:
            if len(lc._records) != n_records:
                msg = "All LossCollections must have the same number of records."
                raise ValueError(msg)
            for i, (r0, r1) in enumerate(zip(ref._records, lc._records, strict=True)):
                if (
                    r0.label != r1.label
                    or r0.contributes_to_update != r1.contributes_to_update
                ):
                    msg = (
                        "LossCollection structure mismatch at index "
                        f"{i}: ({r0.label}, {r0.contributes_to_update}) "
                        f"!= ({r1.label}, {r1.contributes_to_update})"
                    )
                    raise ValueError(msg)

        # Average values per record
        avg_records: list[LossRecord] = []
        n = len(lcs)
        for i in range(n_records):
            values = [lc._records[i].value for lc in lcs]
            mean_val = sum(values) / n

            rec0 = ref._records[i]
            avg_records.append(
                LossRecord(
                    value=mean_val,
                    label=rec0.label,
                    contributes_to_update=rec0.contributes_to_update,
                ),
            )

        return cls(records=avg_records)

    # ================================================
    # Representation
    # ================================================
    def __repr__(self) -> str:
        lc_copy = self.to_float()
        return (
            f"LossCollection(total={lc_copy.total:.4f}, "
            f"trainable={lc_copy.trainable:.4f}, "
            f"auxiliary={lc_copy.auxiliary:.4f}, "
            f"losses={self.loss_labels})"
        )
