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

    def by_label(self) -> dict[str, LossCollection]:
        """Group loss records by loss label."""
        grouped: dict[str, list[LossRecord]] = {}
        for rec in self._records:
            grouped.setdefault(rec.label, []).append(rec)
        return {k: LossCollection(records=vs) for k, vs in grouped.items()}

    @classmethod
    def merge(cls, collections: Iterable[LossCollection]) -> LossCollection:
        """Merge all collections into a new instance."""
        all_recs: list[LossRecord] = []
        for c in collections:
            if not isinstance(c, cls):
                msg = f"Expected LossCollection, got {type(c)}"
                raise TypeError(msg)
            all_recs.extend(c._records)
        return cls(records=all_recs)

    def merge_with(self, *others: LossCollection) -> LossCollection:
        """Merge other LossCollections into this one."""
        return LossCollection.merge(self, *others)

    # ================================================
    # Reduction / Aggregation
    # ================================================
    @classmethod
    def mean(cls, collections: Iterable[LossCollection]) -> LossCollection:
        """
        Compute the mean of multiple LossCollections, grouped by label.

        Description:
            Groups all loss records by (label, contributes_to_update) and
            computes the mean value for each group. The denominator is the
            total number of records in each group, not the number of
            collections.

            Values are cast to Python floats before averaging.

        Args:
            collections (Iterable[LossCollection]):
                LossCollections to average.

        Returns:
            LossCollection:
                A new LossCollection with one averaged record per unique
                (label, contributes_to_update) combination.

        Raises:
            ValueError:
                If collections are empty.

        """
        lcs = list(collections)
        if not lcs:
            raise ValueError("Cannot average an empty list of LossCollections.")

        # Convert all to float
        lcs = [c.to_float() for c in lcs]

        # Group all records by (label, contributes_to_update)
        grouped: dict[tuple[str, bool], list[float]] = {}
        for lc in lcs:
            for rec in lc._records:
                key = (rec.label, rec.contributes_to_update)
                grouped.setdefault(key, []).append(rec.value)

        # Compute mean for each group (divide by number of records, not lcs)
        avg_records: list[LossRecord] = []
        for (label, contributes), values in grouped.items():
            mean_val = sum(values) / len(values)
            avg_records.append(
                LossRecord(
                    value=mean_val,
                    label=label,
                    contributes_to_update=contributes,
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
