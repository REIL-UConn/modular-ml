"""Records for capturing computed losses and aggregations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

from modularml.core.experiment.experiment_context import ExperimentContext
from modularml.core.experiment.experiment_node import ExperimentNode
from modularml.utils.data.conversion import to_python
from modularml.utils.data.multi_keyed_data import AxisSeries

if TYPE_CHECKING:
    from matplotlib.pylab import Hashable


@dataclass
class LossRecord:
    """
    Container for trainable and auxiliary loss values produced by execution steps.

    Attributes:
        label (str):
            Identifier for the loss (for example, `mse_loss`).
        node_id (str | None):
            Identifier of the node that produced the measured data.
        trainable (Any | None):
            Backend value contributing to gradient computation.
        auxiliary (Any | None):
            Backend value excluded from gradient computation.

    """

    label: str
    node_id: str | None = None
    trainable: Any | None = None
    auxiliary: Any | None = None

    # ================================================
    # Properties
    # ================================================
    @property
    def total(self) -> Any:
        """
        Sum of trainable and auxiliary values.

        Returns:
            Any: Backend-specific scalar combining available components.

        """
        return sum([x for x in [self.trainable, self.auxiliary] if x is not None])

    # ================================================
    # Data Formatting and Casting
    # ================================================
    def as_dict(self) -> dict[str, Any]:
        """
        Convert the record to a serializable dictionary.

        Returns:
            dict[str, Any]: Dictionary with label, node, and loss values.

        """
        return {
            "label": self.label,
            "node_id": self.node_id,
            "trainable": self.trainable,
            "auxiliary": self.auxiliary,
        }

    def to_float(self) -> LossRecord:
        """
        Return a copy of this record with backend values cast to floats.

        Description:
            Casting breaks autograd tracking (for example, PyTorch
            backpropagation) on the returned record.

        Returns:
            LossRecord: New record with float-based trainable and auxiliary values.

        """
        return LossRecord(
            label=self.label,
            node_id=self.node_id,
            trainable=to_python(self.trainable),
            auxiliary=to_python(self.auxiliary),
        )

    # ================================================
    # Reduction / Aggregation
    # ================================================
    @classmethod
    def merge(cls, *records: LossRecord) -> LossRecord:
        """
        Merge compatible records into a new instance.

        Args:
            *records (LossRecord):
                Records to merge; either variadic or a single iterable.

        Returns:
            LossRecord:
                Aggregated record with summed trainable and auxiliary values.

        Raises:
            TypeError: If any input is not a :class:`LossRecord`.
            ValueError: If labels or node identifiers do not match.

        """
        # Check if passed list instead of separate args
        if (len(records) == 1) and isinstance(records[0], list):
            records = records[0]

        # Type / value checking
        ref = records[0]
        for lr in records:
            # Validate type
            if not isinstance(lr, LossRecord):
                msg = f"Cannot add non-LossRecord to LossRecord. Received: {type(lr)}."
                raise TypeError(msg)

            # Enforce same label
            if lr.label != ref.label:
                msg = (
                    "Cannot combine LossRecords with different labels: "
                    f"{ref.label} != {lr.label}."
                )
                raise ValueError(msg)

            # Enforce same node_id
            if lr.node_id != ref.node_id:
                msg = (
                    "Cannot combine LossRecords with different node IDs: "
                    f"{ref.node_id} != {lr.node_id}."
                )
                raise ValueError(msg)

        # Combine
        t_vals = [lr.trainable for lr in records if lr.trainable is not None]
        a_vals = [lr.auxiliary for lr in records if lr.auxiliary is not None]
        return LossRecord(
            label=ref.label,
            node_id=ref.node_id,
            trainable=sum(t_vals) if len(t_vals) > 0 else None,
            auxiliary=sum(a_vals) if len(a_vals) > 0 else None,
        )

    def merge_with(self, *others: LossRecord) -> LossRecord:
        """
        Merge this record with additional records.

        Args:
            *others (LossRecord): Additional records to merge.

        Returns:
            LossRecord: Combined record produced by :meth:`LossRecord.merge`.

        """
        return type(self).merge(self, *others)

    def __add__(self, other: LossRecord) -> LossRecord:
        """
        Return a merged record combining this record with `other`.

        Returns:
            LossRecord: Result of :meth:`LossRecord.merge` with both operands.

        """
        return type(self).merge(self, other)

    def __radd__(self, other: LossRecord | int) -> LossRecord:
        """
        Support `sum()` by treating zero as the additive identity.

        Returns:
            LossRecord: Aggregated result respecting commutative addition semantics.

        """
        if other == 0:
            return self
        return type(self).merge(other, self)

    @classmethod
    def mean(cls, *records: LossRecord) -> LossRecord:
        """
        Compute the mean of all provided records.

        Description:
            Trainable and auxiliary values are averaged over the count of records that
            define each component after converting values to Python floats.

        Args:
            *records (LossRecord): Records to average; accepts a single iterable.

        Returns:
            LossRecord: Averaged record retaining the shared label and node identifier.

        """
        # Check if passed list instead of separate args
        if (len(records) == 1) and isinstance(records[0], list):
            records = records[0]

        # Enforce float casting
        lrs_to_avg = [lr.to_float() for lr in records]

        # Get sum of all records
        lr_total = LossRecord.merge(*lrs_to_avg)

        # Get number of non-null trainable and aux records
        counts = {
            "trainable": len([lr for lr in lrs_to_avg if lr.trainable is not None]),
            "auxiliary": len([lr for lr in lrs_to_avg if lr.auxiliary is not None]),
        }
        avg_train = (
            None
            if lr_total.trainable is None
            else (lr_total.trainable / counts["trainable"])
        )
        avg_aux = (
            None
            if lr_total.auxiliary is None
            else (lr_total.auxiliary / counts["auxiliary"])
        )

        # Return averaged
        return LossRecord(
            label=lr_total.label,
            node_id=lr_total.node_id,
            trainable=avg_train,
            auxiliary=avg_aux,
        )


class LossCollection(AxisSeries[LossRecord]):
    """
    Axis-aware collection of :class:`LossRecord` instances keyed by node and label.

    Description:
        Axis `"node"` always maps to the unique `node_id` internally but can be
        accessed using node instances, node IDs, or labels in queries using
        :class:`AxisSeries` helpers.

    """

    supported_reduction_methods: ClassVar[set[str]] = {
        "mean",
        "sum",
        "first",
        "last",
    }

    def __init__(self, records: list[LossRecord]) -> None:
        """
        Initialize the collection and merge duplicate keys.

        Description:
            Records with identical `(node, label)` pairs are merged via
            :meth:`LossRecord.merge`.

        Args:
            records (list[LossRecord]): Records to index by `(node, label)`.

        Returns:
            None: This initializer does not return a value.

        """
        self._records = records

        # Group records by (node, label) for AxisSeries storage
        grouped: dict[tuple[str, ...], list[LossRecord]] = {}
        for rec in records:
            grouped.setdefault((rec.node_id, rec.label), []).append(rec)

        # Merge all values in `grouped`
        combined = {k: LossRecord.merge(*v) for k, v in grouped.items()}

        # Init AxisSeries
        super().__init__(axes=("node", "label"), _data=combined)

    def _interpret_coords(
        self,
        coords: dict[str, Any],
    ) -> dict[str, Hashable]:
        """
        Map Experiment node identifiers to canonical node IDs.

        Args:
            coords (dict[str, Any]):
                Coordinate dictionary potentially containing `node` keys.

        Returns:
            dict[str, Hashable]:
                Coordinates with any node entries replaced by node IDs.

        """
        if "node" in coords:
            # Get actual node from given values, then retrieve node_id
            node = coords["node"]
            if not isinstance(node, ExperimentNode):
                exp_ctx = ExperimentContext.get_active()
                node = exp_ctx.get_node(val=coords["node"])
            # Replace coord value with node_id
            coords["node"] = node.node_id

        return coords

    # ================================================
    # Properties
    # ================================================
    @property
    def loss_labels(self) -> list[str]:
        """
        List unique loss labels in the collection.

        Returns:
            list[str]: Sorted label values stored in the label axis.

        """
        return self.axis_values(axis="label")

    @property
    def node_ids(self) -> list[str]:
        """
        List unique node IDs in the collection.

        Returns:
            list[str]: Node IDs represented in the node axis.

        """
        return self.axis_values(axis="node")

    @property
    def nodes(self) -> list[ExperimentNode]:
        """
        List :class:`ExperimentNode` instances associated with the stored node IDs.

        Returns:
            list[ExperimentNode]:
                Resolved nodes retrieved from the active experiment context.

        """
        exp_ctx = ExperimentContext.get_active()
        return [exp_ctx.get_node(node_id=x) for x in self.node_ids]

    # ================================================
    # Data Casting
    # ================================================
    def to_float(self) -> LossCollection:
        """
        Cast all underlying loss records to floats.

        Returns:
            LossCollection: New collection with float-based records.

        """
        lrs = [lr.to_float() for lr in self.values()]
        return LossCollection(records=lrs)

    # ================================================
    # Accessors / Querying
    # ================================================
    @property
    def trainable(self) -> float:
        """
        Return the total trainable loss summed across records.

        Description:
            Call :meth:`AxisSeries.where` to filter the collection before
            accessing this property.

        Returns:
            float: Sum of non-null `trainable` values.

        """
        vals = [lr.trainable for lr in self.values() if lr.trainable is not None]
        return sum(vals) if vals else 0.0

    @property
    def auxiliary(self) -> float:
        """
        Return the total auxiliary loss summed across records.

        Description:
            Call :meth:`AxisSeries.where` to filter the collection before
            accessing this property.

        Returns:
            float: Sum of non-null `auxiliary` values.

        """
        vals = [lr.auxiliary for lr in self.values() if lr.auxiliary is not None]
        return sum(vals) if vals else 0.0

    # ================================================
    # Representation
    # ================================================
    def __repr__(self):
        return f"LossCollection(keyed_by={self.axes}, len={len(self)})"
