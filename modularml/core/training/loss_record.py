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
    A container for storing the result of computing a loss value.

    Attributes:
        label (str):
            Identifier for the loss (e.g., "mse_loss", "triplet_margin").
        node_id (str | None):
            The node ID that produced the data on which the loss was
            computed.
        trainable (Any | None):
            Raw computed loss value to be used for backpropogration.
            Typically a backend-specific tensor or scalar.
        auxiliary (Any | None):
            Raw computed loss value that should not be used for backpropogration.
            Typically a backend-specific tensor or scalar.

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
        """Sum of trainable and auxiliary values."""
        return sum([x for x in [self.trainable, self.auxiliary] if x is not None])

    # ================================================
    # Data Formatting and Casting
    # ================================================
    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "node_id": self.node_id,
            "trainable": self.trainable,
            "auxiliary": self.auxiliary,
        }

    def to_float(self) -> LossRecord:
        """
        Construct a copy of this loss record with values as floats.

        Backend-specific values are cast to float. Note that this
        will break any auto-grad based loss tracking (i.e., PyTorch
        backpropogation will not be possible on the returned record).

        Returns:
            LossRecord: A new LossRecord instance will float values.

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
        """Merge all records into a new instance."""
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
        """Merge this record with others."""
        return type(self).merge(self, *others)

    def __add__(self, other: LossRecord) -> LossRecord:
        return type(self).merge(self, other)

    def __radd__(self, other: LossRecord | int) -> LossRecord:
        if other == 0:
            return self
        return type(self).merge(other, self)

    @classmethod
    def mean(cls, *records: LossRecord) -> LossRecord:
        """
        Compute the mean of all records.

        Description:
            Trainable and auxiliary loss values are only averaged over
            the number of records with non-None trainable/auxiliary
            attributes. For example, if there are five records, but only
            three have non-null `trainable` values, the returned trainable
            value will have been divided by 3, not 5.

            Values are cast to Python floats before averaging.

        """
        # Enforce float casting
        lrs_to_avg = [lr.to_float() for lr in records]

        # Get number of non-null trainable and aux records
        counts = {
            "trainable": len([lr for lr in lrs_to_avg if lr.trainable is not None]),
            "auxiliary": len([lr for lr in lrs_to_avg if lr.auxiliary is not None]),
        }

        # Get sum of all records
        lr_total = LossRecord.merge(*lrs_to_avg)

        # Return averaged
        return LossRecord(
            label=lr_total.label,
            node_id=lr_total.node_id,
            trainable=lr_total.trainable / counts["trainable"],
            auxiliary=lr_total.auxiliary / counts["auxiliary"],
        )


class LossCollection(AxisSeries[LossRecord]):
    """
    Loss records keyed by (node, label).

    Note that axis `"node"` internally uses the unique node_id value, but can be
    accessed externally via node_id, the node label or the actual node instance.
    """

    supported_reduction_methods: ClassVar[set[str]] = {
        "mean",
        "sum",
        "first",
        "last",
    }

    def __init__(self, records: list[LossRecord]) -> None:
        """
        Initialize a LossCollection from some list of LossRecords.

        All records with the same node and label will be merged using
        `LossRecord.merge`.
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
        Hook mapping potential Experiment node instance/label to its node ID.

        Args:
            coords (dict[str, Any]):
                Maps any coords values for `key="node"` to node ID.

        Returns:
            dict[str, Hashable]: Coords with "node" values updated to node ID..

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
        """List of unique loss labels."""
        return self.axis_values(axis="label")

    @property
    def node_ids(self) -> list[str]:
        """List of unique node IDs."""
        return self.axis_values(axis="node")

    @property
    def nodes(self) -> list[ExperimentNode]:
        """List of unique node instances associated with these losses."""
        exp_ctx = ExperimentContext.get_active()
        return [exp_ctx.get_node(node_id=x) for x in self.node_ids]

    # ================================================
    # Data Casting
    # ================================================
    def to_float(self) -> LossCollection:
        """Casts all underlying loss record values to floats."""
        lrs = [lr.to_float() for lr in self.values()]
        return LossCollection(records=lrs)

    # ================================================
    # Accessors / Querying
    # ================================================
    @property
    def trainable(self) -> float:
        """
        Retrieves the total trainable loss value of all records in this collection.

        To get the trainable loss for a specific node or label, use `get_trainable()`.
        """
        vals = [lr.trainable for lr in self.values() if lr.trainable is not None]
        return sum(vals) if vals else 0.0

    def get_trainable(
        self,
        node: str | ExperimentNode | None = None,
        label: str | None = None,
    ) -> float:
        """
        Retrieves the total trainable loss for the specified entry or entries.

        Description:
            Filtering is first performed with the given `node` and `label` values.
            The `trainable` attribute of all resulting entries is then summed and
            returned.

        Args:
            node (str | ExperimentNode | None, optional):
                The node to filter records to. Can be the node ID, its label, or
                the actual node instance. If None, entries across all nodes are
                summed. Defaults to None.
            label (str | None, optional):
                The loss label to filter records to. If None, entries across all
                loss labels are summed. Defaults to None.

        Returns:
            float: The summed trainable loss value over the filtered entries.

        """
        vals = [
            lr.trainable
            for lr in self.select(node=node, label=label)
            if lr.trainable is not None
        ]
        return sum(vals) if vals else 0.0

    @property
    def auxiliary(self) -> float:
        """
        Retrieves the total auxiliary loss value of all records in this collection.

        To get the auxiliary loss for a specific node or label, use `get_auxiliary()`.
        """
        vals = [lr.auxiliary for lr in self.values() if lr.auxiliary is not None]
        return sum(vals) if vals else 0.0

    def get_auxiliary(
        self,
        node: str | ExperimentNode | None = None,
        label: str | None = None,
    ) -> float:
        """
        Retrieves the total auxiliary loss for the specified entry or entries.

        Description:
            Filtering is first performed with the given `node` and `label` values.
            The `auxiliary` attribute of all resulting entries is then summed and
            returned.

        Args:
            node (str | ExperimentNode | None, optional):
                The node to filter records to. Can be the node ID, its label, or
                the actual node instance. If None, entries across all nodes are
                summed. Defaults to None.
            label (str | None, optional):
                The loss label to filter records to. If None, entries across all
                loss labels are summed. Defaults to None.

        Returns:
            float: The summed auxiliary loss value over the filtered entries.

        """
        vals = [
            lr.auxiliary
            for lr in self.select(node=node, label=label)
            if lr.auxiliary is not None
        ]
        return sum(vals) if vals else 0.0

    # ================================================
    # Representation
    # ================================================
    def __repr__(self):
        return f"LossCollection(keyed_by={self.axes}, len={len(self)})"
