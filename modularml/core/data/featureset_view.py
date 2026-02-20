"""Immutable row and column projection of a :class:`FeatureSet`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pyarrow as pa

from modularml.core.data.sample_collection_mixin import SampleCollectionMixin
from modularml.core.data.schema_constants import (
    DOMAIN_FEATURES,
    DOMAIN_SAMPLE_UUIDS,
    DOMAIN_TAGS,
    DOMAIN_TARGETS,
)
from modularml.core.experiment.experiment_context import ExperimentContext
from modularml.core.io.protocols import Configurable
from modularml.core.splitting.split_mixin import SplitMixin
from modularml.utils.representation.summary import Summarizable

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from modularml.core.data.featureset import FeatureSet
    from modularml.core.data.sample_collection import SampleCollection


@dataclass
class FeatureSetView(SampleCollectionMixin, SplitMixin, Summarizable, Configurable):
    """
    Immutable row and column projection of a :class:`FeatureSet`.

    Intended for inspection, export, and analysis.

    Attributes:
        source (FeatureSet): The parent :class:`FeatureSet` this view projects.
        indices (NDArray[np.int64]): Row indices into the source.
        columns (list[str]): Column names included in this view.
        label (str | None): Optional label for this view.

    """

    source: FeatureSet
    indices: NDArray[np.int64]
    columns: list[str]
    label: str | None = None

    @classmethod
    def from_featureset(
        cls,
        fs: FeatureSet,
        *,
        rows: NDArray[np.int64] | None = None,
        columns: list[str] | None = None,
        label: str | None = None,
    ) -> FeatureSetView:
        """
        Create a :class:`FeatureSetView` from a :class:`FeatureSet`.

        Args:
            fs (FeatureSet): Source :class:`FeatureSet`.
            rows (NDArray[np.int64] | None): Row indices to include.
                Defaults to all rows.
            columns (list[str] | None): Column names to include.
                Defaults to all columns.
            label (str | None): Optional label for the view.

        Returns:
            FeatureSetView: A new view over the given :class:`FeatureSet`.

        """
        if rows is None:
            rows = np.arange(fs.n_samples)
        if columns is None:
            columns = fs.collection.get_all_keys(
                include_domain_prefix=True,
                include_rep_suffix=True,
            )
        return cls(
            source=fs,
            indices=np.asarray(rows, dtype=np.int64),
            columns=columns,
            label=label,
        )

    # ================================================
    # Properties & Dunders
    # ================================================
    def __repr__(self):
        return f"FeatureSetView(source='{self.source.label}', n_samples={self.n_samples}, label='{self.label}')"

    def __eq__(self, other):
        """Compare source and indices. Label is ignored."""
        if not isinstance(other, FeatureSetView):
            msg = f"Cannot compare equality between FeatureSetView and {type(other)}"
            raise TypeError(msg)

        return (self.source == other.source) and (self.indices == other.indices)

    __hash__ = None

    def __setattr__(self, name, value):
        # Overrriding __setattr__ to make all attributes frozen except label
        frozen_attrs = ["source", "indices", "columns"]

        # Check if attr is frozen and if it's already set (to allow init)
        if name in frozen_attrs and name in self.__dict__:
            msg = f"Cannot reassign frozen attribute '{name}'"
            raise AttributeError(msg)

        # Use default __setattr__ behavior
        super().__setattr__(name, value)

    @property
    def valid_indices(self) -> np.ndarray:
        """Indices >= 0 (used for data lookup)."""
        return self.indices[self.indices >= 0]

    # ================================================
    # SampleCollectionMixin
    # ================================================
    def _resolve_caller_attributes(
        self,
    ) -> tuple[SampleCollection, list[str] | None, np.ndarray | None]:
        return (
            self.source.collection,
            self.columns,
            self.indices,
        )

    # ================================================
    # Comparators
    # ================================================
    def is_disjoint_with(self, other: FeatureSetView) -> bool:
        """
        Check if this view has no overlapping samples.

        Description:
            If both views share the same source :class:`FeatureSet`, comparison
            is based on indices. If they originate from different sources,
            comparison falls back to `DOMAIN_SAMPLE_UUIDS` to ensure identity
            consistency across saved or merged datasets.

        Args:
            other (FeatureSetView): The other view to compare against.

        Returns:
            bool: True if the views share no common samples.

        """
        if not isinstance(other, FeatureSetView):
            msg = f"Comparison only valid between FeatureSetViews, not {type(other)}"
            raise TypeError(msg)

        # Same collection (fast index-based check)
        if self.source is other.source:
            return (
                len(np.intersect1d(self.indices, other.indices, assume_unique=True))
                == 0
            )

        # Otherwise compare SAMPLE_IDs
        ids_self = {
            self.source.collection.table[DOMAIN_SAMPLE_UUIDS].to_pylist()[i]
            for i in self.indices
        }
        ids_other = {
            other.source.collection.table[DOMAIN_SAMPLE_UUIDS].to_pylist()[i]
            for i in other.indices
        }
        return ids_self.isdisjoint(ids_other)

    def get_overlap_with(self, other: FeatureSetView) -> list[str]:
        """
        Get overlapping sample identifiers between two :class:`FeatureSetView` instances.

        Args:
            other (FeatureSetView): The other view to compare against.

        Returns:
            list[str]: A list of overlapping `DOMAIN_SAMPLE_UUIDS` values.

        """
        if not isinstance(other, FeatureSetView):
            msg = f"Comparison only valid between FeatureSetViews, not {type(other)}"
            raise TypeError(msg)

        # Same collection (fast index-based check)
        if self.source is other.source:
            overlap = np.intersect1d(self.indices, other.indices, assume_unique=True)
            return (
                self.source.collection.table[DOMAIN_SAMPLE_UUIDS]
                .take(pa.array(overlap))
                .to_pylist()
            )

        # Otherwise compare SAMPLE_IDs
        ids_self = {
            self.source.collection.table[DOMAIN_SAMPLE_UUIDS].to_pylist()[i]
            for i in self.indices
        }
        ids_other = {
            other.source.collection.table[DOMAIN_SAMPLE_UUIDS].to_pylist()[i]
            for i in other.indices
        }
        return list(ids_self.intersection(ids_other))

    # ================================================
    # Convenience Methods
    # ================================================
    def expand_columns(self, label: str | None = None) -> FeatureSetView:
        """
        Create a new view without column filtering.

        Description:
            A new view is created with the same indices, but without
            any filters over which columns to include. The returned
            view will include all columns of the source :class:`FeatureSet`.

        Args:
            label (str | None): Optional label for the expanded view.
                Defaults to `"{self.label}_expanded"`.

        Returns:
            FeatureSetView: A new view with all columns included.

        """
        return FeatureSetView.from_featureset(
            fs=self.source,
            rows=self.indices,
            columns=None,
            label=label or f"{self.label}_expanded",
        )

    # ================================================
    # Configurable
    # ================================================
    def get_config(self) -> dict[str, Any]:
        """
        Return configuration required to reconstruct this view.

        Returns:
            dict[str, Any]: View configuration.

        """
        return {
            "source": {
                "node_label": self.source.label,
                "node_id": self.source.node_id,
            },
            "indices": np.asarray(self.indices).tolist(),
            "columns": self.columns,
            "label": self.label,
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> FeatureSetView:
        """
        Construct a view from configuration.

        Args:
            config (dict[str, Any]): View configuration.

        Returns:
            FeatureSetView: Reconstructed view.

        """
        from modularml.core.data.featureset import FeatureSet

        if not all(x in config for x in ["source", "indices", "columns", "label"]):
            raise ValueError("Invalid config for FeatureSetView.")

        # Re-link source using ExperimentContext
        exp_ctx = ExperimentContext.get_active()
        try:
            node = exp_ctx.get_node(node_id=config["source"]["node_id"])
        except KeyError as e:
            msg = (
                f"There are no registered nodes with id: '{config['source']['node_id']}'. "
                f"Ensure FeatureSet '{config['source']['node_label']}' exists in the current ExperimentContext."
            )
            raise RuntimeError(msg) from e
        if not isinstance(node, FeatureSet):
            msg = f"Node with ID ('{config['source']['node_id']}') is not a FeatureSet. Received: {type(node)}."
            raise TypeError(msg)

        # Create view
        return FeatureSetView(
            source=node,
            indices=np.asarray(config["indices"], dtype=np.int64),
            columns=config["columns"],
            label=config["label"],
        )

    # ================================================
    # Representation
    # ================================================
    def _summary_rows(self) -> list[tuple]:
        return [
            ("label", self.label),
            ("source", self.source.label),
            ("n_samples", self.n_samples),
            (
                "columns",
                [
                    (
                        DOMAIN_FEATURES,
                        str(
                            self.get_feature_keys(
                                include_domain_prefix=False,
                                include_rep_suffix=True,
                            ),
                        ),
                    ),
                    (
                        DOMAIN_TARGETS,
                        str(
                            self.get_target_keys(
                                include_domain_prefix=False,
                                include_rep_suffix=True,
                            ),
                        ),
                    ),
                    (
                        DOMAIN_TAGS,
                        str(
                            self.get_tag_keys(
                                include_domain_prefix=False,
                                include_rep_suffix=True,
                            ),
                        ),
                    ),
                ],
            ),
        ]
