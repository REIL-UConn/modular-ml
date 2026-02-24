"""Results container for fit phases."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from modularml.core.experiment.results.phase_results import PhaseResults

if TYPE_CHECKING:
    from modularml.core.data.batch import Batch
    from modularml.core.data.featureset_view import FeatureSetView
    from modularml.core.references.execution_reference import TensorLike
    from modularml.core.topology.graph_node import GraphNode
    from modularml.utils.data.data_format import DataFormat


@dataclass
class FitResults(PhaseResults):
    """
    Results container for a single FitPhase execution.

    Description:
        FitResults wraps the outputs of a FitPhase, which executes a single
        pass over the complete dataset (epoch=0, batch=0). This class provides
        convenience methods for:

        - Loss aggregation from the fit pass
        - Access to fitted model outputs

    Attributes:
        label (str): Phase label inherited from :class:`PhaseResults`.

    """

    # ================================================
    # Representation
    # ================================================
    def __repr__(self):
        return f"FitResults(label='{self.label}')"

    # ================================================
    # Execution Data & Loss Querying
    # ================================================
    def stacked_tensors(
        self,
        node: str | GraphNode,
        domain: Literal["outputs", "targets", "tags", "sample_uuids"],
        *,
        role: str = "default",
        fmt: DataFormat | None = None,
        unscale: bool = False,
    ) -> TensorLike:
        """
        Retrieve tensors for a node, concatenated across all batches.

        Description:
            Since a :class:`FitPhase` executes all data under a single batch,
            this method simply returns the specified domain of the single
            recorded execution context.

        Args:
            node (str | GraphNode):
                The node to retrieve tensors for. Can be the node instance,
                its ID, or its label.
            domain (Literal["outputs", "targets", "tags", "sample_uuids"]):
                The domain of data to return:
                - outputs: the tensors produced by the node forward pass
                - targets: the expected output tensors (only for tail nodes)
                - tags: any tracked tags during the node's forward pass
                - sample_uuids: the sample identifiers
            role (str, optional):
                If multi-role data, specifies which role to return.
                Defaults to "default".
            fmt (DataFormat | None, optional):
                Format to cast returned tensors to. If None, uses as-produced
                format. Defaults to None.
            unscale (bool, optional):
                Whether to inverse any applied scalers. Only valid for tail
                nodes with domain in ["outputs", "targets"]. Defaults to False.

        Returns:
            TensorLike:
                A single tensor containing concatenated data from all batches.

        Example:
            Getting concatenated tensors across all stacked batches:

            >>> # Get all predictions stacked
            >>> predictions = fit_results.stacked_tensors(  # doctest: +SKIP
            ...     node="output_node",
            ...     domain="outputs",
            ... )
            >>> # Get targets, unscaled, as numpy
            >>> targets = fit_results.stacked_tensors(  # doctest: +SKIP
            ...     node="output_node",
            ...     domain="targets",
            ...     fmt="np",
            ...     unscale=True,
            ... )

        """
        tensor_series = self.tensors(
            node=node,
            domain=domain,
            role=role,
            fmt=fmt,
            unscale=unscale,
        )
        return tensor_series.one()

    def stacked_batches(
        self,
        node: str | GraphNode,
        *,
        fmt: DataFormat | None = None,
    ) -> Batch:
        """
        Retrieve all batches for a node, concatenated into a single Batch.

        Description:
            Since a :class:`FitPhase` executes all data under a single batch,
            this method simply returns the single executed batch.

        Args:
            node (str | GraphNode):
                The node to retrieve batches for.
            fmt (DataFormat | None, optional):
                Format to cast tensor data to. Defaults to None.

        Returns:
            Batch:
                A single Batch containing concatenated data from all batches.

        Example:
            Accessing stacked Batch objects across all execution batches

            >>> batch = fit_results.stacked_batches(  # doctest: +SKIP
            ...     node="output_node"
            ... )
            >>> print(f"Total samples: {batch.batch_size}")  # doctest: +SKIP
            >>> print(f"Outputs shape: {batch.outputs.shape}")  # doctest: +SKIP

        """
        batch = self.batches(node=node).one()
        if fmt is not None:
            return batch.to_format(fmt=fmt)
        return batch

    def aggregated_losses(
        self,
        node: str | GraphNode,
        *,
        reducer: Literal["mean", "sum"] = "mean",
    ) -> dict[str, float]:
        """
        Retrieve losses from the fit phase.

        Args:
            node (str | GraphNode):
                The node to filter losses to.
            reducer (Literal['mean', 'sum']):
                How losses should be aggregated. Defaults to "mean".

        Returns:
            dict[str, float]:
                Losses keyed by the AppliedLoss label.

        """
        n_losses = self.losses(node=node)

        # Only one epoch and one batch in FitPhase
        if n_losses.shape.get("epoch", 1) > 1:
            n_losses = n_losses.collapse(axis="epoch", reducer="first")
        if n_losses.shape.get("batch", 1) > 1:
            n_losses = n_losses.collapse(axis="batch", reducer=reducer)

        if len(n_losses.axes) != 1 or n_losses.axes[0] != "label":
            msg = (
                "Failed to collapse losses. Expected only a remaining axis of "
                f"'label'. Got: {n_losses.axes}."
            )
            raise RuntimeError(msg)

        return {k: lr.auxiliary for k, lr in n_losses.items()}

    # ================================================
    # Source Data Access
    # ================================================
    def source_views(
        self,
        node: str | GraphNode,
        *,
        role: str = "default",
    ) -> dict[str, FeatureSetView]:
        """
        Get the source FeatureSetViews that contributed data to the given node.

        Description:
            Traces the node back to its upstream FeatureSets, collects all
            unique sample UUIDs from execution results, and returns a view
            of each upstream FeatureSet filtered to only the samples used.

            Note that the returned views contain only unique sample UUIDs used
            in generating these phase results. They are not a 1-to-1 mapping
            of result sample to source sample. Use `tensors()` to get exact
            execution data.

        Args:
            node (str | GraphNode):
                The node to trace upstream from. Can be the node instance,
                its ID, or its label.
            role (str, optional):
                Restrict to samples from this role only. Defaults to "default".
            batch (int | None, optional):
                Restrict to samples from this batch only.

        Returns:
            dict[str, FeatureSetView]:
                A mapping of FeatureSet label to FeatureSetView containing
                only the samples used during execution.

        """
        return super().source_views(
            node=node,
            role=role,
            epoch=None,
            batch=None,
        )

    def source_view(
        self,
        node: str | GraphNode,
        *,
        role: str = "default",
    ) -> FeatureSetView:
        """
        Get the single source FeatureSetView for the given node.

        Description:
            Convenience method for the common case where a node has exactly
            one upstream FeatureSet. Raises `ValueError` if multiple
            upstream FeatureSets exist.

            Note that the returned views contain only unique sample UUIDs used
            in generating these phase results. They are not a 1-to-1 mapping
            of result sample to source sample. Use `tensors()` to get exact
            execution data.

        Args:
            node (str | GraphNode):
                The node to trace upstream from.
            role (str, optional):
                Restrict to samples from this role only. Defaults to "default".
            batch (int | None, optional):
                Restrict to samples from this batch only.

        Returns:
            FeatureSetView:
                A view of the single upstream FeatureSet filtered to only
                the samples used during execution.

        Raises:
            ValueError:
                If the node has multiple upstream FeatureSets.

        """
        views = self.source_views(node=node, role=role)
        if len(views) != 1:
            msg = (
                f"Node has {len(views)} upstream FeatureSets: "
                f"{list(views.keys())}. Use source_views() instead."
            )
            raise ValueError(msg)
        return next(iter(views.values()))
