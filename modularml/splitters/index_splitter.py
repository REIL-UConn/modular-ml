"""Explicit index-based sample partitioning."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from modularml.core.splitting.base_splitter import BaseSplitter
from modularml.utils.data.data_format import DataFormat

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from modularml.core.data.featureset_view import FeatureSetView

_VALID_BY = ("position", "sample_uuid")


class IndexSplitter(BaseSplitter):
    """
    Split a :class:`FeatureSetView` into subsets given by explicit samples.

    Description:
        Each subset is listed explicitly, either by row position within the
        view being split (``by="position"``) or by sample UUID
        (``by="sample_uuid"``). This covers partitions computed outside
        ModularML, such as pre-registered cross-validation folds, folds from
        scikit-learn splitters (e.g. ``StratifiedGroupKFold``), or bootstrap
        and bagging subsets.

        The listed order is kept, so a downstream phase sees the samples in
        exactly that order, and a sample may be listed more than once (e.g. a
        bootstrap sample drawn with replacement).

        The lists are stored in the splitter config, so the split is recorded,
        serializable and replayed like any other splitter. UUIDs survive
        saving and reloading a FeatureSet; positions refer to the row order of
        the view the splitter is applied to.

    Attributes:
        subsets (dict[str, list[int] | list[str]]):
            Subset labels mapped to row positions or sample UUIDs.
        by (str):
            ``"position"`` or ``"sample_uuid"``.

    """

    def __init__(
        self,
        subsets: Mapping[str, Sequence[int] | Sequence[str]],
        *,
        by: str = "position",
    ):
        """
        Initialize the index-based splitter.

        Args:
            subsets (Mapping[str, Sequence[int] | Sequence[str]]):
                Subset labels mapped to the samples in that subset, in the
                order they should appear.
            by (str, optional):
                How samples are identified: ``"position"`` (row positions
                relative to the view being split) or ``"sample_uuid"``.
                Defaults to ``"position"``.

        Raises:
            ValueError: If `by` is not a supported mode, or a position is not
                an integer.

        """
        if by not in _VALID_BY:
            msg = f"`by` must be one of {_VALID_BY}. Received: {by!r}"
            raise ValueError(msg)
        self.by = by
        self.subsets: dict[str, list] = {}
        for label, samples in subsets.items():
            arr = np.asarray(samples)
            if arr.ndim != 1:
                msg = f"Subset '{label}' must be a 1-dimensional sequence."
                raise ValueError(msg)
            if by == "position":
                if arr.size and arr.dtype.kind not in "iu":
                    msg = f"Subset '{label}' must list integer positions when by='position'."
                    raise ValueError(msg)
                self.subsets[str(label)] = arr.astype(np.int64).tolist()
            else:
                self.subsets[str(label)] = arr.astype(str).tolist()

    def split(
        self,
        view: FeatureSetView,
        *,
        return_views: bool = True,
    ) -> Mapping[str, FeatureSetView] | Mapping[str, Sequence[int]]:
        """
        Split a :class:`FeatureSetView` into the configured subsets.

        Args:
            view (FeatureSetView):
                Input view to partition.
            return_views (bool):
                If True, return :class:`FeatureSetView` objects; otherwise
                return relative index arrays.

        Returns:
            Mapping[str, FeatureSetView] | Mapping[str, Sequence[int]]:
                Subset label mapped to either views or relative indices.

        Raises:
            KeyError: If a sample UUID is not part of `view`.
            IndexError: If a position is outside the view.

        """
        if self.by == "position":
            split_indices = {
                label: np.asarray(pos, dtype=np.int64)
                for label, pos in self.subsets.items()
            }
        else:
            view_uuids = (
                np.asarray(
                    view.get_sample_uuids(fmt=DataFormat.NUMPY),
                )
                .reshape(-1)
                .astype(str)
            )
            order = np.argsort(view_uuids, kind="stable")
            sorted_uuids = view_uuids[order]
            split_indices = {}
            for label, uuids in self.subsets.items():
                req = np.asarray(uuids, dtype=str)
                loc = np.searchsorted(sorted_uuids, req)
                loc_c = np.minimum(loc, max(len(sorted_uuids) - 1, 0))
                found = (loc < len(sorted_uuids)) & (sorted_uuids[loc_c] == req)
                if not np.all(found):
                    missing = req[~found][:5].tolist()
                    msg = (
                        f"{int((~found).sum())} sample UUID(s) of subset '{label}' "
                        f"are not in the view, e.g. {missing}."
                    )
                    raise KeyError(msg)
                split_indices[label] = order[loc_c].astype(np.int64)

        return self._return_splits(
            view=view,
            split_indices=split_indices,
            return_views=return_views,
        )

    # ================================================
    # Configurable
    # ================================================
    def get_config(self) -> dict[str, Any]:
        """
        Return configuration required to reconstruct this splitter.

        Returns:
            dict[str, Any]: Serializable splitter configuration.

        """
        return {
            "splitter_name": "IndexSplitter",
            "by": self.by,
            "subsets": {k: list(v) for k, v in self.subsets.items()},
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> IndexSplitter:
        """
        Construct a splitter from configuration.

        Args:
            config (dict[str, Any]): Serialized splitter configuration.

        Returns:
            IndexSplitter: Unbound splitter instance.

        """
        return cls(subsets=config["subsets"], by=config.get("by", "position"))
