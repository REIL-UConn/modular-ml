"""Segment-wise scaling utilities."""

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from modularml.core.transforms.scaler import Scaler


class SegmentedScaler(BaseEstimator, TransformerMixin):
    """
    Apply independent scalers to contiguous feature segments.

    Description:
        The feature dimension is partitioned using `boundaries`, and a cloned
        :class:`Scaler` is fit on each segment independently. During transformation
        the per-segment scalers are applied and concatenated back together.

    Attributes:
        boundaries (tuple[int, ...]):
            Boundary indices defining contiguous segments.
        scaler (str):
            Name of the wrapped scaler implementation.
        scaler_kwargs (dict[str, Any] | None):
            Keyword arguments used when instantiating the scaler.
        scaler_template (Scaler):
            Template used to spawn per-segment scalers.
        _segment_scalers (list[Scaler]):
            Runtime list of fitted scalers, one per segment.

    """

    def __init__(
        self,
        boundaries: tuple[int],
        scaler: Scaler | str | Any,
        scaler_kwargs: dict[str, Any] | None = None,
    ):
        """
        Initialize the segmented scaler.

        Args:
            boundaries (tuple[int]):
                Sorted boundary indices (must start at 0).
            scaler (Scaler | str | Any):
                Scaler instance, scaler name, or sklearn-compatible transformer.
            scaler_kwargs (dict[str, Any] | None):
                Optional keyword arguments when instantiating the scaler.

        Raises:
            ValueError: If boundaries do not start at 0 or are not strictly increasing.

        """
        # Validate boundaries
        if 0 not in boundaries:
            raise ValueError("Boundaries must start at 0.")
        if any(boundaries[i] >= boundaries[i + 1] for i in range(len(boundaries) - 1)):
            raise ValueError("Boundaries must be strictly increasing.")

        # Normalize scaler input
        # Accepts: scaler instance, scaler name, sklearn-like object
        if isinstance(scaler, Scaler):
            self.scaler_template = scaler
        else:
            self.scaler_template = Scaler(
                scaler=scaler,
                scaler_kwargs=scaler_kwargs,
            )

        # Store all init args exactly as name (requirement of BaseEstimator)
        self.boundaries = boundaries
        self.scaler = self.scaler_template.scaler_name
        self.scaler_kwargs = self.scaler_template.scaler_kwargs

        # Runtime-only scaler (one per defined boundary)
        self._segment_scalers: list[Scaler] = []

    def get_params(self, deep=True):  # noqa: FBT002
        """
        Return estimator parameters for scikit-learn compatibility.

        Args:
            deep (bool): Ignored; included for API compatibility.

        Returns:
            dict[str, Any]: Dictionary containing boundaries and scaler metadata.

        """
        params = super().get_params(deep)
        params["boundaries"] = self.boundaries
        params["scaler"] = self.scaler
        params["scaler_kwargs"] = self.scaler_kwargs
        return params

    def fit(self, X: np.ndarray, y: np.ndarray | None = None):
        """
        Fit an independent scaler on each feature segment.

        Args:
            X (np.ndarray):
                Training data with shape `(n_samples, n_features)`.
            y (np.ndarray | None):
                Ignored target array for API compatibility.

        Returns:
            SegmentedScaler: Returns `self`.

        Raises:
            ValueError: If the final boundary does not match the feature dimension.

        """
        self._segment_scalers.clear()

        if X.shape[1] != self.boundaries[-1]:
            msg = (
                "Last boundary does not match feature length: "
                f"{self.boundaries[-1]} != {X.shape[1]}."
            )
            raise ValueError(msg)
        for i in range(len(self.boundaries) - 1):
            start, end = self.boundaries[i], self.boundaries[i + 1]
            segment = X[:, start:end]

            scaler: Scaler = self.scaler_template.clone_unfitted()
            scaler.fit(segment)
            self._segment_scalers.append(scaler)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform each segment using its corresponding fitted scaler.

        Args:
            X (np.ndarray): Input array with shape `(n_samples, n_features)`.

        Returns:
            np.ndarray: Concatenated segment outputs.

        Raises:
            RuntimeError: If :meth:`fit` has not been called.

        """
        if not self._segment_scalers:
            raise RuntimeError("SegmentedScaler has not been fit.")

        segments = []
        for i, scaler in enumerate(self._segment_scalers):
            start, end = self.boundaries[i], self.boundaries[i + 1]
            segment = X[:, start:end]
            transformed = scaler.transform(segment)
            segments.append(transformed)

        return np.concatenate(segments, axis=1)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply the inverse transform for each segment.

        Args:
            X (np.ndarray): Transformed array with shape `(n_samples, n_features)`.

        Returns:
            np.ndarray: Array restored to the original feature scale.

        """
        segments = []
        for i, scaler in enumerate(self._segment_scalers):
            start, end = self.boundaries[i], self.boundaries[i + 1]
            segment = X[:, start:end]
            inverse = scaler.inverse_transform(segment)
            segments.append(inverse)
        return np.concatenate(segments, axis=1)
