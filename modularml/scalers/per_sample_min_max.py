"""Per-sample min/max scaling transformers."""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class PerSampleMinMaxScaler(BaseEstimator, TransformerMixin):
    """
    Scale each sample independently to a target range.

    Description:
        For every sample `x`, the transformer computes
        `(x - min) / (max - min) * (feature_range[1] - feature_range[0]) + feature_range[0]`
        so each sample spans `feature_range`.

    Attributes:
        feature_range (tuple[float, float]): Desired `(min, max)` range after scaling.
        min_ (float): Lower bound portion of `feature_range`.
        max_ (float): Upper bound portion of `feature_range`.
        _sample_min (np.ndarray | None): Per-sample minima captured during :meth:`transform`.
        _sample_range (np.ndarray | None): Per-sample ranges captured during :meth:`transform`.

    """

    def __init__(self, feature_range=(0, 1)):
        """
        Initialize the per-sample min/max scaler.

        Args:
            feature_range (tuple[float, float]): Desired `(min, max)` range for each sample.

        """
        self.feature_range = feature_range
        self.min_, self.max_ = self.feature_range

        self._sample_min = None
        self._sample_range = None

    def get_params(self, deep=True):  # noqa: FBT002
        """
        Return estimator parameters for scikit-learn compatibility.

        Args:
            deep (bool): Ignored; included for API compatibility.

        Returns:
            dict[str, Any]: Dictionary containing `feature_range`.

        """
        params = super().get_params(deep)
        params["feature_range"] = self.feature_range
        return params

    def fit(self, X, y=None):
        """
        No-op fit for compatibility with scikit-learn estimators.

        Args:
            X (np.ndarray): Training data with shape `(n_samples, n_features)`.
            y (np.ndarray | None): Ignored target array for API compatibility.

        Returns:
            PerSampleMinMaxScaler: Returns `self`.

        """
        return self

    def transform(self, X):
        """
        Scale each sample independently into the configured range.

        Args:
            X (np.ndarray): Input array with shape `(n_samples, n_features)`.

        Returns:
            np.ndarray: Scaled array with each sample spanning `feature_range`.

        Raises:
            ValueError: If `X` is not two-dimensional.

        """
        X = np.asarray(X)

        if X.ndim != 2:
            msg = f"Expected 2D array (n_samples, n_features), got shape {X.shape}"
            raise ValueError(msg)

        sample_min = np.min(X, axis=1, keepdims=True)
        sample_max = np.max(X, axis=1, keepdims=True)
        sample_range = np.where(
            (sample_max - sample_min) == 0,
            1,
            sample_max - sample_min,
        )

        self._sample_min = sample_min
        self._sample_range = sample_range

        scale = self.max_ - self.min_
        return (X - sample_min) / sample_range * scale + self.min_

    def inverse_transform(self, X):
        """
        Reconstruct the original samples using stored min/max statistics.

        Args:
            X (np.ndarray): Scaled array with shape `(n_samples, n_features)`.

        Returns:
            np.ndarray: Array in the original sample scale.

        Raises:
            ValueError: If `X` is not two-dimensional.

        """
        X = np.asarray(X)
        if X.ndim != 2:
            msg = f"Expected 2D array (n_samples, n_features), got shape {X.shape}"
            raise ValueError(msg)

        scale = self.max_ - self.min_

        return (((X - self.min_) / scale) * self._sample_range) + self._sample_min
