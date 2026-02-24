"""Absolute-value scaling transformers."""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class Absolute(BaseEstimator, TransformerMixin):
    """
    Apply per-element absolute value while tracking original signs.

    Attributes:
        _mask (np.ndarray | None): Sign mask captured during :meth:`transform` for inversion.

    """

    def __init__(self):
        """Initialize the absolute-value transformer."""
        super().__init__()
        self._mask = None

    def fit(self, X, y=None):
        """
        No-op fit for compatibility with scikit-learn estimators.

        Args:
            X (np.ndarray): Training data with shape `(n_samples, n_features)`.
            y (np.ndarray | None): Ignored target array for API compatibility.

        Returns:
            Absolute: Returns `self`.

        """
        return self

    def transform(self, X):
        """
        Apply absolute value while recording per-element signs.

        Args:
            X (np.ndarray): Input array with shape `(n_samples, n_features)`.

        Returns:
            np.ndarray: Absolute-valued input.

        Raises:
            ValueError: If `X` is not two-dimensional.

        """
        if X.ndim != 2:
            msg = f"Expected 2D array (n_samples, n_features), got shape {X.shape}"
            raise ValueError(msg)
        X = np.asarray(X)
        self._mask = np.sign(X)
        return X * self._mask

    def inverse_transform(self, X):
        """
        Restore the original signs captured during :meth:`transform`.

        Args:
            X (np.ndarray): Absolute-valued array with shape `(n_samples, n_features)`.

        Returns:
            np.ndarray: Array with original signs restored.

        Raises:
            ValueError: If `X` is not two-dimensional.

        """
        if X.ndim != 2:
            msg = f"Expected 2D array (n_samples, n_features), got shape {X.shape}"
            raise ValueError(msg)
        X = np.asarray(X)
        return X * self._mask
