"""Sign-flipping scalers."""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class Negate(BaseEstimator, TransformerMixin):
    """Multiply inputs by -1 while supporting inversion."""

    def fit(self, X, y=None):
        """
        No-op fit for compatibility with scikit-learn estimators.

        Args:
            X (np.ndarray): Training data with shape `(n_samples, n_features)`.
            y (np.ndarray | None): Ignored target array for API compatibility.

        Returns:
            Negate: Returns `self`.

        """
        return self

    def transform(self, X):
        """
        Multiply the input array by -1.

        Args:
            X (np.ndarray): Input array with shape `(n_samples, n_features)`.

        Returns:
            np.ndarray: Negated array.

        Raises:
            ValueError: If `X` is not two-dimensional.

        """
        if X.ndim != 2:
            msg = f"Expected 2D array (n_samples, n_features), got shape {X.shape}"
            raise ValueError(msg)
        X = np.asarray(X)

        return -X

    def inverse_transform(self, X):
        """
        Restore the original sign by negating the input.

        Args:
            X (np.ndarray): Negated array with shape `(n_samples, n_features)`.

        Returns:
            np.ndarray: Array with original sign restored.

        Raises:
            ValueError: If `X` is not two-dimensional.

        """
        if X.ndim != 2:
            msg = f"Expected 2D array (n_samples, n_features), got shape {X.shape}"
            raise ValueError(msg)
        X = np.asarray(X)
        return -X
