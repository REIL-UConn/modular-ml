"""Per-sample zero-start shifting transformer."""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class PerSampleZeroStart(BaseEstimator, TransformerMixin):
    """
    Shift each sample so its first feature becomes zero.

    Description:
        For each sample `x`, the transformer computes `x - x[0]`, storing the original
        offset for use during :meth:`inverse_transform`.

    Attributes:
        x0_ (np.ndarray | None): Column vector of per-sample offsets captured during :meth:`fit`.

    """

    def __init__(self):
        """Initialize the zero-start scaler."""
        super().__init__()
        self.x0_ = None

    def get_params(self, deep=True):  # noqa: FBT002
        """
        Return estimator parameters for scikit-learn compatibility.

        Args:
            deep (bool): Ignored; included for API compatibility.

        Returns:
            dict[str, Any]: Empty dictionary (no hyperparameters).

        """
        params = super().get_params(deep)
        return params

    def fit(self, X, y=None):
        """
        Compute per-sample offsets used later for inverse transforms.

        Args:
            X (np.ndarray): Input data with shape `(n_samples, n_features)`.
            y (np.ndarray | None): Ignored target array for API compatibility.

        Returns:
            PerSampleZeroStart: Returns `self`.

        Raises:
            ValueError: If `X` is not two-dimensional.

        """
        if X.ndim != 2:
            msg = f"Expected 2D array (n_samples, n_features), got shape {X.shape}"
            raise ValueError(msg)
        X = np.asarray(X)

        # store offsets for later inverse
        self.x0_ = X[:, [0]]
        return self

    def transform(self, X):
        """
        Shift each sample so its first value becomes zero.

        Args:
            X (np.ndarray): Input array with shape `(n_samples, n_features)`.

        Returns:
            np.ndarray: Shifted array with zero-based starts per sample.

        """
        self.fit(X)
        return X - self.x0_

    def inverse_transform(self, X):
        """
        Restore originals by adding the stored per-sample offsets.

        Args:
            X (np.ndarray): Shifted array with shape `(n_samples, n_features)`.

        Returns:
            np.ndarray: Array with original offsets restored.

        Raises:
            ValueError: If `X` is not two-dimensional.

        """
        if X.ndim != 2:
            msg = f"Expected 2D array (n_samples, n_features), got shape {X.shape}"
            raise ValueError(msg)
        X = np.asarray(X)
        return X + self.x0_
