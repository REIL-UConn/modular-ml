from __future__ import annotations

import pickle
from enum import Enum
from typing import Any

import numpy as np

from modularml.core.models.base_model import BaseModel
from modularml.utils.data.shape_utils import ensure_tuple_shape
from modularml.utils.nn.backend import Backend


class ScikitTrainingMode(Enum):
    """Training mode for scikit-learn models."""

    PARTIAL_FIT = "partial_fit"
    """Model supports incremental batch training via `partial_fit()`."""

    BATCH_FIT = "batch_fit"
    """Model requires all training data at once via `fit()`."""

    AUTO = "auto"
    """Auto-detect from model capabilities (`hasattr(model, 'partial_fit')`)."""


class ScikitModelWrapper(BaseModel):
    """
    Wraps a scikit-learn estimator into the ModularML BaseModel interface.

    Supports both batch-fit models (e.g., RandomForest, SVM) and incremental
    models (e.g., SGDRegressor, MLPRegressor) via the `training_mode` parameter.

    Args:
        model (Any):
            A scikit-learn `BaseEstimator` instance.
        training_mode (ScikitTrainingMode | str, optional):
            How the model should be trained. `"auto"` detects from
            `hasattr(model, 'partial_fit')`. Can be overridden to
            `"partial_fit"` or `"batch_fit"`.
        output_method (str, optional):
            Which method to call for predictions. `"auto"` picks
            `predict` for estimators. Can be set to `"predict_proba"`
            or `"decision_function"` for classifiers.
        partial_fit_kwargs (dict[str, Any] | None, optional):
            Extra keyword arguments passed to `partial_fit()` on every call
            (e.g., `{"classes": np.array([0, 1])}` for classifiers).

    """

    def __init__(
        self,
        model: Any,
        *,
        training_mode: ScikitTrainingMode | str = "auto",
        output_method: str = "auto",
        partial_fit_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ):
        super().__init__(
            backend=kwargs.pop("backend", None) or Backend.SCIKIT,
            model_class=type(model),
            training_mode=training_mode
            if isinstance(training_mode, str)
            else training_mode.value,
            output_method=output_method,
            partial_fit_kwargs=partial_fit_kwargs,
        )

        self.model = model
        self._training_mode_raw = (
            ScikitTrainingMode(training_mode)
            if isinstance(training_mode, str)
            else training_mode
        )
        self._output_method = output_method
        self._partial_fit_kwargs = partial_fit_kwargs or {}

        # Shape tracking
        self._input_shape: tuple[int, ...] | None = None
        self._output_shape: tuple[int, ...] | None = None

        # Track whether the sklearn model has been fitted
        self._is_fitted = False

    # ================================================
    # Properties
    # ================================================
    @property
    def input_shape(self) -> tuple[int, ...] | None:
        return self._input_shape

    @property
    def output_shape(self) -> tuple[int, ...] | None:
        return self._output_shape

    @property
    def supports_partial_fit(self) -> bool:
        """Whether the underlying model supports ``partial_fit()``."""
        return hasattr(self.model, "partial_fit") and callable(self.model.partial_fit)

    @property
    def resolved_training_mode(self) -> ScikitTrainingMode:
        """Resolved training mode (auto-detected if `AUTO`)."""
        if self._training_mode_raw == ScikitTrainingMode.AUTO:
            if self.supports_partial_fit:
                return ScikitTrainingMode.PARTIAL_FIT
            return ScikitTrainingMode.BATCH_FIT
        return self._training_mode_raw

    @property
    def is_fitted(self) -> bool:
        """Whether the sklearn model has been fitted."""
        return self._is_fitted

    # ================================================
    # Build
    # ================================================
    def build(
        self,
        input_shape: tuple[int, ...] | None = None,
        output_shape: tuple[int, ...] | None = None,
        *,
        force: bool = False,
    ):
        """
        Store shape information for graph propagation.

        Scikit-learn models do not require explicit building like neural networks.
        This method records input/output shapes so that the ModelGraph can perform
        shape inference across nodes. Actual model fitting happens in `fit()` or
        `partial_fit()`.
        """
        if self.is_built and not force:
            return

        if input_shape is not None:
            self._input_shape = ensure_tuple_shape(
                shape=input_shape,
                min_len=1,
                max_len=None,
                allow_null_shape=False,
            )

        if output_shape is not None:
            self._output_shape = ensure_tuple_shape(
                shape=output_shape,
                min_len=1,
                max_len=None,
                allow_null_shape=False,
            )

        self._built = True

    # ================================================
    # Forward Pass
    # ================================================
    def _resolve_output_method(self) -> str:
        """Resolve which prediction method to use."""
        if self._output_method != "auto":
            return self._output_method
        return "predict"

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through the scikit-learn model.

        Calls the resolved output method (`predict`, `predict_proba`, etc.)
        and ensures the output is always a 2D numpy array of shape `(n_samples, n_outputs)`.

        Args:
            x: Input features as a numpy array of shape `(n_samples, n_features)`.

        Returns:
            np.ndarray: Predictions of shape `(n_samples, n_outputs)`.

        Raises:
            RuntimeError: If the model has not been fitted yet.

        """
        if not self._is_fitted:
            msg = (
                "Scikit-learn model has not been fitted yet. "
                "Call fit() or partial_fit() before forward()."
            )
            raise RuntimeError(msg)

        method_name = self._resolve_output_method()
        method = getattr(self.model, method_name)
        out = method(x)

        # Ensure 2D output: sklearn .predict() returns 1D for single-output models
        if isinstance(out, np.ndarray) and out.ndim == 1:
            out = out.reshape(-1, 1)

        return out

    # ================================================
    # Fit Methods
    # ================================================
    def _infer_output_shape(self, x: np.ndarray) -> None:
        """Infer output shape by running a dummy prediction."""
        try:
            dummy_out = self.forward(x[:1])
            self._output_shape = tuple(dummy_out.shape[1:])
        except Exception:
            # Some models may not be ready for prediction right after fit
            # (e.g., partial_fit with insufficient data). Set to None.
            return

    def fit(self, x: np.ndarray, y: np.ndarray | None = None) -> None:
        """
        Fit the scikit-learn model on complete training data.

        Args:
            x (np.ndarray):
                Training features of shape `(n_samples, n_features)`.
            y (np.ndarray):
                Training targets. Shape depends on the estimator.

        """
        self.model.fit(x, y)
        self._is_fitted = True

        # Infer shapes after fitting
        self._input_shape = tuple(x.shape[1:])
        self._infer_output_shape(x)

    def partial_fit(self, x: np.ndarray, y: np.ndarray | None = None, **kwargs) -> None:
        """
        Incrementally fit the model on a mini-batch.

        Args:
            x (np.ndarray):
                Batch features of shape `(batch_size, n_features)`.
            y (np.ndarray):
                Batch targets.
            **kwargs: Additional arguments (e.g., `classes` for classifiers).

        Raises:
            AttributeError: If the model does not support `partial_fit()`.

        """
        if not self.supports_partial_fit:
            msg = (
                f"Model {type(self.model).__name__} does not support partial_fit(). "
                "Use FitPhase for batch-fit models instead of TrainPhase."
            )
            raise AttributeError(msg)

        # Merge stored kwargs with call-time kwargs (call-time takes priority)
        merged_kwargs = {**self._partial_fit_kwargs, **kwargs}
        self.model.partial_fit(x, y, **merged_kwargs)
        self._is_fitted = True

        # Infer shapes after first partial_fit
        if self._input_shape is None:
            self._input_shape = tuple(x.shape[1:])
        if self._output_shape is None:
            self._infer_output_shape(x)

    # ================================================
    # Weight Handling
    # ================================================
    def get_weights(self) -> dict[str, Any]:
        """
        Serialize the fitted sklearn model state via pickle.

        Returns:
            dict: Contains the pickled model bytes.

        """
        return {
            "model_pickle": pickle.dumps(self.model),
            "is_fitted": self._is_fitted,
        }

    def set_weights(self, weights: dict[str, Any]) -> None:
        """
        Restore the sklearn model from pickled state.

        Args:
            weights: Dict produced by `get_weights()`.

        """
        if "model_pickle" in weights:
            self.model = pickle.loads(weights["model_pickle"])
            self._is_fitted = weights.get("is_fitted", False)

    # ================================================
    # Configurable
    # ================================================
    def get_config(self) -> dict[str, Any]:
        cfg = super().get_config()
        cfg.update(
            {
                "model_params": self.model.get_params(),
                "training_mode": self._training_mode_raw.value,
                "output_method": self._output_method,
                "partial_fit_kwargs": self._partial_fit_kwargs,
            },
        )
        return cfg

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> ScikitModelWrapper:
        """Reconstruct a ScikitModelWrapper from configuration."""
        model_cls = config["model_class"]
        if isinstance(model_cls, str):
            from modularml.core.io.symbol_registry import symbol_registry

            model_cls = symbol_registry.resolve(model_cls)

        model_params = config.get("model_params", {})
        model = model_cls(**model_params)

        wrapper = cls(
            model=model,
            training_mode=config.get("training_mode", "auto"),
            output_method=config.get("output_method", "auto"),
            partial_fit_kwargs=config.get("partial_fit_kwargs"),
        )

        if config.get("is_built", False):
            wrapper.build(
                input_shape=config.get("input_shape"),
                output_shape=config.get("output_shape"),
                force=True,
            )

        return wrapper
