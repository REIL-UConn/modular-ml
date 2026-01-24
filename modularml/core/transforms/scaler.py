from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from modularml.core.io.protocols import Configurable, Stateful
from modularml.utils.data.comparators import deep_equal
from modularml.utils.io.cloning import clone_via_serialization

if TYPE_CHECKING:
    import numpy as np


class Scaler(Configurable, Stateful):
    """
    Wrapper for feature scaling and transformation operations.

    Description:
        Provides a standardized interface for initializing, fitting, transforming,
        and serializing feature scaling objects.
    """

    def __init__(
        self,
        scaler: str | Any,
        scaler_kwargs: dict[str, Any] | None = None,
    ):
        """
        Initialize a ModularML Scaler wrapper.

        Args:
            scaler (str | Any):
                Name of a registered scaler (preferred), a scaler class, or an instance.
            scaler_kwargs (dict[str, Any] | None):
                Keyword arguments for constructing the scaler.

        """
        # Ensure all items in registry are imported
        from modularml.scalers import scaler_registry

        # Case 1: scaler given by name
        if isinstance(scaler, str):
            if scaler not in scaler_registry:
                msg = f"Scaler '{scaler}' not recognized. Run `Scaler.get_supported_scalers()` to see supported scalers."
                raise ValueError(msg)
            self.scaler_name = scaler
            self.scaler_kwargs = scaler_kwargs or {}
            self._scaler = scaler_registry[scaler](**self.scaler_kwargs)

        # Case 2: scaler given as class
        elif isinstance(scaler, type):
            cls_name = scaler.__name__
            self.scaler_name = scaler_registry.get_original_key(cls_name) or cls_name
            self.scaler_kwargs = scaler_kwargs or {}
            self._scaler = scaler(**self.scaler_kwargs)

        # Case 3: scaler given as instance
        else:
            cls_name = scaler.__class__.__name__
            self.scaler_name = scaler_registry.get_original_key(cls_name) or cls_name
            self.scaler_kwargs = scaler_kwargs or getattr(scaler, "get_params", dict)()
            self._scaler = scaler

        self._is_fit = False

        # Validate scaler
        self._validate_scaler()

    def __eq__(self, other: Scaler):
        if not isinstance(other, Scaler):
            msg = f"Cannot compare equality between Scaler and {type(other)}"
            raise TypeError(msg)

        # Compare configs
        if not deep_equal(self.get_config(), other.get_config()):
            return False

        # Compare states
        return deep_equal(self.get_state(), other.get_state())

    __hash__ = None

    # ================================================
    # Configurable
    # ================================================
    def get_config(self) -> dict[str, Any]:
        """
        Return configuration required to reconstruct this Scaler.

        Returns:
            dict[str, Any]: Scaler configuration.

        """
        # If internal scaler is a Scaler instance, unwrap it
        if isinstance(self._scaler, Scaler):
            return self._scaler.get_config()

        return {
            "scaler_cls": type(self._scaler),
            "scaler_kwargs": self.scaler_kwargs,
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Scaler:
        """
        Construct a Scaler from configuration.

        Args:
            config (dict[str, Any]): Scaler configuration.

        Returns:
            Scaler: Unfitted Scaler instance.

        """
        if config["scaler_cls"] is cls:
            msg = "Invalid Scaler config: `scaler_cls` cannot be Scaler itself. "
            raise RuntimeError(msg)

        return cls(
            scaler=config["scaler_cls"],
            scaler_kwargs=config.get("scaler_kwargs"),
        )

    # ================================================
    # Stateful
    # ================================================
    def get_state(self) -> dict[str, Any]:
        """
        Return learned state of the scaler.

        Returns:
            dict[str, Any]: Learned scaler state.

        """
        state: dict[str, Any] = {"is_fit": self._is_fit}

        if self._is_fit:
            state["learned"] = {
                k: v for k, v in self._scaler.__dict__.items() if k.endswith("_")
            }

        return state

    def set_state(self, state: dict[str, Any]) -> None:
        """
        Restore learned state of the scaler.

        Args:
            state (dict[str, Any]):
                State produced by get_state().

        """
        if not state.get("is_fit", False):
            self._is_fit = False
            return

        for attr, val in state.get("learned", {}).items():
            setattr(self._scaler, attr, val)

        self._is_fit = True

    # ================================================
    # Helpers
    # ================================================
    def _validate_scaler(self):
        if not hasattr(self._scaler, "fit"):
            raise AttributeError(
                "Underlying scaler instance does not have a `fit()` method.",
            )
        if not hasattr(self._scaler, "transform"):
            raise AttributeError(
                "Underlying scaler instance does not have a `transform()` method.",
            )

    @classmethod
    def get_supported_scalers(cls) -> dict[str, Any]:
        """
        Return the registry of supported scalers.

        Returns:
            dict[str, Any]:
                Mapping of registered scaler names to their corresponding classes.

        """
        # Ensure all scalers are registered
        from modularml.scalers import scaler_registry

        return scaler_registry

    def clone_unfitted(self) -> Scaler:
        """Create a fresh, unfitted Scaler with the same config."""
        clone = clone_via_serialization(self)
        clone._is_fit = False
        return clone

    # ================================================
    # Core logic
    # ================================================
    def fit(self, data: np.ndarray):
        """
        Fit the scaler to input data.

        Args:
            data (np.ndarray):
                Input data of shape `(n_samples, n_features)` used to compute \
                the transformation parameters.

        """
        self._scaler.fit(data)
        self._is_fit = True

    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Apply the fitted transformation to new data.

        Args:
            data (np.ndarray):
                Input data to transform. Must have the same feature layout \
                as the data used during fitting.

        Returns:
            np.ndarray:
                Transformed data array.

        Raises:
            RuntimeError:
                If the scaler has not been fit before calling this method.

        """
        if not self._is_fit:
            raise RuntimeError("Scaler has not been fit yet.")
        return self._scaler.transform(data)

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Fit the scaler and transform the data in a single step.

        Args:
            data (np.ndarray):
                Input data of shape `(n_samples, n_features)`.

        Returns:
            np.ndarray:
                Transformed data after fitting the scaler.

        """
        if hasattr(self._scaler, "fit_transform"):
            out = self._scaler.fit_transform(data)
        else:
            self._scaler.fit(data)
            out = self._scaler.transform(data)
        self._is_fit = True
        return out

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Reverse the transformation applied by the scaler, if supported.

        Args:
            data (np.ndarray):
                Transformed data to invert.

        Returns:
            np.ndarray:
                Original-scale data.

        Raises:
            NotImplementedError:
                If the underlying scaler does not implement `inverse_transform`.

        """
        if not self._is_fit:
            raise RuntimeError("Scaler has not been fit yet.")
        if not hasattr(self._scaler, "inverse_transform"):
            msg = f"{self.scaler_name} does not support inverse_transform."
            raise NotImplementedError(msg)
        return self._scaler.inverse_transform(data)

    # ================================================
    # Serialization
    # ================================================
    def save(self, filepath: Path, *, overwrite: bool = False) -> Path:
        """
        Serializes this Scaler to the specified filepath.

        Args:
            filepath (Path):
                File location to save to. Note that the suffix may be overwritten
                to enforce the ModularML file extension schema.
            overwrite (bool, optional):
                Whether to overwrite any existing file at the save location.
                Defaults to False.

        Returns:
            Path: The actual filepath to write the Scaler is saved.

        """
        from modularml.core.io.serialization_policy import SerializationPolicy
        from modularml.core.io.serializer import serializer

        return serializer.save(
            self,
            filepath,
            policy=SerializationPolicy.BUILTIN,
            overwrite=overwrite,
        )

    @classmethod
    def load(cls, filepath: Path, *, allow_packaged_code: bool = False) -> Scaler:
        """
        Load a Scaler from file.

        Args:
            filepath (Path):
                File location of a previously saved Scaler.
            allow_packaged_code : bool
                Whether bundled code execution is allowed.

        Returns:
            Scaler: The reloaded scaler.

        """
        from modularml.core.io.serializer import _enforce_file_suffix, serializer

        # Append proper sufficx only if no suffix is given
        if Path(filepath).suffix == "":
            filepath = _enforce_file_suffix(path=filepath, cls=cls)

        return serializer.load(filepath, allow_packaged_code=allow_packaged_code)
