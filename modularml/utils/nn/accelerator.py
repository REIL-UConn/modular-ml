"""Hardware accelerator configuration for ModularML model training."""

from __future__ import annotations

import re
from contextlib import contextmanager
from typing import Any

from modularml.utils.environment.optional_imports import (
    check_tensorflow,
    check_torch,
    ensure_jax,
)
from modularml.utils.logging.warnings import warn


# ================================================
# Helpers / Validation
# ================================================
def _validate_device_str(device: str) -> None:
    """Raise ValueError if device string is not recognized."""
    valid_device_patterns = re.compile(
        r"^(cpu|mps|cuda|cuda:\d+|gpu|gpu:\d+)$",
    )
    if not valid_device_patterns.match(device):
        msg = (
            f"Unrecognized device string: {device!r}. "
            "Expected one of: 'cpu', 'cuda', 'cuda:N', 'mps', 'gpu', 'gpu:N'."
        )
        raise ValueError(msg)


def _parse_device_index(device: str) -> int | None:
    """Extract numeric index from a device string like 'cuda:0' or 'gpu:1'."""
    match = re.search(r":(\d+)$", device)
    return int(match.group(1)) if match else None


# ================================================
# Accelerator
# ================================================
class Accelerator:
    """
    Hardware accelerator configuration for ModularML training.

    Specifies which device (CPU, GPU, MPS, etc.) models and tensors should
    be placed on during training. Supports PyTorch and TensorFlow backends.

    Attributes:
        device (str): Device identifier string. Accepted forms:
            ``"cpu"``, ``"cuda"``, ``"cuda:0"``, ``"cuda:1"``, ``"mps"``,
            ``"gpu"``, ``"gpu:0"``, ``"gpu:1"``.

    Example:
        Phase-level acceleration (single backend)

        ```python
            train_phase = TrainPhase.from_split(
                "train",
                ...,
                accelerator=Accelerator.cuda(0),
            )
        ```

        Node-level acceleration (mixed backends):

        ```python
            torch_node = ModelNode("enc", model=..., accelerator=Accelerator.cuda(0))
        ```

    """

    def __init__(self, device: str = "cpu", *, pin_memory: bool = False):
        """
        Initialize an Accelerator.

        Args:
            device (str): Device string. Accepted values: ``"cpu"``,
                ``"cuda"``, ``"cuda:N"``, ``"mps"``, ``"gpu"``, ``"gpu:N"``.
            pin_memory (bool): If ``True``, CPU tensors are placed in
                page-locked (pinned) memory before the GPU transfer, enabling
                asynchronous PCIe DMA via ``non_blocking=True``. Has no effect
                for CPU or MPS devices. Defaults to ``False``.

        Raises:
            ValueError: If the device string is not recognized.

        """
        device = device.strip().lower()
        _validate_device_str(device)
        self.device = device
        self.pin_memory = pin_memory

    # ================================================
    # Device String Conversion
    # ================================================
    def torch_device_str(self) -> str:
        """
        Return PyTorch-compatible device string.

        Returns:
            str: Device string accepted by ``tensor.to(device)`` and
                ``module.to(device)``. ``"gpu"`` / ``"gpu:N"`` aliases map
                to ``"cuda"`` / ``"cuda:N"``; all others pass through unchanged.

        """
        d = self.device
        if d == "gpu":
            return "cuda"
        if d.startswith("gpu:"):
            idx = _parse_device_index(d)
            return f"cuda:{idx}"
        return d

    def tf_device_str(self) -> str:
        """
        Return TensorFlow-compatible device string.

        Returns:
            str: Device string for ``tf.device()``. Maps:

            * ``"cpu"`` --> ``"/CPU:0"``
            * ``"cuda"`` / ``"gpu"`` --> ``"/GPU:0"``
            * ``"cuda:N"`` / ``"gpu:N"`` --> ``"/GPU:N"``
            * ``"mps"`` --> ``"/CPU:0"`` (TensorFlow does not support MPS)

        """
        d = self.device
        if d == "cpu":
            return "/CPU:0"
        if d == "mps":
            warn(
                "TensorFlow does not support MPS (Apple Silicon GPU). "
                "Falling back to /CPU:0.",
                category=UserWarning,
                stacklevel=3,
            )
            return "/CPU:0"
        if d in ("cuda", "gpu"):
            return "/GPU:0"
        if d.startswith(("cuda:", "gpu:")):
            idx = _parse_device_index(d)
            return f"/GPU:{idx}"
        return "/CPU:0"

    # ================================================
    # PyTorch Helpers
    # ================================================
    def jax_device(self):
        """Resolve a JAX CPU/GPU device, raising if the requested device is absent."""
        if self.device == "mps":
            raise ValueError("JAX Accelerator does not support the mps device alias.")
        platform = "cpu" if self.device == "cpu" else "gpu"
        devices = ensure_jax().devices(platform)
        index = _parse_device_index(self.device) or 0
        if index >= len(devices):
            msg = f"JAX device index {index} is unavailable for {platform}."
            raise ValueError(msg)
        return devices[index]

    @contextmanager
    def jax_device_scope(self):
        """Use the requested device for JAX allocations and computations."""
        with ensure_jax().default_device(self.jax_device()):
            yield

    def setup_torch_model(self, torch_module: Any) -> None:
        """
        Move a :class:`torch.nn.Module` to this device in-place.

        Args:
            torch_module (torch.nn.Module): Module to move.

        """
        torch_module.to(self.torch_device_str())

    def move_torch_tensor(self, tensor: Any) -> Any:
        """
        Return a PyTorch tensor placed on this device.

        When ``pin_memory=True``, the tensor is first moved to page-locked
        CPU memory (if not already pinned) before the GPU transfer, enabling
        asynchronous PCIe DMA.

        Args:
            tensor (torch.Tensor): Tensor to move.

        Returns:
            torch.Tensor: Tensor on this device.

        """
        if self.pin_memory and not tensor.is_pinned():
            tensor = tensor.pin_memory()
        return tensor.to(self.torch_device_str(), non_blocking=self.pin_memory)

    # ================================================
    # TensorFlow Helpers
    # ================================================
    @contextmanager
    def tf_device_scope(self):
        """
        Context manager that places TensorFlow ops on this device.

        Yields:
            None

        Raises:
            ImportError: If TensorFlow is not installed.

        """
        tf = check_tensorflow()
        if tf is None:
            yield
            return
        with tf.device(self.tf_device_str()):
            yield

    # ================================================
    # Availability
    # ================================================
    def is_available(self) -> bool:
        """
        Return True if this device is physically available.

        For ``"cuda"`` / ``"gpu"`` devices, checks PyTorch first via
        ``torch.cuda.is_available()``, then falls back to TensorFlow via
        ``tf.config.list_physical_devices("GPU")`` if PyTorch is not installed.
        For ``"mps"``, checks ``torch.backends.mps.is_available()`` (PyTorch only).
        CPU is always available.

        Returns:
            bool: True when the requested device is usable.

        """
        d = self.device
        if d == "cpu":
            return True
        if d.startswith(("cuda", "gpu")):
            torch = check_torch()
            if torch is not None:
                if not torch.cuda.is_available():
                    return False
                idx = _parse_device_index(d)
                if idx is not None:
                    return idx < torch.cuda.device_count()
                return True
            # Fall back to TensorFlow GPU check
            tf = check_tensorflow()
            if tf is not None:
                gpus = tf.config.list_physical_devices("GPU")
                idx = _parse_device_index(d)
                if idx is not None:
                    return idx < len(gpus)
                return len(gpus) > 0
            return False
        if d == "mps":
            torch = check_torch()
            if torch is None:
                return False
            return torch.backends.mps.is_available()
        return False

    # ================================================
    # Convenience Constructors
    # ================================================
    @classmethod
    def cpu(cls) -> Accelerator:
        """Return a CPU accelerator."""
        return cls(device="cpu")

    @classmethod
    def cuda(cls, index: int = 0, *, pin_memory: bool = False) -> Accelerator:
        """
        Return a CUDA accelerator for the given device index.

        Args:
            index (int): CUDA device index (0-based). Defaults to 0.
            pin_memory (bool): Enable pinned memory for async PCIe transfers.
                Defaults to ``False``.

        Returns:
            Accelerator: Accelerator targeting ``cuda:{index}``.

        """
        return cls(device=f"cuda:{index}", pin_memory=pin_memory)

    @classmethod
    def mps(cls) -> Accelerator:
        """Return an MPS (Apple Silicon) accelerator."""
        return cls(device="mps")

    @classmethod
    def gpu(cls, index: int = 0, *, pin_memory: bool = False) -> Accelerator:
        """
        Return a GPU accelerator (backend-agnostic alias for :meth:`cuda`).

        Maps to ``"cuda:N"`` for PyTorch and ``"/GPU:N"`` for TensorFlow.

        Args:
            index (int): GPU device index (0-based). Defaults to 0.
            pin_memory (bool): Enable pinned memory for async PCIe transfers.
                Defaults to ``False``.

        Returns:
            Accelerator: Accelerator targeting ``gpu:{index}``.

        """
        return cls(device=f"gpu:{index}", pin_memory=pin_memory)

    @classmethod
    def all_available(cls) -> list[str]:
        """
        Return a list of all available device strings on this machine.

        Tries PyTorch first; if not installed, falls back to TensorFlow.
        CPU is always included. CUDA/GPU devices are listed as ``"gpu:N"``
        (backend-agnostic form). MPS is included when available (PyTorch only).

        Returns:
            list[str]: Available device strings, e.g.
                ``["cpu", "gpu:0", "gpu:1", "mps"]``.

        """
        devices: list[str] = ["cpu"]
        torch = check_torch()
        if torch is not None:
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    devices.append(f"gpu:{i}")
            if torch.backends.mps.is_available():
                devices.append("mps")
            return devices
        tf = check_tensorflow()
        if tf is not None:
            gpus = tf.config.list_physical_devices("GPU")
            for i in range(len(gpus)):
                devices.append(f"gpu:{i}")
        return devices

    # ================================================
    # Dunders
    # ================================================
    def __repr__(self) -> str:
        if self.pin_memory:
            return f"Accelerator(device={self.device!r}, pin_memory=True)"
        return f"Accelerator(device={self.device!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Accelerator):
            return NotImplemented
        return self.device == other.device and self.pin_memory == other.pin_memory

    def __hash__(self) -> int:
        return hash((self.device, self.pin_memory))

    def get_config(self) -> dict[str, Any]:
        """
        Return a JSON-serializable configuration for this accelerator.

        Returns:
            dict[str, Any]: Accelerator configuration payload.

        """
        return {
            "device": self.device,
            "pin_memory": self.pin_memory,
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Accelerator:
        """
        Reconstruct an :class:`Accelerator` from :meth:`get_config` output.

        Args:
            config (dict[str, Any]): Serialized accelerator dictionary.

        Returns:
            Accelerator: Restored instance.

        """
        return cls(
            device=config.get("device", "cpu"),
            pin_memory=config.get("pin_memory", False),
        )
