"""Padding utilities shared across multiple neural network backends."""

from enum import Enum

from modularml.utils.nn.backend import Backend


class PadMode(str, Enum):
    """
    Enum representing universal padding modes.

    Description:
        This enum standardizes padding mode names across different backends
        (e.g., PyTorch, TensorFlow, NumPy). Use this to ensure consistent
        behavior regardless of the chosen backend.

    Values:
        CONSTANT: Pad with a constant value.
        REFLECT: Reflect without including edge (mirror).
        REPLICATE: Repeat edge values.
        CIRCULAR: Wrap around (circular padding).
    """

    CONSTANT = "constant"
    REFLECT = "reflect"
    REPLICATE = "replicate"
    CIRCULAR = "circular"


def map_pad_mode_to_backend(mode: PadMode, backend: str | Backend) -> str:
    """
    Map a universal PadMode to backend-specific string.

    Args:
        mode (PadMode): Universal padding mode.
        backend (str | Backend): Backend identifier (`torch`, `tensorflow`, `scikit`).

    Returns:
        str: Backend-specific padding mode string.

    Raises:
        ValueError: If mode or backend is unsupported.

    """
    if isinstance(backend, str):
        backend = Backend(backend.lower())

    match (mode, backend):
        case (PadMode.CONSTANT, _):
            return "constant"
        case (
            (PadMode.REFLECT, "torch")
            | (PadMode.REFLECT, "tensorflow")
            | (PadMode.REFLECT, "scikit")
        ):
            return "reflect"
        case (PadMode.REPLICATE, "torch"):
            return "replicate"
        case (PadMode.REPLICATE, "tensorflow"):
            return "SYMMETRIC"
        case (PadMode.REPLICATE, "scikit"):
            return "edge"
        case (PadMode.CIRCULAR, "torch"):
            return "circular"
        case (PadMode.CIRCULAR, "scikit"):
            return "wrap"
        case _:
            msg = f"Pad mode {mode} not supported for backend '{backend}'"
            raise ValueError(msg)
