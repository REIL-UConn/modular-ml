"""Torch-native reference models shipped with ModularML."""

from .sequential_cnn import SequentialCNN
from .sequential_mlp import SequentialMLP

__all__ = [
    "SequentialCNN",
    "SequentialMLP",
]
