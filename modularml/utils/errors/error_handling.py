"""Lightweight error mode enumeration used across utilities."""

from enum import Enum


class ErrorMode(str, Enum):
    """Strategies for responding to recoverable errors."""

    RAISE = "raise"
    WARN = "warn"
    IGNORE = "ignore"
    COERCE = "coerce"
