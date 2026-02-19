"""Declarative similarity/dissimilarity rules for sampler roles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from modularml.core.io.protocols import Configurable

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass
class SimilarityCondition(Configurable):
    """
    A flexible rule for defining similarity or dissimilarity between two samples.

    Attributes:
        mode ({"similar","dissimilar"}, default="similar"):
            Whether to select pairs that are within tolerance ("similar") \
            or explicitly outside tolerance ("dissimilar").

        tolerance (float, default=0.0):
            Numeric threshold for values to be considered "similar". \
            Example: tolerance=0.05 means |a - b| <= 0.05 is a valid match.

        metric (Callable[[float, float], float] | None, default=None):
            Optional custom distance function. If not provided:
            - Numeric types use absolute difference.
            - Other types use equality.

        weight_mode ({"binary","linear","exp"}, default="binary"):
            Strategy for assigning weights to valid or fallback pairs:
            - "binary": All matches get weight 1.0; non-matches get 0.1 if
            `allow_fallback=True`.
            - "linear": Weight = tolerance / diff (≥1 for matches, <1 for non-matches).
            - "exp": Weight = exp(1 - diff / tolerance), clipped at `max_weight`.

        max_weight (float, default=100):
            Maximum weight allowed when using `weight_mode != 'uniform'`

        min_weight (float, default=0.1):
            Minimum weight allowed when using `weight_mode != 'uniform'`

        allow_fallback (bool, default=False):
            If True, pairs that fail the match condition are not discarded; \
            they are instead given a down-weighted score (<1). \
            If False, non-matches always score 0.0.

    Weight semantics:
        - Matches always receive weight ≥ 1.0 (better matches → larger weight).
        - Non-matches receive weight < 1.0 if `allow_fallback=True`, else 0.0.
        - `fallback=False` is equivalent to \
            `weightmode='binary', min_weight=0.0, max_weight=1.0`

    Examples:
        >>> cond = SimilarityCondition(mode="similar", tolerance=0.5, weight_mode="linear")
        >>> cond.score(0.80, 0.82)  # doctest: +FLOAT_CMP
        25.0

        >>> cond.score(0.80, 0.90)  # doctest: +FLOAT_CMP
        5.0

        >>> cond.score(0.80, 1.5)  # doctest: +FLOAT_CMP
        0.0

    """

    mode: Literal["similar", "dissimilar"] = "similar"

    tolerance: float = 0.0
    metric: Callable[[float, float], float] | None = None

    weight_mode: Literal["binary", "linear", "exp"] = "binary"
    max_weight: float = 1.0
    min_weight: float = 0.0
    allow_fallback: bool = False

    def __postinit__(self):
        """Validate mode and weight-mode configuration."""
        valid_modes = ["similar", "dissimilar"]
        if self.mode not in valid_modes:
            msg = f"`mode` must be one of: {valid_modes}. Received: {self.mode}"
            raise ValueError(msg)

        valid_weight_modes = ["binary", "linear", "exp"]
        if self.weight_mode not in valid_weight_modes:
            msg = (
                f"`weight_mode` must be one of: {valid_weight_modes}. "
                f"Received: {self.weight_mode}"
            )
            raise ValueError(msg)

    def score(self, a, b) -> float:
        """
        Compute the similarity/dissimilarity score between two values.

        Args:
            a: First value (anchor).
            b: Second value (candidate).

        Returns:
            float: A non-negative weight score.

        Notes:
            - For matches (diff <= tolerance if mode="similar"):
                Score >= 1.0 (better matches = larger weight).
            - For non-matches:
                Score < 1.0 if `allow_fallback=True`.
                Score = 0.0 if `allow_fallback=False`.

        """
        # Step 1: compute difference
        if self.metric:
            diff = abs(self.metric(a, b))
        elif isinstance(a, (int, float, np.number)) and isinstance(
            b,
            (int, float, np.number),
        ):
            diff = abs(a - b)
        else:
            diff = 0 if a == b else float("inf")

        # Step 2: check match condition
        is_match = diff <= self.tolerance
        if self.mode == "dissimilar":
            is_match = not is_match
        if not self.allow_fallback:
            return int(is_match)

        # Step 3: assign weight
        # - All valid matches get a weight >= 1
        # - All invalid matches get a weight < 1
        if self.weight_mode == "binary":
            return self.max_weight if is_match else self.min_weight

        if self.weight_mode == "linear":
            # Inverse linear weight based on how much better than match condition this pairing is
            if self.mode == "similar":
                # Smaller diff = higher weight
                # e.g. diff=0.2, tol=0.5 -> 2.5
                return np.clip(
                    self.tolerance / max(diff, 1e-9),
                    a_max=self.max_weight,
                    a_min=self.min_weight,
                ).__float__()
            # Dissimilar = Larger diff -> higher weight
            # e.g. diff=1.0, tol=0.5 -> 2.0
            return np.clip(
                diff / max(self.tolerance, 1e-9),
                a_max=self.max_weight,
                a_min=self.min_weight,
            ).__float__()

        if self.weight_mode == "exp":
            # Exponential weight based on how much better than match condition this pairing is
            if self.mode == "similar":
                # Smaller diff = higher weight
                # e.g. diff=0.2, tol=0.5 -> np.exp(1 - 0.2/0.5) = 1.822
                return np.clip(
                    float(np.exp(1 - diff / max(self.tolerance, 1e-9))),
                    a_max=self.max_weight,
                    a_min=self.min_weight,
                ).__float__()
            # Dissimilar = Larger diff -> higher weight
            # e.g., diff=1.0, tol=0.5 -> np.exp(1/0.5 - 1) = 2.718
            return np.clip(
                float(np.exp(diff / max(self.tolerance, 1e-9) - 1)),
                a_max=self.max_weight,
                a_min=self.min_weight,
            ).__float__()

        return 0.0

    # ================================================
    # Configuration
    # ================================================
    def get_config(self) -> dict[str, Any]:
        """
        Return a JSON-serializable configuration for this condition.

        Returns:
            dict[str, Any]: Configuration capturing all constructor arguments.

        """
        return {
            "mode": self.mode,
            "tolerance": self.tolerance,
            "metric": self.metric,
            "weight_mode": self.weight_mode,
            "max_weight": self.max_weight,
            "min_weight": self.min_weight,
            "allow_fallback": self.allow_fallback,
        }

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> SimilarityCondition:
        """
        Reconstruct a condition from configuration.

        Args:
            cfg (dict[str, Any]): Dictionary previously produced by :meth:`get_config`.

        Returns:
            SimilarityCondition: Rehydrated similarity condition.

        """
        return cls(**cfg)
