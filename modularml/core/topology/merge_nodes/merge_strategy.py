"""Merge strategies for non-concatenation domain aggregation."""

from enum import Enum


class MergeStrategy(str, Enum):
    """
    Strategy for merging domain data across multiple inputs.

    Description:
        This enum defines how a MergeNode should combine data from
        multiple upstream inputs for a given domain (e.g., targets, tags).
        Strategies other than `CONCAT` bypass axis-based concatenation
        and apply an aggregation function instead.

    Values:
        CONCAT: Concatenate along a specified axis (requires an int axis).
        FIRST: Use data from the first input only.
        LAST: Use data from the last input only.
        MEAN: Element-wise mean across inputs (shapes must match).

    """

    CONCAT = "concat"
    FIRST = "first"
    LAST = "last"
    MEAN = "mean"
