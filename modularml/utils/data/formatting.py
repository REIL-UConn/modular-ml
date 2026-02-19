"""Data formatting helpers for rounding, flattening, and slicing structures."""

import math
from collections import Counter
from collections.abc import Iterable, Sequence
from typing import Any, TypeVar

import numpy as np

T = TypeVar("T")


def normal_round(num, ndigits: int = 0):
    """
    Round a numeric value to the specified number of decimal places.

    Args:
        num (float | int): Value to round.
        ndigits (int): Number of digits to round to.

    Returns:
        float | int: Rounded value.

    """
    if ndigits == 0:
        return int(num + (np.sign(num) * 0.5))
    digit_value = 10**ndigits
    return int(num * digit_value + (np.sign(num) * 0.5)) / digit_value


def format_value_to_sig_digits(
    value: float,
    sig_digits: int = 3,
    *,
    round_integers: bool = True,
) -> str:
    """
    Format a value to a specified number of significant digits.

    Args:
        value (float): Input number.
        sig_digits (int): Number of significant digits to keep.
        round_integers (bool): Round integers to the specified significant digits when True.

    Returns:
        str: String representation of the formatted number.

    Examples:
        >>> format_value_to_sig_digits(1512.345, 3)
        '1510'
        >>> format_value_to_sig_digits(12.3456, 3)
        '12.3'

    """
    if value == 0 or math.isnan(value):
        return "0"

    order = math.floor(math.log10(abs(value)))
    if round_integers:
        # Scale number so we can round to sig_digits
        factor = 10 ** (order - sig_digits + 1)
        rounded = normal_round(value / factor) * factor
        # Decide decimals: if order >= sig_digits-1 → no decimals, else show needed
        decimals = max(0, sig_digits - order - 1)
        return f"{rounded:.{decimals}f}"

    decimals = max(0, sig_digits - order - 1)
    return f"{value:.{decimals}f}"


def flatten_dict_paths(
    d: dict[str, any],
    prefix: str = "",
    separator: str = ".",
) -> list[str]:
    """
    Recursively flatten a nested dictionary into separator-joined key paths.

    Description:
        This function traverses any nested dictionary whose:
          - Keys are strings
          - Values may be: a string, a list of strings, or another nested dictionary

        The resulting paths are returned as strings joined by the given
        separator (default: "."). Each terminal element (string or list item)
        forms the end of a full path.

    Args:
        d (dict[str, any]):
            Input nested dictionary to flatten.
        prefix (str, optional):
            Internal recursion prefix. Should generally be left empty.
        separator (str, optional):
            String used to join nested key names. Defaults to ".".

    Returns:
        list[str]:
            A list of fully flattened paths using the specified separator.

    Raises:
        TypeError: If a sequence contains non-string elements or any value has an unsupported type.

    Examples:
        >>> flatten_dict_paths({"a": ["b", "c"]})
        ['a.b', 'a.c']
        >>> flatten_dict_paths({"a": {"b": "c", "d": "e"}})
        ['a.b.c', 'a.d.e']
        >>> flatten_dict_paths({"a": {"b": ["c", "f"], "d": ["e", "g"]}})
        ['a.b.c', ...]

    """
    paths = []
    for k, v in d.items():
        full_prefix = f"{prefix}.{k}" if prefix else k

        if isinstance(v, dict):
            # Recurse deeper into nested dict
            paths.extend(flatten_dict_paths(v, prefix=full_prefix, separator=separator))
        elif isinstance(v, str):
            # Terminal string value
            paths.append(f"{full_prefix}.{v}")
        elif isinstance(v, list):
            # List of terminal strings
            for item in v:
                if not isinstance(item, str):
                    msg = f"List values must be strings, got {type(item)} in key '{k}'"
                    raise TypeError(msg)
                paths.append(f"{full_prefix}.{item}")
        else:
            msg = f"Unsupported value type {type(v)} for key '{k}'"
            raise TypeError(msg)

    return paths


def ensure_list(x):
    """
    Ensure that the input is returned as a list.

    - None: return []
    - list: return itself (unchanged)
    - scalar (str, int, float, bool, etc.): return wrapped in a list
    - any other non-sequence type: return wrapped in a list
    - any other sequence (tuple, set, np.ndarray): return converted to list

    Args:
        x (Any): Input value.

    Returns:
        list[Any]: List representation of `x`.

    """
    if x is None:
        return []

    if isinstance(x, list):
        return x

    if isinstance(x, (str, bytes, int, float, bool)):
        return [x]

    if isinstance(x, Sequence):
        return list(x)

    return [x]


def ensure_tuple(x):
    """
    Convert an object to a tuple.

    Description:
        Scalars (int, float, str, etc.) are wrapped as a single-element tuple.
        Iterables (excluding strings and bytes) are converted via tuple(x).

    Args:
        x (Any): Object to convert.

    Returns:
        tuple[Any, ...]: Tuple representation of the input.

    """
    if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
        return tuple(x)
    return (x,)


def to_hashable(val: Any):
    """
    Convert a value into a hashable representation suitable for grouping keys.

    Description:
        - Scalars are returned as-is.
        - NumPy arrays are converted to tuples.
        - Lists are converted to tuples.
        - Nested structures are recursively converted.

    Args:
        val (Any): Value to convert.

    Returns:
        Hashable: Hashable representation of `val`.

    """
    if isinstance(val, np.ndarray):
        return tuple(to_hashable(x) for x in val.tolist())

    if isinstance(val, (list, tuple)):
        return tuple(to_hashable(x) for x in val)

    # NumPy scalar to Python scalar
    if isinstance(val, np.generic):
        return val.item()

    return val


def find_duplicates(
    items: list[T],
    *,
    ignore_case: bool = False,
) -> list[T]:
    """
    Find all duplicate elements in a list.

    Args:
        items (list[T]): Elements to inspect.
        ignore_case (bool, optional): Compare string elements case-insensitively.

    Returns:
        list[T]: Duplicate entries preserving their canonical representation.

    """
    if ignore_case:
        items = [itm.lower() if isinstance(itm, str) else itm for itm in items]

    counts = Counter(items)
    return [item for item, count in counts.items() if count > 1]


def sort_split_names(split_names: Iterable[str]) -> list[str]:
    """
    Sort FeatureSet split names by semantic priority (train -> val -> test -> other).

    Description:
        Orders split names such that any split containing "train" appears first,
        followed by splits containing "val", then "test", and finally all remaining
        splits. Matching is case-insensitive and stable within each priority group.

    Args:
        split_names (Iterable[str]): Split name strings to sort.

    Returns:
        list[str]: Sorted list of split names following semantic priority.

    """

    def split_priority(name: str) -> tuple[int, str]:
        lname = name.lower()

        if "train" in lname:
            priority = 0
        elif "val" in lname:
            priority = 1
        elif "test" in lname:
            priority = 2
        else:
            priority = 3

        # Secondary key keeps deterministic ordering
        return priority, name

    return sorted(split_names, key=split_priority)
