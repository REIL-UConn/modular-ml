import math
from collections import Counter
from collections.abc import Iterable, Sequence
from typing import Any, TypeVar

import numpy as np

T = TypeVar("T")


def normal_round(num, ndigits: int = 0):
    """
    Rounds a float to the specified number of decimal places.

    Args:
        num (any): the value to round
        ndigits: the number of digits to round to.

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
        round_integers (bool): If True, round integers to the specified
            significant digits as well. If False, only apply decimal rounding.

    Returns:
        str: String representation of the formatted number.

    Examples:
        >>> format_value_to_sig_digits(1512.345, 3)
        '1510'
        >>> format_value_to_sig_digits(12.3456, 3)
        '12.3'
        >>> format_value_to_sig_digits(0.003123, 3)
        '0.00312'
        >>> format_value_to_sig_digits(0.00001234, 3)
        '0.0000123'
        >>> format_value_to_sig_digits(1235.123, 2)
        '1200'
        >>> format_value_to_sig_digits(1235.123, 2, round_integers=False)
        '1235'

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
        TypeError:
            If a list contains non-string elements, or if an unsupported
            value type is encountered in the dictionary.

    Examples:
        ```python
        flatten_dict_paths({"a": ["b", "c"]})
        # ['a.b', 'a.c']

        flatten_dict_paths({"a": {"b": "c", "d": "e"}})
        # ['a.b.c', 'a.d.e']

        flatten_dict_paths({"a": {"b": ["c", "f"], "d": ["e", "g"]}})
        # ['a.b.c', 'a.b.f', 'a.d.e', 'a.d.g']
        ```

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
        x: Any input value.

    Returns:
        list: A list representation of `x`.

    Raises:
        TypeError: If the input is not convertible to a list.

    """
    if x is None:
        return []

    # If it's already a list, return directly
    if isinstance(x, list):
        return x

    # Treat strings and all scalar types as atomic -> wrap in list
    if isinstance(x, (str, bytes, int, float, bool)):
        return [x]

    # If it's a sequence (tuple, np.array, etc.), convert to list
    if isinstance(x, Sequence):
        return list(x)

    # For any other single object (e.g. Enum, custom class), also wrap in list
    return [x]


def ensure_tuple(x):
    """
    Convert an object to a tuple.

    Description:
        Scalars (int, float, str, etc.) are wrapped as a single-element tuple.
        Iterables (excluding strings and bytes) are converted via tuple(x).

    Args:
        x (Any):
            Object to convert.

    Returns:
        tuple:
            Tuple representation of the input.

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
        items (list[T]):
            A list of elements.

        ignore_case (bool, optional):
            Whether to ignore the case of any string elements in `items`.
            E.g., if True, `'mse'` would be a duplicate of `'MSE'`.

    Returns:
        list[T]: A list of all elements which are not unique.

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
        split_names (Iterable[str]):
            Collection of split name strings to sort.

    Returns:
        list[str]:
            Sorted list of split names following semantic priority.

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
