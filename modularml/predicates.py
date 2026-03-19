"""Serializable condition predicates for :class:`ConditionSplitter`."""

from __future__ import annotations

import ast
import operator
import textwrap
from pathlib import Path
from typing import TYPE_CHECKING, Any

from modularml.utils.logging.logger import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable

logger = get_logger("Predicate")


class Predicate:
    """
    Serializable callable condition for filtering :class:`FeatureSet` samples.

    Description:
        Predicates wrap common comparison and membership operations in a
        form that can be saved and restored via :meth:`to_dict` /
        :meth:`from_dict`.  Use these instead of raw lambdas when you need
        the containing :class:`ConditionSplitter` (and therefore the
        :class:`FeatureSet`) to be serializable.

        For arbitrary logic that cannot be expressed with the built-in
        predicates, use :class:`Lambda` and supply the source string
        explicitly.

    Available predicates::

        LT(v)      -> x < v
        LTE(v)     -> x <= v
        GT(v)      -> x > v
        GTE(v)     -> x >= v
        EQ(v)      -> x == v
        NE(v)      -> x != v
        In([...])  -> x in [...]
        NotIn([…]) -> x not in [...]
        Lambda("lambda x: …")

    """

    def __call__(self, x: Any) -> bool:
        raise NotImplementedError

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of this predicate."""
        raise NotImplementedError

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Predicate:
        """Reconstruct a :class:`Predicate` from a dictionary produced by :meth:`to_dict`."""
        _type_map: dict[str, type[Predicate]] = {
            "lt": LT,
            "lte": LTE,
            "gt": GT,
            "gte": GTE,
            "eq": EQ,
            "ne": NE,
            "in": In,
            "not_in": NotIn,
            "lambda": Lambda,
        }
        pred_type = d.get("type")
        if pred_type not in _type_map:
            msg = f"Unknown predicate type: '{pred_type}'"
            raise ValueError(msg)
        return _type_map[pred_type].from_dict(d)


# ================================================
# Comparison predicates
# ================================================
class _ComparisonPredicate(Predicate):
    _type: str
    _op: Callable[[Any, Any], bool]

    def __init__(self, value: Any) -> None:
        self.value = value

    def __call__(self, x: Any) -> bool:
        return self._op(x, self.value)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.value!r})"

    def to_dict(self) -> dict[str, Any]:
        return {"type": self._type, "value": self.value}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> _ComparisonPredicate:
        return cls(d["value"])


class LT(_ComparisonPredicate):
    """True when ``x < value``."""

    _type = "lt"
    _op = staticmethod(operator.lt)


class LTE(_ComparisonPredicate):
    """True when ``x <= value``."""

    _type = "lte"
    _op = staticmethod(operator.le)


class GT(_ComparisonPredicate):
    """True when ``x > value``."""

    _type = "gt"
    _op = staticmethod(operator.gt)


class GTE(_ComparisonPredicate):
    """True when ``x >= value``."""

    _type = "gte"
    _op = staticmethod(operator.ge)


class EQ(_ComparisonPredicate):
    """True when ``x == value``."""

    _type = "eq"
    _op = staticmethod(operator.eq)


class NE(_ComparisonPredicate):
    """True when ``x != value``."""

    _type = "ne"
    _op = staticmethod(operator.ne)


# ================================================
# Membership predicates
# ================================================
class In(Predicate):
    """True when ``x in values``."""

    def __init__(self, values: list[Any]) -> None:
        self.values = list(values)

    def __call__(self, x: Any) -> bool:
        import numpy as np

        if isinstance(x, np.ndarray):
            return np.isin(x, self.values)
        return x in self.values

    def __repr__(self) -> str:
        return f"In({self.values!r})"

    def to_dict(self) -> dict[str, Any]:
        return {"type": "in", "values": self.values}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> In:
        return cls(d["values"])


class NotIn(Predicate):
    """True when ``x not in values``."""

    def __init__(self, values: list[Any]) -> None:
        self.values = list(values)

    def __call__(self, x: Any) -> bool:
        import numpy as np

        if isinstance(x, np.ndarray):
            return ~np.isin(x, self.values)
        return x not in self.values

    def __repr__(self) -> str:
        return f"NotIn({self.values!r})"

    def to_dict(self) -> dict[str, Any]:
        return {"type": "not_in", "values": self.values}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> NotIn:
        return cls(d["values"])


# ================================================
# Lambda predicate
# ================================================
class Lambda(Predicate):
    """
    Predicate backed by an explicit lambda source string.

    Args:
        source (str):
            Python expression string for a single-argument callable,
            e.g. ``"lambda x: x < 45"``.

    Example:
    ```python
        pred = Lambda("lambda x: x < 45")
        pred(40)   # True
        pred(50)   # False
    ```

    """

    def __init__(self, source: str) -> None:
        self.source = source
        self._fn: Callable[[Any], bool] = eval(source)  # noqa: S307

    def __call__(self, x: Any) -> bool:
        return self._fn(x)

    def __repr__(self) -> str:
        return f"Lambda({self.source!r})"

    def __getstate__(self) -> dict[str, Any]:
        return {"source": self.source}

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.source = state["source"]
        self._fn = eval(self.source)  # noqa: S307

    def to_dict(self) -> dict[str, Any]:
        return {"type": "lambda", "source": self.source}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Lambda:
        return cls(d["source"])


# ================================================
# Lambda source extraction helper
# ================================================
def _extract_lambda_source(fn: Callable) -> str:
    """
    Attempt to extract the source string of a lambda function.

    Description:
        First tries to parse the originating source file via :mod:`ast`
        (most reliable). Falls back to :func:`inspect.getsource` for
        environments where the source file is unavailable (e.g. Jupyter
        notebooks accessed through ``linecache``).

    Args:
        fn: A lambda (or named) callable whose source should be extracted.

    Returns:
        str: The extracted source string, e.g. ``"lambda x: x < 45"``.

    Raises:
        ValueError:
            If the source cannot be extracted automatically. The error
            message explains how to use :class:`Lambda` explicitly.

    """
    import inspect

    filename = getattr(fn.__code__, "co_filename", None)
    lineno = getattr(fn.__code__, "co_firstlineno", None)

    # Approach 1: read from source file using AST (most reliable)
    if filename and Path(filename).is_file():
        try:
            with Path(filename).open(encoding="utf-8") as f:
                source = f.read()
            tree = ast.parse(source)
            lambdas_at_line = [
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.Lambda) and node.lineno == lineno
            ]
            if len(lambdas_at_line) == 1:
                seg = ast.get_source_segment(source, lambdas_at_line[0])
                if seg:
                    return seg
        except Exception as e:  # noqa: BLE001
            msg = f"Lambda extraction method 1 failed: {e}"
            logger.debug(msg)

    # Approach 2: inspect.getsource (works for notebooks via linecache)
    try:
        raw = inspect.getsource(fn)
        src = textwrap.dedent(raw)
        # The raw source may be a line fragment; try parsing with/without a wrapper
        for wrapper in ("", "_x = "):
            try:
                wrapped = f"{wrapper}{src}"
                tree = ast.parse(wrapped)
                lambdas = [n for n in ast.walk(tree) if isinstance(n, ast.Lambda)]
                if len(lambdas) == 1:
                    seg = ast.get_source_segment(wrapped, lambdas[0])
                    if seg:
                        return seg
            except SyntaxError as e:  # noqa: PERF203
                msg = f"Lambda extraction method 2; failed to parse line fragment: {e}"
                logger.debug(msg)
                continue

    except Exception as e:  # noqa: BLE001
        msg = f"Lambda extraction method 2 failed: {e}"
        logger.debug(msg)

    msg = (
        "Cannot automatically extract lambda source. "
        "Wrap the condition explicitly:\n\n"
        "    from modularml.predicates import Lambda\n"
        '    "col_name": Lambda("lambda x: x < 45")\n'
    )
    raise ValueError(msg)
