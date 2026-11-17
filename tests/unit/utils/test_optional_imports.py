"""Unit tests for modularml.utils.environment.optional_imports module."""

import pytest

from modularml.utils.environment import optional_imports


@pytest.mark.unit
@pytest.mark.parametrize(
    "check",
    [
        optional_imports.check_jax,
        optional_imports.check_torch,
        optional_imports.check_tensorflow,
    ],
)
def test_check_functions_cache_their_result(check):
    """
    Test check_* helpers only attempt each import once.

    A failed import is not remembered by Python, so an uncached `check_jax()`
    re-scanned `sys.path` on every call. It runs once per element inside
    `to_numpy`, which made converting plain Python lists extremely slow whenever
    JAX was not installed.
    """
    check.cache_clear()
    first = check()
    second = check()

    assert first is second
    assert check.cache_info().misses == 1
    assert check.cache_info().hits == 1
