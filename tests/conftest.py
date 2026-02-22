"""Shared fixtures and utilities for unit tests."""

from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest

from modularml.core.data.featureset import FeatureSet
from modularml.core.experiment.experiment_context import ExperimentContext

rng = np.random.default_rng(seed=42)


@pytest.fixture(autouse=True)
def _fresh_experiment_context():
    """Provide a fresh ExperimentContext for every test to avoid label collisions."""
    ctx = ExperimentContext()
    with ctx.activate():
        yield


def generate_dummy_data(
    shape: tuple[int, ...] = (),
    dtype: str = "float",
    *,
    min_val: Any = None,
    max_val: Any = None,
    choices: Sequence[Any] | None = None,
) -> float | int | str | np.ndarray:
    """Generate dummy scalar or array data of specified type."""
    size = int(np.prod(shape)) if shape else 1

    if dtype == "str":
        if not choices:
            raise ValueError("For dtype='str', please provide `choices` list.")
        values = rng.choice(choices, size=size)
    elif choices is not None:
        values = rng.choice(choices, size=size)
    elif dtype == "float":
        lo = min_val if min_val is not None else 0.0
        hi = max_val if max_val is not None else 1.0
        values = rng.uniform(lo, hi, size)
    elif dtype == "int":
        lo = min_val if min_val is not None else 0
        hi = max_val if max_val is not None else 10
        values = rng.integers(lo, hi + 1, size)
    else:
        msg = f"Unsupported dtype: {dtype}"
        raise ValueError(msg)

    if not shape or shape == (1,):
        return values[0]
    return np.array(values).reshape(shape)


def generate_dummy_featureset(
    feature_shape_map: dict[str, tuple[int, ...]] | None = None,
    target_shape_map: dict[str, tuple[int, ...]] | None = None,
    tag_type_map: dict[str, str] | None = None,
    target_type: str = "numeric",
    n_samples: int = 1000,
    label: str = "TestFeatureSet",
) -> FeatureSet:
    """Generate a synthetic FeatureSet for testing."""
    feature_shape_map = feature_shape_map or {"X1": (1, 100), "X2": (1, 100)}
    target_shape_map = target_shape_map or {"Y1": (1, 1), "Y2": (1, 10)}
    tag_type_map = tag_type_map or {"T_FLOAT": "float", "T_STR": "str"}

    for k in feature_shape_map:
        feature_shape_map[k] = (n_samples, *feature_shape_map[k])
    for k in target_shape_map:
        target_shape_map[k] = (n_samples, *target_shape_map[k])

    features = {
        k: generate_dummy_data(shape=v, dtype="float")
        for k, v in feature_shape_map.items()
    }

    if target_type == "numeric":
        targets = {
            k: generate_dummy_data(shape=v, dtype="float")
            for k, v in target_shape_map.items()
        }
    elif target_type == "categorical":
        targets = {
            k: generate_dummy_data(shape=v, dtype="str", choices=["A", "B", "C"])
            for k, v in target_shape_map.items()
        }
    else:
        msg = f"Unsupported target_type: {target_type}"
        raise ValueError(msg)

    tags = {
        k: generate_dummy_data(
            shape=(n_samples,),
            dtype=v,
            choices=["red", "blue", "green"] if v == "str" else None,
        )
        for k, v in tag_type_map.items()
    }

    return FeatureSet.from_dict(
        label=label,
        data=features | targets | tags,
        feature_keys=list(feature_shape_map.keys()),
        target_keys=list(target_shape_map.keys()),
        tag_keys=list(tag_type_map.keys()),
    )
