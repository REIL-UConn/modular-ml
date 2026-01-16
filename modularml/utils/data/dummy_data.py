import uuid

import numpy as np

from modularml.core.data.sample_data import SampleData


def make_dummy_sample_data(
    batch_size: int,
    feature_shape: tuple[int, ...],
    target_shape: tuple[int, ...] | None = None,
    tag_shape: tuple[int, ...] | None = None,
    seed: int = 1,
):
    """Feature/target/tag shapes must not include batch_dim."""
    rng = np.random.default_rng(seed=seed)
    return SampleData(
        sample_uuids=np.asarray([str(uuid.uuid4()) for _ in range(batch_size)]),
        features=rng.random(size=(batch_size, *feature_shape)),
        targets=rng.random(size=(batch_size, *target_shape))
        if target_shape is not None
        else None,
        tags=rng.random(size=(batch_size, *tag_shape))
        if tag_shape is not None
        else None,
    )
