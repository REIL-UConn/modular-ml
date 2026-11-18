"""FeatureSet.from_dict accepts the object-dtype arrays pandas produces."""

import numpy as np
import pandas as pd
import pytest

from modularml.core.data.featureset import FeatureSet


@pytest.mark.unit
def test_from_dict_accepts_pandas_object_columns():
    df = pd.DataFrame(
        {
            "x": [0.1, 0.2, 0.3],
            "y": [0, 1, 0],
            "site": ["a", "b", "a"],
            "count": pd.Series([1, 2, 3], dtype=object),
        },
    )
    assert df["site"].to_numpy().dtype == object
    fs = FeatureSet.from_dict(
        "fs",
        {c: df[c].to_numpy() for c in df.columns},
        feature_keys="x",
        target_keys="y",
        tag_keys=["site", "count"],
    )
    tags = fs.get_tags(fmt="pandas", tags=["site", "count"])
    assert list(tags["site"]) == ["a", "b", "a"]
    assert list(tags["count"]) == [1, 2, 3]


@pytest.mark.unit
def test_from_dict_rejects_mixed_object_columns():
    with pytest.raises(TypeError, match="'bad'"):
        FeatureSet.from_dict(
            "fs",
            {
                "x": np.zeros(2),
                "y": np.zeros(2),
                "bad": np.array(["a", 1], dtype=object),
            },
            feature_keys="x",
            target_keys="y",
            tag_keys="bad",
        )
