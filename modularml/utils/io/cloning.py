"""Utilities for cloning objects via serialization."""

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from modularml.core.io.serialization_policy import SerializationPolicy


def clone_via_serialization(obj: Any):
    """
    Create an isolated clone of an object by serializing and deserializing it.

    Args:
        obj (Any): Object implementing the serializer protocols.

    Returns:
        Any: Deep-cloned instance created via artifact round-trip.

    """
    from modularml.core.io.serializer import serializer

    with TemporaryDirectory() as tmpdir:
        artifact_path = Path(tmpdir) / "tmp_clone.mml"

        # Serializer may update save path based on expected file extension
        # It returns the path it actually saved to
        save_path = serializer.save(
            obj=obj,
            save_path=artifact_path,
            policy=SerializationPolicy.STATE_ONLY,
            overwrite=True,
        )
        clone = serializer.load(
            path=save_path,
            provided_class=type(obj),
            allow_packaged_code=True,
        )

    return clone
