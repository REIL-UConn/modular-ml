"""
Unit tests for modularml.core.data.registry module.

Note: register_builtin() and register_kinds() are called at import time,
so we verify their effects rather than calling them again (which would raise).
"""

import pytest

from modularml.core.data.featureset import FeatureSet
from modularml.core.io.conventions import kind_registry
from modularml.core.io.symbol_registry import symbol_registry


# ---------------------------------------------------------------------
# Registration tests
# ---------------------------------------------------------------------
@pytest.mark.unit
def test_featureset_in_symbol_registry():
    """Test that FeatureSet is registered in symbol registry (via register_builtin)."""
    assert "FeatureSet" in symbol_registry._builtin_classes
    assert symbol_registry._builtin_classes["FeatureSet"] is FeatureSet


@pytest.mark.unit
def test_featureset_kind_registered():
    """Test that FeatureSet serialization kind is registered (via register_kinds)."""
    kind = kind_registry.get_kind(FeatureSet)
    assert kind is not None
    assert kind.name == "FeatureSet"
    assert kind.kind == "fs"
