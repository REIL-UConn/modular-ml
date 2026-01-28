from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modularml.core.topology.model_graph import ModelGraph


@contextmanager
def preserve_frozen_state(model_graph: ModelGraph):
    """Context manager that restores frozen nodes after mutation."""
    # Node IDs of frozen nodes prior to context execution
    frozen_before = set(model_graph.frozen_nodes.keys())
    try:
        yield
    finally:
        # Node IDs of frozen nodes after context execution
        frozen_after = set(model_graph.frozen_nodes.keys())
        if frozen_after != frozen_before:
            model_graph.unfreeze(nodes=frozen_after)
            model_graph.freeze(nodes=frozen_before)
