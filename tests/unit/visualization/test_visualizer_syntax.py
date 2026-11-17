"""Unit tests for the Mermaid `syntax` option of the Visualizer."""

import re
from pathlib import Path

import pytest

from modularml.visualization.visualizer.visualizer import Visualizer
from tests.conftest import generate_dummy_featureset

_NODE_DECL = r"^\s*(n\d+)(?:@\{|\[|\(\[|\(\()"
_EDGE = r"-->|-\.->|==>"


@pytest.fixture
def featureset_viz():
    """Visualizer over a FeatureSet with two splits (so the diagram has edges)."""
    fs = generate_dummy_featureset(
        n_samples=20,
        feature_shape_map={"X": (3,)},
        target_shape_map={"Y": (1,)},
        tag_type_map={"group": "str"},
        label="VizFS",
    )
    fs.split_random(ratios={"train": 0.7, "test": 0.3}, seed=0)
    return Visualizer(fs)


def _has_v11_only_syntax(text: str) -> bool:
    """Node declarations like `n0@{ ... }` and edge ids like `e0@-->`."""
    return "@{" in text or re.search(r"\w@-", text) is not None


@pytest.mark.unit
def test_default_syntax_is_v11(featureset_viz):
    """Test the default output is unchanged (Mermaid 11 syntax)."""
    default = featureset_viz.get_mermaid_str()
    assert default == featureset_viz.get_mermaid_str(syntax="v11")
    assert "@{ label:" in default
    assert _has_v11_only_syntax(default)


@pytest.mark.unit
def test_classic_syntax_has_no_v11_only_constructs(featureset_viz):
    """Test classic output avoids the syntax that older Mermaid cannot parse."""
    classic = featureset_viz.get_mermaid_str(syntax="classic")

    assert classic.startswith("flowchart LR")
    assert not _has_v11_only_syntax(classic)
    # Classic node declarations, e.g. n0["label"]
    assert re.search(r'^\s*n\d+\["', classic, flags=re.MULTILINE)


@pytest.mark.unit
def test_classic_syntax_keeps_all_nodes_and_edges(featureset_viz):
    """Test switching syntax changes how the diagram is written, not its content."""
    v11 = featureset_viz.get_mermaid_str(syntax="v11")
    classic = featureset_viz.get_mermaid_str(syntax="classic")

    v11_nodes = set(re.findall(_NODE_DECL, v11, flags=re.MULTILINE))
    classic_nodes = set(re.findall(_NODE_DECL, classic, flags=re.MULTILINE))
    assert classic_nodes == v11_nodes
    assert len(classic_nodes) > 0

    n_edges_v11 = len(re.findall(_EDGE, v11))
    assert len(re.findall(_EDGE, classic)) == n_edges_v11
    assert n_edges_v11 > 0

    # Node styling is kept
    assert "classDef" in classic
    assert re.search(r"^\s*class n\d+ \w+", classic, flags=re.MULTILINE)


@pytest.mark.unit
def test_unknown_syntax_raises(featureset_viz):
    """Test an unsupported syntax name raises a clear error."""
    with pytest.raises(ValueError, match="syntax"):
        featureset_viz.get_mermaid_str(syntax="mermaid-9000")


@pytest.mark.unit
def test_mermaid_to_mmd_supports_classic_syntax(featureset_viz, tmp_path):
    """Test the syntax option is passed through when writing a .mmd file."""
    path = featureset_viz.mermaid_to_mmd(tmp_path / "diagram", syntax="classic")

    written = Path(path).read_text(encoding="utf-8")
    assert written == featureset_viz.get_mermaid_str(syntax="classic")
