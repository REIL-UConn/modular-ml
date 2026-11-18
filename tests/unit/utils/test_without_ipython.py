"""ModularML imports and draws diagrams without IPython (it is not a dependency)."""

import subprocess
import sys

import pytest

_CODE = """
import sys

class _BlockIPython:
    def find_spec(self, name, path=None, target=None):
        if name == "IPython" or name.startswith("IPython."):
            raise ImportError("IPython blocked for this test")
        return None

sys.meta_path.insert(0, _BlockIPython())
import numpy as np
import modularml
from modularml import Experiment, FeatureSet
from modularml.visualization.visualizer.visualizer import Visualizer

Experiment("no_ipython", registration_policy="overwrite")
fs = FeatureSet.from_dict(
    "fs", {"x": np.arange(4.0), "y": np.arange(4)}, feature_keys="x", target_keys="y",
)
print("flowchart" in Visualizer(fs).get_mermaid_str())
"""


@pytest.mark.unit
def test_modularml_works_without_ipython():
    out = subprocess.run(  # noqa: S603 - fixed code string, current interpreter
        [sys.executable, "-c", _CODE],
        capture_output=True,
        text=True,
        check=False,
    )
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip().splitlines()[-1] == "True"
