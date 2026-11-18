"""TensorFlow is imported only when something uses it."""

import subprocess
import sys

import numpy as np
import pytest


@pytest.mark.unit
def test_import_modularml_does_not_import_tensorflow():
    code = (
        "import sys\n"
        "import modularml\n"
        "from modularml import Experiment, FeatureSet, FitPhase, EvalPhase\n"
        "from modularml.utils.data.conversion import to_numpy\n"
        "to_numpy([1.0, 2.0])\n"
        "print('tensorflow' in sys.modules)\n"
    )
    out = subprocess.run(  # noqa: S603 - fixed code string, current interpreter
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.strip().splitlines()[-1] == "False"


@pytest.mark.unit
def test_tensorflow_format_still_available():
    pytest.importorskip("tensorflow")
    from modularml.utils.data.conversion import convert_to_format
    from modularml.utils.data.data_format import DataFormat
    from modularml.utils.nn.backend import Backend, infer_backend

    t = convert_to_format(np.ones((2, 3)), fmt=DataFormat.TENSORFLOW)
    assert infer_backend(t) == Backend.TENSORFLOW
    assert tuple(t.shape) == (2, 3)
