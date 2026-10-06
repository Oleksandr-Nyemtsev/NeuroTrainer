"""Fresh interpreters: cached torch imports cannot conceal DLL order regressions."""
from pathlib import Path
import subprocess
import sys
import pytest


@pytest.mark.parametrize("prefix", [
    "import numpy as np; np.dot(np.ones((8, 8)), np.ones((8, 8)))",
    "import runpy; runpy.run_path('tests/test_alpha_tracker.py')",
    "import torch; import numpy as np; np.dot(np.ones((8, 8)), np.ones((8, 8)))",
])
def test_computation_then_torch_in_fresh_process(prefix):
    code = prefix + """
import torch
import pyriemann
import mne
import brainflow
import sounddevice
from brainflow.board_shim import BoardShim, BoardIds
assert BoardShim.get_sampling_rate(BoardIds.MUSE_2_BOARD.value) == 256
assert torch.equal(torch.eye(2) @ torch.ones(2, 2), torch.ones(2, 2))
print('native order OK')
"""
    result = subprocess.run([sys.executable, "-B", "-c", code],
                            cwd=Path(__file__).resolve().parents[1],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "native order OK" in result.stdout
