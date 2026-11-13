import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "experiments/04-learned-dynamics/scripts/smoke_predictor.py"


def test_smoke_cli_help_and_cpu_from_different_directory(tmp_path):
    help_result = subprocess.run([sys.executable, str(SCRIPT), "--help"], cwd=tmp_path,
                                 capture_output=True, text=True)
    assert help_result.returncode == 0
    result = subprocess.run([sys.executable, str(SCRIPT), "--device", "cpu"], cwd=tmp_path,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    assert data["device"] == "cpu"
    assert data["fit_rows"] == 940
    assert data["scope"] == "synthetic_operation_check"
    assert len(data["predicted_state"]) == 2


def test_smoke_cli_on_actual_mps(tmp_path):
    import torch
    if not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable; smoke CLI does not establish GPU behavior")
    result = subprocess.run([sys.executable, str(SCRIPT), "--device", "mps"], cwd=tmp_path,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["device"] == "mps"
