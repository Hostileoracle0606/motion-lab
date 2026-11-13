from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "experiments/04-learned-dynamics/scripts/smoke_predictor.py"
TRAIN_SCRIPT = ROOT / "experiments/04-learned-dynamics/scripts/train.py"


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


def test_training_cli_help_and_short_configured_cpu_run(tmp_path, committed_config_writer):
    help_result = subprocess.run([sys.executable, str(TRAIN_SCRIPT), "--help"], cwd=tmp_path,
                                 capture_output=True, text=True)
    assert help_result.returncode == 0
    config = committed_config_writer(tmp_path / "training.toml", epochs=1, seeds=(0,))
    result = subprocess.run([sys.executable, str(TRAIN_SCRIPT), "--config", str(config),
                             "--output", str(tmp_path / "results")], cwd=tmp_path,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["planned_runs"] == 1
    assert payload["successful_runs"] == 1
    assert (tmp_path / "results/run-manifest.json").is_file()
    curve = json.loads(next((tmp_path / "results").glob("*/learning-curve.json")).read_text())
    assert curve["epochs"][0]["batch_sizes"][-1] == 44
    assert curve["epochs"][0]["updates_this_epoch"] == 15
    repeated = subprocess.run([sys.executable, str(TRAIN_SCRIPT), "--config", str(config),
                               "--output", str(tmp_path / "results")], cwd=tmp_path,
                              capture_output=True, text=True)
    assert repeated.returncode != 0
    assert "already exists" in repeated.stderr


def test_short_synthetic_fit_on_actual_mps(tmp_path, training_config_writer, synthetic_views):
    import torch
    if not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable; training smoke does not establish GPU behavior")
    from learned_dynamics.config import load_training_config
    from learned_dynamics.training import fit_model
    config = load_training_config(training_config_writer(tmp_path / "training.toml", epochs=1, seeds=(0,)))
    run = replace(config.runs[0], device="mps")
    outcome = fit_model(config, run, synthetic_views[0], synthetic_views[1])
    assert outcome.device == "mps"
    assert np.isfinite(outcome.selected_validation_mse)
