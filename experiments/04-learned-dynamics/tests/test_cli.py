from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from learned_dynamics.training_evidence import load_run_manifest


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
    manifest = json.loads((tmp_path / "results/run-manifest.json").read_text())
    assert curve["training_config_sha256"] == manifest["training_config_sha256"]
    assert curve["run_spec"]["seed"] == 0
    assert curve["epochs"][0]["batch_sizes"][-1] == 44
    assert curve["epochs"][0]["updates_this_epoch"] == 15
    repeated = subprocess.run([sys.executable, str(TRAIN_SCRIPT), "--config", str(config),
                               "--output", str(tmp_path / "results")], cwd=tmp_path,
                              capture_output=True, text=True)
    assert repeated.returncode != 0
    assert "already exists" in repeated.stderr

    curve_path = next((tmp_path / "results").glob("*/learning-curve.json"))
    outer_path = tmp_path / "results/manifest.json"
    outer = json.loads(outer_path.read_text())
    import hashlib

    def repair_curve_hash():
        outer["files"][str(curve_path.relative_to(tmp_path / "results"))] = hashlib.sha256(
            curve_path.read_bytes()).hexdigest()
        outer_path.write_text(json.dumps(outer))

    original_curve = json.loads(curve_path.read_text())
    curve["epochs"][0]["batch_sizes"][0] -= 1
    curve["epochs"][0]["batch_sizes"][-1] += 1
    curve_path.write_text(json.dumps(curve))
    repair_curve_hash()
    with pytest.raises(ValueError, match="original training membership"):
        load_run_manifest(tmp_path / "results")

    curve = original_curve
    curve["selected_validation_mse"] += 1.0
    curve_path.write_text(json.dumps(curve))
    repair_curve_hash()
    with pytest.raises(ValueError, match="learning curve selected metric"):
        load_run_manifest(tmp_path / "results")


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
