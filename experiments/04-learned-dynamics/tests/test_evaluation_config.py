from pathlib import Path

import pytest

from learned_dynamics.evaluation import BASELINE_IDS
from learned_dynamics.evaluation_config import load_evaluation_config


def write_config(path: Path, **replacements) -> Path:
    text = '''schema_version = 1
dataset_dir = "dataset"
dataset_version = "experiment-02-transition-v1"
training_roster_dir = "training-results"
output_dir = "evaluation-results"
artifact_budget_bytes = 524288000
horizons = [1, 5, 10, 25, 50, 100]
memberships = ["all", "history_common", "loss_common"]
baseline_ids = ["mujoco", "no_change", "constant_velocity", "analytical_rk4_0.002", "analytical_rk4_0.001", "analytical_rk4_0.0005"]
prediction_tolerance = 1e-12
'''
    for old, new in replacements.items():
        text = text.replace(old, new)
    path.write_text(text)
    return path


def test_loads_frozen_stage4_protocol(tmp_path):
    path = write_config(tmp_path / "evaluation.toml")
    config = load_evaluation_config(path)
    assert config.horizons == (1, 5, 10, 25, 50, 100)
    assert config.memberships == ("all", "history_common", "loss_common")
    assert config.baseline_ids == BASELINE_IDS
    assert config.config_sha256


@pytest.mark.parametrize("old,new", [
    ("schema_version = 1", "schema_version = true"),
    ("horizons = [1, 5, 10, 25, 50, 100]", "horizons = [1, 5]"),
    ("memberships = [\"all\", \"history_common\", \"loss_common\"]", "memberships = [\"all\"]"),
    ("baseline_ids = [", "unknown = 1\nbaseline_ids = ["),
    ('dataset_dir = "dataset"', 'dataset_dir = "/tmp/dataset"'),
    ("prediction_tolerance = 1e-12", "prediction_tolerance = 0.0"),
])
def test_rejects_protocol_drift_and_unsafe_paths(tmp_path, old, new):
    path = write_config(tmp_path / "evaluation.toml", **{old: new})
    with pytest.raises(ValueError):
        load_evaluation_config(path)


@pytest.mark.parametrize("output", ["dataset/nested", "training-results/nested", "."])
def test_rejects_output_overlapping_immutable_inputs(tmp_path, output):
    path = write_config(tmp_path / "evaluation.toml", **{
        'output_dir = "evaluation-results"': f'output_dir = "{output}"'})
    with pytest.raises(ValueError, match="overlap"):
        load_evaluation_config(path)
