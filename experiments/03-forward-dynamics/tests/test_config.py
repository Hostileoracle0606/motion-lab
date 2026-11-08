from pathlib import Path

import pytest

from forward_dynamics.config import load_evaluation_config


ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "experiments/03-forward-dynamics/configs/evaluation-v1.toml"


def test_config_resolves_inputs_from_file_not_current_directory(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    config = load_evaluation_config(CONFIG)
    assert config.dataset_dir == ROOT / "experiments/02-state-transition-dataset/data/dataset-v1"
    assert config.model_path == ROOT / "experiments/01-feedback-control/models/one_link.xml"
    assert config.predictor_ids == ("mujoco", "no_change", "constant_velocity", "analytical_rk4_0.002", "analytical_rk4_0.001", "analytical_rk4_0.0005")
    assert config.horizons == (1, 5, 10, 25, 50, 100)
    assert config.analytical_timesteps_s == (0.002, 0.001, 0.0005)
    assert config.scope == "full"


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("horizons", "[true, 5]"), ("horizons", "[1.0, 5]"),
        ("horizons", "[1, 1]"), ("horizons", "[5, 1]"), ("horizons", "[0, 1]"),
        ("predictor_ids", '["mujoco", "mujoco"]'), ("predictor_ids", '["unknown"]'),
        ("predictor_ids", '["mujoco", "no_change", "constant_velocity"]'),
        ("horizons", "[1, 5]"), ("analytical_timesteps_s", "[0.002, 0.001]"),
        ("predictor_ids", "[]"), ("schema_version", "true"), ("schema_version", "2"),
        ("scope", '"unknown"'), ("simulator_absolute_tolerance", "nan"),
        ("simulator_absolute_tolerance", "true"), ("simulator_absolute_tolerance", "-1.0"),
        ("analytical_timesteps_s", "[0.002, 0.002]"), ("analytical_timesteps_s", "[false]"),
        ("analytical_timesteps_s", "[1e100]"),
        ("environment_version", '"' + "0" * 64 + '"'),
    ],
)
def test_rejects_ambiguous_or_incompatible_protocol_values(tmp_path, key, value) -> None:
    lines = CONFIG.read_text().splitlines()
    # Use absolute source paths so this fixture changes only the tested field.
    lines = [
        'dataset_dir = "' + str(ROOT / "experiments/02-state-transition-dataset/data/dataset-v1") + '"'
        if line.startswith("dataset_dir =") else
        'model_path = "' + str(ROOT / "experiments/01-feedback-control/models/one_link.xml") + '"'
        if line.startswith("model_path =") else line
        for line in lines
    ]
    lines = [f"{key} = {value}" if line.startswith(key + " =") else line for line in lines]
    path = tmp_path / "invalid.toml"
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError):
        load_evaluation_config(path)
