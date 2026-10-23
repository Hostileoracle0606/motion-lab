from pathlib import Path

import pytest

from feedback_control.config import load_evaluation_config


CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "evaluation-v1.toml"
MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "one_link.xml"


def _write_config(tmp_path: Path, text: str) -> Path:
    config_path = tmp_path / "evaluation.toml"
    config_path.write_text(
        text.replace("../models/one_link.xml", str(MODEL_PATH)),
        encoding="utf-8",
    )
    return config_path


def test_evaluation_config_declares_shared_scenarios_in_both_directions() -> None:
    config = load_evaluation_config(CONFIG_PATH)

    directions = {
        "positive"
        if scenario.target_position_rad > scenario.start_position_rad
        else "negative"
        for scenario in config.scenarios
    }

    assert config.schema_version == 1
    assert config.seeds == (0,)
    assert len(config.scenarios) == 4
    assert directions == {"positive", "negative"}
    assert config.p.kp == 2.0
    assert config.pd.kp == 2.0
    assert config.pd.kd == 1.3
    assert config.episode.duration_s == 4.0
    assert config.episode.control_interval_s == 0.01
    assert config.episode.torque_limit_nm == 3.0
    assert config.metrics.rise_lower_fraction == 0.1
    assert config.metrics.rise_upper_fraction == 0.9
    assert config.metrics.settling_position_tolerance_rad == 0.05
    assert config.metrics.settling_velocity_tolerance_rad_s == 0.1
    assert config.metrics.steady_state_window_s == 0.5
    assert config.noise.position_std_rad == 0.0
    assert config.noise.velocity_std_rad_s == 0.0
    assert config.noise.action_std_nm == 0.0
    assert config.failure_rules.fail_on_non_finite_state
    assert config.failure_rules.fail_on_joint_limit_violation
    assert config.failure_rules.fail_if_not_settled


def test_evaluation_config_rejects_degenerate_scenario(tmp_path: Path) -> None:
    invalid = CONFIG_PATH.read_text(encoding="utf-8").replace(
        "target_position_rad = 1.047198",
        "target_position_rad = -1.047198",
        1,
    )

    with pytest.raises(ValueError, match="command magnitude"):
        load_evaluation_config(_write_config(tmp_path, invalid))


def test_evaluation_config_rejects_scenario_outside_joint_range(
    tmp_path: Path,
) -> None:
    invalid = CONFIG_PATH.read_text(encoding="utf-8").replace(
        "target_position_rad = 1.047198",
        "target_position_rad = 1.7",
        1,
    )

    with pytest.raises(ValueError, match="outside the model joint range"):
        load_evaluation_config(_write_config(tmp_path, invalid))


def test_evaluation_config_rejects_duplicate_scenario_ids(tmp_path: Path) -> None:
    invalid = CONFIG_PATH.read_text(encoding="utf-8").replace(
        'id = "positive-to-negative-wide"',
        'id = "negative-to-positive-wide"',
    )

    with pytest.raises(ValueError, match="scenario ids must be unique"):
        load_evaluation_config(_write_config(tmp_path, invalid))


def test_evaluation_config_rejects_empty_scenario_id(tmp_path: Path) -> None:
    invalid = CONFIG_PATH.read_text(encoding="utf-8").replace(
        'id = "negative-to-positive-wide"',
        'id = ""',
    )

    with pytest.raises(ValueError, match="scenario id must not be empty"):
        load_evaluation_config(_write_config(tmp_path, invalid))


def test_evaluation_config_rejects_duplicate_start_target_pairs(
    tmp_path: Path,
) -> None:
    invalid = CONFIG_PATH.read_text(encoding="utf-8").replace(
        "start_position_rad = -0.523599\ntarget_position_rad = 0.523599",
        "start_position_rad = -1.047198\ntarget_position_rad = 1.047198",
    )

    with pytest.raises(ValueError, match="start-target pairs must be unique"):
        load_evaluation_config(_write_config(tmp_path, invalid))


def test_evaluation_config_requires_both_command_directions(tmp_path: Path) -> None:
    sections = CONFIG_PATH.read_text(encoding="utf-8").split("[[scenarios]]")
    only_first_positive_scenario = sections[0] + "[[scenarios]]" + sections[1]

    with pytest.raises(ValueError, match="positive and negative command directions"):
        load_evaluation_config(_write_config(tmp_path, only_first_positive_scenario))


def test_evaluation_config_rejects_inconsistent_control_clock(tmp_path: Path) -> None:
    invalid = CONFIG_PATH.read_text(encoding="utf-8").replace(
        "control_interval_s = 0.010",
        "control_interval_s = 0.009",
    )

    with pytest.raises(ValueError, match="control interval / simulation timestep"):
        load_evaluation_config(_write_config(tmp_path, invalid))


def test_evaluation_config_rejects_actuator_limit_mismatch(tmp_path: Path) -> None:
    invalid = CONFIG_PATH.read_text(encoding="utf-8").replace(
        "torque_limit_nm = 3.0",
        "torque_limit_nm = 2.0",
    )

    with pytest.raises(ValueError, match="actuator limits"):
        load_evaluation_config(_write_config(tmp_path, invalid))


@pytest.mark.parametrize(
    ("declared", "invalid_value", "message"),
    [
        (
            'keyframe = "start"',
            'keyframe = "missing"',
            "unknown model keyframe",
        ),
        (
            "simulation_timestep_s = 0.002",
            "simulation_timestep_s = 0.001",
            "MuJoCo timestep",
        ),
    ],
)
def test_evaluation_config_rejects_model_contract_mismatch(
    tmp_path: Path,
    declared: str,
    invalid_value: str,
    message: str,
) -> None:
    invalid = CONFIG_PATH.read_text(encoding="utf-8").replace(
        declared,
        invalid_value,
    )

    with pytest.raises(ValueError, match=message):
        load_evaluation_config(_write_config(tmp_path, invalid))


def test_evaluation_config_rejects_inconsistent_rise_thresholds(
    tmp_path: Path,
) -> None:
    invalid = CONFIG_PATH.read_text(encoding="utf-8").replace(
        "rise_lower_fraction = 0.10",
        "rise_lower_fraction = 0.95",
    )

    with pytest.raises(ValueError, match="rise thresholds"):
        load_evaluation_config(_write_config(tmp_path, invalid))


def test_evaluation_config_rejects_partial_sample_steady_state_window(
    tmp_path: Path,
) -> None:
    invalid = CONFIG_PATH.read_text(encoding="utf-8").replace(
        "steady_state_window_s = 0.50",
        "steady_state_window_s = 0.505",
    )

    with pytest.raises(ValueError, match="steady-state window / control interval"):
        load_evaluation_config(_write_config(tmp_path, invalid))


def test_evaluation_config_requires_at_least_one_seed(tmp_path: Path) -> None:
    invalid = CONFIG_PATH.read_text(encoding="utf-8").replace(
        "seeds = [0]",
        "seeds = []",
    )

    with pytest.raises(ValueError, match="at least one seed"):
        load_evaluation_config(_write_config(tmp_path, invalid))


@pytest.mark.parametrize(
    ("declared", "invalid_value", "message"),
    [
        (
            "schema_version = 1",
            "schema_version = 1.5",
            "schema_version must be an integer",
        ),
        ("seeds = [0]", "seeds = [0.5]", "seed must be an integer"),
    ],
)
def test_evaluation_config_requires_integer_version_and_seeds(
    tmp_path: Path,
    declared: str,
    invalid_value: str,
    message: str,
) -> None:
    invalid = CONFIG_PATH.read_text(encoding="utf-8").replace(
        declared,
        invalid_value,
    )

    with pytest.raises(ValueError, match=message):
        load_evaluation_config(_write_config(tmp_path, invalid))


def test_evaluation_config_rejects_unsupported_schema_version(
    tmp_path: Path,
) -> None:
    invalid = CONFIG_PATH.read_text(encoding="utf-8").replace(
        "schema_version = 1",
        "schema_version = 2",
    )

    with pytest.raises(ValueError, match="unsupported evaluation schema version"):
        load_evaluation_config(_write_config(tmp_path, invalid))


def test_evaluation_config_requires_boolean_failure_rules(tmp_path: Path) -> None:
    invalid = CONFIG_PATH.read_text(encoding="utf-8").replace(
        "fail_if_not_settled = true",
        'fail_if_not_settled = "false"',
    )

    with pytest.raises(ValueError, match="fail_if_not_settled must be a boolean"):
        load_evaluation_config(_write_config(tmp_path, invalid))


def test_evaluation_config_rejects_non_finite_scenario_state(
    tmp_path: Path,
) -> None:
    invalid = CONFIG_PATH.read_text(encoding="utf-8").replace(
        "initial_velocity_rad_s = 0.0",
        "initial_velocity_rad_s = nan",
        1,
    )

    with pytest.raises(ValueError, match="initial_velocity_rad_s must be finite"):
        load_evaluation_config(_write_config(tmp_path, invalid))


@pytest.mark.parametrize(
    ("declared", "invalid_value", "message"),
    [
        ("kp = 2.0", "kp = -1.0", "p.kp must be non-negative"),
        (
            "settling_position_tolerance_rad = 0.05",
            "settling_position_tolerance_rad = 0.0",
            "settling_position_tolerance_rad must be greater than zero",
        ),
        (
            "steady_state_window_s = 0.50",
            "steady_state_window_s = 5.0",
            "steady_state_window_s must not exceed duration_s",
        ),
        (
            "minimum_command_magnitude_rad = 0.000001",
            "minimum_command_magnitude_rad = 0.0",
            "minimum_command_magnitude_rad must be greater than zero",
        ),
        (
            "position_std_rad = 0.0",
            "position_std_rad = -0.1",
            "position_std_rad must be non-negative",
        ),
        (
            "target_velocity_rad_s = 0.0",
            "target_velocity_rad_s = nan",
            "target_velocity_rad_s must be finite",
        ),
    ],
)
def test_evaluation_config_rejects_invalid_declared_values(
    tmp_path: Path,
    declared: str,
    invalid_value: str,
    message: str,
) -> None:
    invalid = CONFIG_PATH.read_text(encoding="utf-8").replace(
        declared,
        invalid_value,
        1,
    )

    with pytest.raises(ValueError, match=message):
        load_evaluation_config(_write_config(tmp_path, invalid))
