from dataclasses import replace
from pathlib import Path

import pytest

from feedback_control.config import ExperimentConfig, load_experiment_config
from feedback_control.simulation import ControllerName, run_episode


CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "demo.toml"


@pytest.fixture(scope="module")
def config() -> ExperimentConfig:
    return load_experiment_config(CONFIG_PATH)


def test_episode_starts_from_configured_state(config: ExperimentConfig) -> None:
    result = run_episode(config, "p")
    first = result.samples[0]

    assert first.step_index == 0
    assert first.time_s == pytest.approx(0.0)
    assert first.position_rad == pytest.approx(config.episode.start_position_rad)
    assert first.velocity_rad_s == pytest.approx(0.0)


def test_episode_uses_declared_control_schedule(config: ExperimentConfig) -> None:
    result = run_episode(config, "p")
    sample_times = [sample.time_s for sample in result.samples]

    assert len(result.samples) == 400
    assert result.final_time_s == pytest.approx(config.episode.duration_s)
    assert all(later > earlier for earlier, later in zip(sample_times, sample_times[1:]))
    assert all(
        later - earlier == pytest.approx(config.episode.control_interval_s)
        for earlier, later in zip(sample_times, sample_times[1:])
    )


@pytest.mark.parametrize("controller", ["p", "pd"])
def test_controller_moves_arm_closer_to_target(
    config: ExperimentConfig,
    controller: ControllerName,
) -> None:
    result = run_episode(config, controller)
    initial_error = abs(
        config.episode.target_position_rad - config.episode.start_position_rad
    )
    final_error = abs(config.episode.target_position_rad - result.final_position_rad)

    assert final_error < initial_error


@pytest.mark.parametrize("controller", ["p", "pd"])
def test_applied_torque_stays_within_actuator_limit(
    config: ExperimentConfig,
    controller: ControllerName,
) -> None:
    result = run_episode(config, controller)
    limit = config.episode.torque_limit_nm

    assert all(abs(sample.applied_torque_nm) <= limit for sample in result.samples)


def test_episode_is_deterministic(config: ExperimentConfig) -> None:
    short_episode = replace(
        config,
        episode=replace(config.episode, duration_s=0.1),
    )

    first_run = run_episode(short_episode, "pd")
    second_run = run_episode(short_episode, "pd")

    assert first_run == second_run


def test_sample_records_observation_before_applying_action(
    config: ExperimentConfig,
) -> None:
    short_episode = replace(
        config,
        episode=replace(config.episode, duration_s=0.02),
    )
    result = run_episode(short_episode, "p")
    first, second = result.samples
    expected_raw_torque = config.p.kp * (
        config.episode.target_position_rad - first.position_rad
    )

    assert first.raw_torque_nm == pytest.approx(expected_raw_torque)
    assert second.position_rad > first.position_rad


def test_unknown_controller_is_rejected(config: ExperimentConfig) -> None:
    with pytest.raises(ValueError, match="unknown controller"):
        run_episode(config, "pid")  # type: ignore[arg-type]


def test_non_integer_control_clock_ratio_is_rejected(
    config: ExperimentConfig,
) -> None:
    invalid_config = replace(
        config,
        episode=replace(config.episode, control_interval_s=0.009),
    )

    with pytest.raises(ValueError, match="control interval / simulation timestep"):
        run_episode(invalid_config, "p")
