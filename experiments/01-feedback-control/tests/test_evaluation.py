from dataclasses import replace
from pathlib import Path

import pytest

from feedback_control.config import MetricThresholds, load_experiment_config
from feedback_control.evaluation import MetricAvailability, evaluate_episode
from feedback_control.metrics import TimingOutcome
from feedback_control.simulation import EpisodeResult, TrajectorySample, run_episode


CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "demo.toml"
TEST_THRESHOLDS = MetricThresholds(
    rise_lower_fraction=0.1,
    rise_upper_fraction=0.9,
    settling_position_tolerance_rad=0.1,
    settling_velocity_tolerance_rad_s=0.25,
    steady_state_window_s=0.01,
    minimum_command_magnitude_rad=0.01,
)


def test_episode_evaluation_reports_all_declared_metrics() -> None:
    positions = (0.0, 0.2, 0.9, 1.05)
    velocities = (0.0, 1.0, 0.2, 0.0)
    torques = (1.0, 1.0, 0.5, 0.0)
    result = EpisodeResult(
        controller="pd",
        samples=tuple(
            TrajectorySample(
                step_index=index,
                time_s=index * 0.1,
                position_rad=position,
                velocity_rad_s=velocity,
                target_position_rad=1.0,
                target_velocity_rad_s=0.0,
                raw_torque_nm=torque,
                applied_torque_nm=torque,
                saturated=abs(torque) == 1.0,
            )
            for index, (position, velocity, torque) in enumerate(
                zip(positions, velocities, torques, strict=True)
            )
        ),
        final_time_s=0.4,
        final_position_rad=1.05,
        final_velocity_rad_s=0.0,
    )
    thresholds = MetricThresholds(
        rise_lower_fraction=0.1,
        rise_upper_fraction=0.9,
        settling_position_tolerance_rad=0.1,
        settling_velocity_tolerance_rad_s=0.25,
        steady_state_window_s=0.2,
        minimum_command_magnitude_rad=0.01,
    )

    metrics = evaluate_episode(
        result,
        control_interval_s=0.1,
        thresholds=thresholds,
        torque_limit_nm=1.0,
    )

    assert metrics.integrated_absolute_error_rad_s == pytest.approx(0.195)
    assert metrics.rise_time.outcome is TimingOutcome.MEASURED
    assert metrics.rise_time.time_s == pytest.approx(0.1)
    assert metrics.normalized_maximum_overshoot == pytest.approx(0.05)
    assert metrics.settling_time.outcome is TimingOutcome.MEASURED
    assert metrics.settling_time.time_s == pytest.approx(0.2)
    assert metrics.steady_state_absolute_error_rad == pytest.approx(0.075)
    assert (
        metrics.steady_state_error_availability
        is MetricAvailability.MEASURED
    )
    assert metrics.squared_control_effort_nm2_s == pytest.approx(0.225)
    assert metrics.saturation_fraction == pytest.approx(0.5)


def test_real_one_step_episode_uses_one_pre_action_sample() -> None:
    config = load_experiment_config(CONFIG_PATH)
    control_interval_s = config.episode.control_interval_s
    one_step_config = replace(
        config,
        episode=replace(config.episode, duration_s=control_interval_s),
    )
    result = run_episode(one_step_config, "p")

    metrics = evaluate_episode(
        result,
        control_interval_s=control_interval_s,
        thresholds=TEST_THRESHOLDS,
        torque_limit_nm=config.episode.torque_limit_nm,
    )

    expected_iae = abs(
        config.episode.target_position_rad - config.episode.start_position_rad
    ) * control_interval_s
    assert len(result.samples) == 1
    assert result.final_position_rad != pytest.approx(result.samples[0].position_rad)
    assert metrics.integrated_absolute_error_rad_s == pytest.approx(expected_iae)


def test_early_terminal_episode_marks_tail_metric_unavailable() -> None:
    config = load_experiment_config(CONFIG_PATH)
    one_step_config = replace(
        config,
        episode=replace(
            config.episode,
            duration_s=config.episode.control_interval_s,
        ),
    )
    result = run_episode(one_step_config, "p")
    long_tail_thresholds = replace(
        TEST_THRESHOLDS,
        steady_state_window_s=0.5,
    )

    metrics = evaluate_episode(
        result,
        control_interval_s=config.episode.control_interval_s,
        thresholds=long_tail_thresholds,
        torque_limit_nm=config.episode.torque_limit_nm,
    )

    assert metrics.steady_state_absolute_error_rad is None
    assert (
        metrics.steady_state_error_availability
        is MetricAvailability.INSUFFICIENT_DURATION
    )


def test_evaluate_episode_rejects_empty_episode() -> None:
    result = EpisodeResult(
        controller="p",
        samples=(),
        final_time_s=0.0,
        final_position_rad=0.0,
        final_velocity_rad_s=0.0,
    )

    with pytest.raises(ValueError, match="episode must contain at least one sample"):
        evaluate_episode(
            result,
            control_interval_s=0.01,
            thresholds=TEST_THRESHOLDS,
            torque_limit_nm=3.0,
        )


def test_evaluate_episode_rejects_changing_target() -> None:
    config = load_experiment_config(CONFIG_PATH)
    two_step_config = replace(
        config,
        episode=replace(
            config.episode,
            duration_s=2 * config.episode.control_interval_s,
        ),
    )
    result = run_episode(two_step_config, "p")
    first, second = result.samples
    inconsistent_result = replace(
        result,
        samples=(
            first,
            replace(second, target_position_rad=second.target_position_rad + 0.1),
        ),
    )

    with pytest.raises(ValueError, match="episode must use one target position"):
        evaluate_episode(
            inconsistent_result,
            control_interval_s=config.episode.control_interval_s,
            thresholds=TEST_THRESHOLDS,
            torque_limit_nm=config.episode.torque_limit_nm,
        )


def test_evaluate_episode_rejects_mismatched_control_interval() -> None:
    config = load_experiment_config(CONFIG_PATH)
    two_step_config = replace(
        config,
        episode=replace(
            config.episode,
            duration_s=2 * config.episode.control_interval_s,
        ),
    )
    result = run_episode(two_step_config, "p")

    with pytest.raises(
        ValueError,
        match="control_interval_s does not match the episode schedule",
    ):
        evaluate_episode(
            result,
            control_interval_s=2 * config.episode.control_interval_s,
            thresholds=TEST_THRESHOLDS,
            torque_limit_nm=config.episode.torque_limit_nm,
        )
