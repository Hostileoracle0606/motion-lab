from dataclasses import replace
from pathlib import Path

import pytest

from feedback_control.config import load_experiment_config
from feedback_control.evaluation import evaluate_episode
from feedback_control.simulation import EpisodeResult, run_episode


CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "demo.toml"


def test_real_one_step_episode_uses_one_pre_action_sample() -> None:
    config = load_experiment_config(CONFIG_PATH)
    control_interval_s = config.episode.control_interval_s
    one_step_config = replace(
        config,
        episode=replace(config.episode, duration_s=control_interval_s),
    )
    result = run_episode(one_step_config, "p")

    metrics = evaluate_episode(result, control_interval_s=control_interval_s)

    expected_iae = abs(
        config.episode.target_position_rad - config.episode.start_position_rad
    ) * control_interval_s
    assert len(result.samples) == 1
    assert result.final_position_rad != pytest.approx(result.samples[0].position_rad)
    assert metrics.integrated_absolute_error_rad_s == pytest.approx(expected_iae)


def test_evaluate_episode_rejects_empty_episode() -> None:
    result = EpisodeResult(
        controller="p",
        samples=(),
        final_time_s=0.0,
        final_position_rad=0.0,
        final_velocity_rad_s=0.0,
    )

    with pytest.raises(ValueError, match="episode must contain at least one sample"):
        evaluate_episode(result, control_interval_s=0.01)


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
        )
