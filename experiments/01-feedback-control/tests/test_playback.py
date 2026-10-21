from dataclasses import replace
from pathlib import Path

import pytest

from feedback_control.config import ExperimentConfig, load_experiment_config
from feedback_control.playback import (
    ARM_COLORS,
    PlaybackOptions,
    build_comparison_frames,
    build_playback_frames,
    controller_description,
    validate_playback_options,
)
from feedback_control.simulation import run_episode


CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "demo.toml"
COMPARISON_MODEL_PATH = (
    Path(__file__).resolve().parents[1] / "models" / "one_link_comparison.xml"
)


@pytest.fixture(scope="module")
def config() -> ExperimentConfig:
    return load_experiment_config(CONFIG_PATH)


def test_playback_frames_are_derived_from_recorded_episode(
    config: ExperimentConfig,
) -> None:
    short_config = replace(
        config,
        episode=replace(config.episode, duration_s=0.02),
    )
    result = run_episode(short_config, "pd")

    frames = build_playback_frames(result)

    assert len(frames) == len(result.samples) + 1
    assert frames[0].position_rad == result.samples[0].position_rad
    assert frames[0].applied_torque_nm == result.samples[0].applied_torque_nm
    assert frames[-1].time_s == result.final_time_s
    assert frames[-1].position_rad == result.final_position_rad
    assert frames[-1].velocity_rad_s == result.final_velocity_rad_s
    assert frames[-1].applied_torque_nm == 0.0


def test_comparison_pairs_p_and_pd_on_the_same_timeline(
    config: ExperimentConfig,
) -> None:
    short_config = replace(
        config,
        episode=replace(config.episode, duration_s=0.02),
    )
    p_result = run_episode(short_config, "p")
    pd_result = run_episode(short_config, "pd")

    pairs = build_comparison_frames(p_result, pd_result)

    assert len(pairs) == len(p_result.samples) + 1
    assert all(p_frame.time_s == pd_frame.time_s for p_frame, pd_frame in pairs)
    assert pairs[0][0].position_rad == pairs[0][1].position_rad


def test_comparison_rejects_results_in_the_wrong_order(
    config: ExperimentConfig,
) -> None:
    pd_result = run_episode(
        replace(config, episode=replace(config.episode, duration_s=0.01)),
        "pd",
    )

    with pytest.raises(ValueError, match="P results followed by PD results"):
        build_comparison_frames(pd_result, pd_result)


@pytest.mark.parametrize(
    ("options", "message"),
    [
        (PlaybackOptions(speed=0.0), "speed must be greater than zero"),
        (PlaybackOptions(intro_seconds=-1.0), "intro_seconds must be non-negative"),
        (PlaybackOptions(hold_seconds=-1.0), "hold_seconds must be non-negative"),
        (PlaybackOptions(speed=float("nan")), "speed must be finite"),
    ],
)
def test_invalid_playback_timing_is_rejected(
    options: PlaybackOptions,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_playback_options(options)


def test_controller_labels_explain_the_visual_difference(
    config: ExperimentConfig,
) -> None:
    p_title, p_lesson = controller_description(config, "p")
    pd_title, pd_lesson = controller_description(config, "pd")

    assert "Kp=2.00" in p_title
    assert "position error only" in p_lesson
    assert "Kp=2.00" in pd_title
    assert "Kd=1.30" in pd_title
    assert "opposes velocity" in pd_lesson
    assert ARM_COLORS["p"] != ARM_COLORS["pd"]
