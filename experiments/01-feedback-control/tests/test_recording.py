from pathlib import Path

import pytest

from feedback_control.config import load_experiment_config
from feedback_control.playback import PlaybackFrame
from feedback_control.recording import RecordingOptions, sample_playback_frame


CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "demo.toml"


def _frame(
    *,
    time_s: float,
    position: float,
    velocity: float,
    torque: float,
    saturated: bool,
) -> PlaybackFrame:
    return PlaybackFrame(
        time_s=time_s,
        position_rad=position,
        velocity_rad_s=velocity,
        target_position_rad=1.0,
        applied_torque_nm=torque,
        saturated=saturated,
    )


def test_video_sampling_interpolates_state_and_holds_action() -> None:
    frames = (
        _frame(
            time_s=0.0,
            position=0.0,
            velocity=2.0,
            torque=3.0,
            saturated=True,
        ),
        _frame(
            time_s=0.1,
            position=1.0,
            velocity=0.0,
            torque=-1.0,
            saturated=False,
        ),
    )

    sampled = sample_playback_frame(frames, 0.025)

    assert sampled.time_s == pytest.approx(0.025)
    assert sampled.position_rad == pytest.approx(0.25)
    assert sampled.velocity_rad_s == pytest.approx(1.5)
    assert sampled.applied_torque_nm == 3.0
    assert sampled.saturated is True


def test_video_sampling_clamps_to_recorded_endpoints() -> None:
    frames = (
        _frame(
            time_s=0.0,
            position=0.0,
            velocity=0.0,
            torque=1.0,
            saturated=False,
        ),
        _frame(
            time_s=1.0,
            position=2.0,
            velocity=0.0,
            torque=0.0,
            saturated=False,
        ),
    )

    assert sample_playback_frame(frames, -1.0) == frames[0]
    assert sample_playback_frame(frames, 2.0) == frames[-1]


def test_recording_defaults_match_presentation_contract() -> None:
    config = load_experiment_config(CONFIG_PATH)
    options = RecordingOptions()

    assert options.fps == 30
    assert (options.width, options.height) == (1280, 720)
    assert config.episode.duration_s == 4.0
