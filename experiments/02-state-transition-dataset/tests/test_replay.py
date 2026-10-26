from dataclasses import replace
from pathlib import Path

import pytest

from feedback_control.report_evidence import load_evaluation_evidence
from feedback_control.trajectory import StateRecord
from state_transition_dataset.builder import build_episode_manifests, build_rows
from state_transition_dataset.config import load_dataset_config
from state_transition_dataset.replay import verify_action_replay


ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = (
    ROOT / "experiments/02-state-transition-dataset/configs/dataset-v1.toml"
)


def test_replays_recorded_applied_actions_within_exact_tolerance() -> None:
    config = load_dataset_config(CONFIG_PATH)
    evidence = load_evaluation_evidence(
        config.source_evaluation_dir,
        config.source_evaluation_config,
    )

    result = verify_action_replay(
        evidence.episodes[0].trajectory,
        evidence.config.model.path,
        config.replay_absolute_tolerance,
    )

    assert result.max_position_error_rad <= config.replay_absolute_tolerance
    assert result.max_velocity_error_rad_s <= config.replay_absolute_tolerance


def test_replay_uses_applied_action_not_raw_controller_output() -> None:
    config = load_dataset_config(CONFIG_PATH)
    evidence = load_evaluation_evidence(
        config.source_evaluation_dir,
        config.source_evaluation_config,
    )
    trajectory = evidence.episodes[0].trajectory
    first = replace(trajectory.transitions[0], raw_torque_nm=0.0)
    changed_raw = replace(
        trajectory,
        transitions=(first,) + trajectory.transitions[1:],
    )

    original = verify_action_replay(
        trajectory,
        evidence.config.model.path,
        config.replay_absolute_tolerance,
    )
    changed = verify_action_replay(
        changed_raw,
        evidence.config.model.path,
        config.replay_absolute_tolerance,
    )

    assert changed == original


def test_rejects_model_checksum_mismatch() -> None:
    config = load_dataset_config(CONFIG_PATH)
    evidence = load_evaluation_evidence(
        config.source_evaluation_dir,
        config.source_evaluation_config,
    )
    trajectory = replace(
        evidence.episodes[0].trajectory,
        environment_version="0" * 64,
    )

    with pytest.raises(ValueError, match="model checksum mismatch"):
        verify_action_replay(
            trajectory,
            evidence.config.model.path,
            config.replay_absolute_tolerance,
        )


@pytest.mark.parametrize("tolerance", [0.0, -1.0, float("nan"), float("inf")])
def test_rejects_invalid_replay_tolerance(tolerance: float) -> None:
    config = load_dataset_config(CONFIG_PATH)
    evidence = load_evaluation_evidence(
        config.source_evaluation_dir,
        config.source_evaluation_config,
    )

    with pytest.raises(
        ValueError,
        match="absolute_tolerance must be finite and greater than zero",
    ):
        verify_action_replay(
            evidence.episodes[0].trajectory,
            evidence.config.model.path,
            tolerance,
        )


def test_rejects_saved_next_state_outside_tolerance() -> None:
    config = load_dataset_config(CONFIG_PATH)
    evidence = load_evaluation_evidence(
        config.source_evaluation_dir,
        config.source_evaluation_config,
    )
    trajectory = evidence.episodes[0].trajectory
    transition = trajectory.transitions[0]
    changed_next_state = StateRecord(
        position_rad=transition.next_state.position_rad + 1e-6,
        velocity_rad_s=transition.next_state.velocity_rad_s,
    )
    changed = replace(
        transition,
        next_state=changed_next_state,
    )
    tampered = replace(
        trajectory,
        transitions=(changed,) + trajectory.transitions[1:],
    )

    with pytest.raises(ValueError, match="replay state error exceeds tolerance"):
        verify_action_replay(
            tampered,
            evidence.config.model.path,
            config.replay_absolute_tolerance,
        )


def test_builds_replay_verified_manifests_for_every_episode() -> None:
    config = load_dataset_config(CONFIG_PATH)
    build = build_rows(config)

    manifests = build_episode_manifests(
        build,
        config.replay_absolute_tolerance,
    )

    assert len(manifests) == len(build.episodes) == 8
    assert {manifest.episode_id for manifest in manifests} == {
        episode.episode_id for episode in build.episodes
    }
    assert all(manifest.replay_verified for manifest in manifests)
    assert all(
        manifest.max_position_error_rad <= config.replay_absolute_tolerance
        and manifest.max_velocity_error_rad_s
        <= config.replay_absolute_tolerance
        for manifest in manifests
    )
