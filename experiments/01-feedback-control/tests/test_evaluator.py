import hashlib
import json
from pathlib import Path
import subprocess
import sys

from feedback_control.evaluator import run_evaluation
from feedback_control.simulation import EpisodeResult, TrajectorySample
from feedback_control.trajectory import load_trajectory


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = EXPERIMENT_DIR / "configs" / "evaluation-v1.toml"
MODEL_PATH = EXPERIMENT_DIR / "models" / "one_link.xml"
SOURCE_COMMIT = "b" * 40
COMMAND = (
    "uv run python experiments/01-feedback-control/scripts/run_evaluation.py "
    "--config experiments/01-feedback-control/configs/evaluation-v1.toml "
    "--output results/evaluation-v1"
)
SCRIPT_PATH = EXPERIMENT_DIR / "scripts" / "run_evaluation.py"


def _write_evaluation_config(
    tmp_path: Path,
    *,
    duration_s: float = 0.02,
    steady_state_window_s: float = 0.01,
) -> Path:
    contents = CONFIG_PATH.read_text(encoding="utf-8")
    contents = contents.replace(
        'path = "../models/one_link.xml"',
        f'path = "{MODEL_PATH}"',
    )
    contents = contents.replace(
        "duration_s = 4.0",
        f"duration_s = {duration_s}",
    )
    contents = contents.replace(
        "steady_state_window_s = 0.50",
        f"steady_state_window_s = {steady_state_window_s}",
    )
    config_path = tmp_path / "evaluation-smoke.toml"
    config_path.write_text(contents, encoding="utf-8")
    return config_path


def test_short_evaluation_persists_traceable_complete_plan(
    tmp_path: Path,
) -> None:
    config_path = _write_evaluation_config(tmp_path)
    output_dir = tmp_path / "evaluation-output"

    result = run_evaluation(
        config_path=config_path,
        output_dir=output_dir,
        code_version=SOURCE_COMMIT,
        command=COMMAND,
    )

    assert len(result.episodes) == 8
    p_cases = [
        (episode.scenario_id, episode.seed)
        for episode in result.episodes
        if episode.controller == "p"
    ]
    pd_cases = [
        (episode.scenario_id, episode.seed)
        for episode in result.episodes
        if episode.controller == "pd"
    ]
    assert p_cases == pd_cases
    assert all(episode.included_in_aggregate for episode in result.episodes)
    assert all(episode.exclusion_reason is None for episode in result.episodes)
    assert all(episode.failure_reasons == ("not_settled",) for episode in result.episodes)

    for episode in result.episodes:
        trajectory_path = output_dir / episode.trajectory_file
        trajectory = load_trajectory(trajectory_path)
        assert trajectory.episode_id == episode.episode_id
        assert trajectory.code_version == SOURCE_COMMIT
        assert trajectory.seed == episode.seed
        assert (
            trajectory.configuration.initial_velocity_rad_s
            == trajectory.transitions[0].state.velocity_rad_s
        )
        assert episode.config_digest == trajectory.config_digest
        assert episode.trajectory_sha256 == hashlib.sha256(
            trajectory_path.read_bytes()
        ).hexdigest()

    assert [aggregate.controller for aggregate in result.aggregates] == ["p", "pd"]
    assert all(aggregate.episode_count == 4 for aggregate in result.aggregates)
    assert all(aggregate.included_episode_count == 4 for aggregate in result.aggregates)
    assert all(aggregate.failed_episode_count == 4 for aggregate in result.aggregates)

    summary_path = output_dir / "evaluation.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["provenance"]["code_version"] == SOURCE_COMMIT
    assert payload["provenance"]["command"] == COMMAND
    assert payload["provenance"]["config_sha256"] == hashlib.sha256(
        config_path.read_bytes()
    ).hexdigest()
    assert len(payload["episodes"]) == 8


def test_command_runs_complete_short_evaluation(tmp_path: Path) -> None:
    config_path = _write_evaluation_config(tmp_path)
    output_dir = tmp_path / "cli-output"

    completed = subprocess.run(
        (
            sys.executable,
            str(SCRIPT_PATH),
            "--config",
            str(config_path),
            "--output",
            str(output_dir),
            "--allow-dirty",
        ),
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(
        (output_dir / "evaluation.json").read_text(encoding="utf-8")
    )
    assert "evaluated 8 episodes" in completed.stdout
    assert payload["provenance"]["code_version"]
    assert isinstance(payload["provenance"]["working_tree_dirty"], bool)
    assert len(tuple((output_dir / "trajectories").glob("*.json"))) == 8


def test_joint_limit_failure_is_persisted_and_retained(tmp_path: Path) -> None:
    config_path = _write_evaluation_config(
        tmp_path,
        duration_s=1.0,
        steady_state_window_s=0.1,
    )
    output_dir = tmp_path / "joint-limit-output"

    result = run_evaluation(
        config_path=config_path,
        output_dir=output_dir,
        code_version=SOURCE_COMMIT,
        command=COMMAND,
    )

    failed = next(
        episode
        for episode in result.episodes
        if episode.controller == "p"
        and episode.scenario_id == "negative-to-positive-wide"
    )
    trajectory = load_trajectory(output_dir / failed.trajectory_file)
    assert "joint_limit_violation" in failed.failure_reasons
    assert failed.included_in_aggregate is True
    assert trajectory.transitions[-1].terminated is True
    assert trajectory.transitions[-1].truncated is False
    assert trajectory.transitions[-1].termination_reason == "joint_limit_violation"
    assert (
        trajectory.transitions[-1].next_state.position_rad
        > trajectory.configuration.position_upper_limit_rad
    )


def test_non_finite_failure_is_persisted_and_retained(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = _write_evaluation_config(tmp_path)
    output_dir = tmp_path / "non-finite-output"

    def non_finite_episode(config, controller, **_options):
        return EpisodeResult(
            controller=controller,
            samples=(
                TrajectorySample(
                    step_index=0,
                    time_s=0.0,
                    position_rad=config.episode.start_position_rad,
                    velocity_rad_s=config.episode.initial_velocity_rad_s,
                    target_position_rad=config.episode.target_position_rad,
                    target_velocity_rad_s=config.episode.target_velocity_rad_s,
                    raw_torque_nm=0.0,
                    applied_torque_nm=0.0,
                    saturated=False,
                ),
            ),
            final_time_s=config.episode.control_interval_s,
            final_position_rad=None,
            final_velocity_rad_s=None,
            terminated=True,
            truncated=False,
            termination_reason="non_finite_state",
            non_finite_state_fields=("position_rad",),
        )

    monkeypatch.setattr(
        "feedback_control.evaluator.run_episode",
        non_finite_episode,
    )
    result = run_evaluation(
        config_path=config_path,
        output_dir=output_dir,
        code_version=SOURCE_COMMIT,
        command=COMMAND,
    )

    assert len(result.episodes) == 8
    assert all(
        "non_finite_state" in episode.failure_reasons
        for episode in result.episodes
    )
    assert all(episode.included_in_aggregate for episode in result.episodes)
    trajectory = load_trajectory(
        output_dir / result.episodes[0].trajectory_file
    )
    final_transition = trajectory.transitions[-1]
    assert final_transition.next_state is None
    assert final_transition.non_finite_state_fields == ("position_rad",)
    assert final_transition.termination_reason == "non_finite_state"


def test_same_command_regenerates_identical_output(tmp_path: Path) -> None:
    config_path = _write_evaluation_config(tmp_path)
    output_dir = tmp_path / "repeatable-output"

    first = run_evaluation(
        config_path=config_path,
        output_dir=output_dir,
        code_version=SOURCE_COMMIT,
        command=COMMAND,
    )
    first_summary = (output_dir / "evaluation.json").read_bytes()
    first_trajectories = {
        path.name: path.read_bytes()
        for path in (output_dir / "trajectories").glob("*.json")
    }
    second = run_evaluation(
        config_path=config_path,
        output_dir=output_dir,
        code_version=SOURCE_COMMIT,
        command=COMMAND,
    )

    assert second == first
    assert (output_dir / "evaluation.json").read_bytes() == first_summary
    assert {
        path.name: path.read_bytes()
        for path in (output_dir / "trajectories").glob("*.json")
    } == first_trajectories
