from dataclasses import replace
from functools import partial
import json
from pathlib import Path

import pytest

from feedback_control.config import ExperimentConfig, load_experiment_config
from feedback_control.simulation import run_episode
from feedback_control.trajectory import (
    load_trajectory,
    save_trajectory as _save_trajectory,
)


CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "demo.toml"
SOURCE_COMMIT = "a" * 40
save_trajectory = partial(_save_trajectory, code_version=SOURCE_COMMIT)


@pytest.fixture(scope="module")
def config() -> ExperimentConfig:
    return load_experiment_config(CONFIG_PATH)


def test_saved_episode_round_trip_preserves_transitions(
    tmp_path: Path,
    config: ExperimentConfig,
) -> None:
    short_config = replace(
        config,
        episode=replace(config.episode, duration_s=0.02),
    )
    result = run_episode(short_config, "pd")
    output_path = tmp_path / "episode.json"

    saved = save_trajectory(
        output_path,
        episode_id="pd-smoke-seed-7",
        seed=7,
        result=result,
        config=short_config,
    )
    loaded = load_trajectory(output_path)

    assert loaded == saved
    assert loaded.episode_id == "pd-smoke-seed-7"
    assert loaded.seed == 7
    assert loaded.code_version == SOURCE_COMMIT
    assert loaded.configuration.position_lower_limit_rad == pytest.approx(-1.570796)
    assert loaded.configuration.position_upper_limit_rad == pytest.approx(1.570796)
    assert loaded.controller.name == "pd"
    assert loaded.controller.kp == short_config.pd.kp
    assert loaded.controller.kd == short_config.pd.kd
    assert len(loaded.transitions) == 2
    assert loaded.transitions[0].state.position_rad == pytest.approx(
        result.samples[0].position_rad
    )
    assert loaded.transitions[0].next_state.position_rad == pytest.approx(
        result.samples[1].position_rad
    )
    assert loaded.transitions[-1].next_state.position_rad == pytest.approx(
        result.final_position_rad
    )
    assert loaded.transitions[-1].truncated is True
    assert loaded.transitions[-1].termination_reason == "time_limit"


def test_loading_rejects_a_missing_required_field(
    tmp_path: Path,
    config: ExperimentConfig,
) -> None:
    short_config = replace(
        config,
        episode=replace(config.episode, duration_s=0.01),
    )
    result = run_episode(short_config, "p")
    output_path = tmp_path / "missing-field.json"
    save_trajectory(
        output_path,
        episode_id="p-smoke-seed-0",
        seed=0,
        result=result,
        config=short_config,
    )
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    del payload["config_digest"]
    output_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="missing required field: config_digest"):
        load_trajectory(output_path)


def test_loading_rejects_missing_code_version(
    tmp_path: Path,
    config: ExperimentConfig,
) -> None:
    short_config = replace(
        config,
        episode=replace(config.episode, duration_s=0.01),
    )
    result = run_episode(short_config, "p")
    output_path = tmp_path / "missing-code-version.json"
    save_trajectory(
        output_path,
        episode_id="p-smoke-seed-0",
        seed=0,
        result=result,
        config=short_config,
    )
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    del payload["code_version"]
    output_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="missing required field: code_version"):
        load_trajectory(output_path)


@pytest.mark.parametrize("code_version", ("main", "A" * 40, "a" * 39))
def test_saving_rejects_invalid_code_version(
    tmp_path: Path,
    config: ExperimentConfig,
    code_version: str,
) -> None:
    short_config = replace(
        config,
        episode=replace(config.episode, duration_s=0.01),
    )
    result = run_episode(short_config, "p")

    with pytest.raises(
        ValueError,
        match="code_version must be a full lowercase Git commit object ID",
    ):
        _save_trajectory(
            tmp_path / "invalid-code-version.json",
            episode_id="p-smoke-seed-0",
            seed=0,
            code_version=code_version,
            result=result,
            config=short_config,
        )


def test_loading_rejects_a_missing_transition_field(
    tmp_path: Path,
    config: ExperimentConfig,
) -> None:
    short_config = replace(
        config,
        episode=replace(config.episode, duration_s=0.01),
    )
    result = run_episode(short_config, "p")
    output_path = tmp_path / "missing-transition-field.json"
    save_trajectory(
        output_path,
        episode_id="p-smoke-seed-0",
        seed=0,
        result=result,
        config=short_config,
    )
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    del payload["transitions"][0]["next_state"]
    output_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match=r"missing required field: transitions\[0\]\.next_state",
    ):
        load_trajectory(output_path)


def test_loading_rejects_an_unsupported_schema_version(
    tmp_path: Path,
    config: ExperimentConfig,
) -> None:
    short_config = replace(
        config,
        episode=replace(config.episode, duration_s=0.01),
    )
    result = run_episode(short_config, "p")
    output_path = tmp_path / "unsupported-schema.json"
    save_trajectory(
        output_path,
        episode_id="p-smoke-seed-0",
        seed=0,
        result=result,
        config=short_config,
    )
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    payload["schema_version"] = 1
    output_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported trajectory schema version: 1"):
        load_trajectory(output_path)


def test_loading_rejects_configuration_that_does_not_match_digest(
    tmp_path: Path,
    config: ExperimentConfig,
) -> None:
    short_config = replace(
        config,
        episode=replace(config.episode, duration_s=0.01),
    )
    result = run_episode(short_config, "p")
    output_path = tmp_path / "changed-config.json"
    save_trajectory(
        output_path,
        episode_id="p-smoke-seed-0",
        seed=0,
        result=result,
        config=short_config,
    )
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    payload["configuration"]["target_position_rad"] += 0.25
    output_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="config_digest does not match configuration"):
        load_trajectory(output_path)


def test_loading_requires_one_terminal_boundary_at_the_final_transition(
    tmp_path: Path,
    config: ExperimentConfig,
) -> None:
    short_config = replace(
        config,
        episode=replace(config.episode, duration_s=0.02),
    )
    result = run_episode(short_config, "p")
    output_path = tmp_path / "unterminated.json"
    save_trajectory(
        output_path,
        episode_id="p-smoke-seed-0",
        seed=0,
        result=result,
        config=short_config,
    )
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    payload["transitions"][-1]["truncated"] = False
    payload["transitions"][-1]["termination_reason"] = None
    output_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="trajectory must have exactly one final terminal boundary",
    ):
        load_trajectory(output_path)


def test_loading_rejects_state_with_the_wrong_shape(
    tmp_path: Path,
    config: ExperimentConfig,
) -> None:
    short_config = replace(
        config,
        episode=replace(config.episode, duration_s=0.01),
    )
    result = run_episode(short_config, "p")
    output_path = tmp_path / "bad-state-shape.json"
    save_trajectory(
        output_path,
        episode_id="p-smoke-seed-0",
        seed=0,
        result=result,
        config=short_config,
    )
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    state = payload["transitions"][0]["state"]
    payload["transitions"][0]["state"] = [
        state["position_rad"],
        state["velocity_rad_s"],
    ]
    output_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match=r"transitions\[0\]\.state must be an object",
    ):
        load_trajectory(output_path)


def test_loading_rejects_controller_metadata_that_conflicts_with_config(
    tmp_path: Path,
    config: ExperimentConfig,
) -> None:
    short_config = replace(
        config,
        episode=replace(config.episode, duration_s=0.01),
    )
    result = run_episode(short_config, "pd")
    output_path = tmp_path / "controller-mismatch.json"
    save_trajectory(
        output_path,
        episode_id="pd-smoke-seed-0",
        seed=0,
        result=result,
        config=short_config,
    )
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    payload["controller"]["kd"] += 1.0
    output_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="controller metadata does not match configuration",
    ):
        load_trajectory(output_path)


def test_saving_rejects_result_from_a_different_configuration(
    tmp_path: Path,
    config: ExperimentConfig,
) -> None:
    short_config = replace(
        config,
        episode=replace(config.episode, duration_s=0.01),
    )
    result = run_episode(short_config, "p")
    different_config = replace(
        short_config,
        episode=replace(
            short_config.episode,
            target_position_rad=short_config.episode.target_position_rad - 0.2,
        ),
    )

    with pytest.raises(
        ValueError,
        match="transition target does not match configuration",
    ):
        save_trajectory(
            tmp_path / "mismatched-config.json",
            episode_id="p-smoke-seed-0",
            seed=0,
            result=result,
            config=different_config,
        )


@pytest.mark.parametrize(
    ("episode_id", "seed", "message"),
    (
        ("", 0, "episode_id must not be empty"),
        ("p-smoke", -1, "seed must be a non-negative integer"),
    ),
)
def test_saving_rejects_invalid_episode_identity(
    tmp_path: Path,
    config: ExperimentConfig,
    episode_id: str,
    seed: int,
    message: str,
) -> None:
    short_config = replace(
        config,
        episode=replace(config.episode, duration_s=0.01),
    )
    result = run_episode(short_config, "p")

    with pytest.raises(ValueError, match=message):
        save_trajectory(
            tmp_path / "invalid-identity.json",
            episode_id=episode_id,
            seed=seed,
            result=result,
            config=short_config,
        )


def test_loading_rejects_transition_count_that_conflicts_with_duration(
    tmp_path: Path,
    config: ExperimentConfig,
) -> None:
    short_config = replace(
        config,
        episode=replace(config.episode, duration_s=0.02),
    )
    result = run_episode(short_config, "p")
    output_path = tmp_path / "missing-step.json"
    save_trajectory(
        output_path,
        episode_id="p-smoke-seed-0",
        seed=0,
        result=result,
        config=short_config,
    )
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    payload["transitions"] = payload["transitions"][:1]
    output_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="transition count does not match configured episode duration",
    ):
        load_trajectory(output_path)


def test_loading_rejects_non_finite_state(
    tmp_path: Path,
    config: ExperimentConfig,
) -> None:
    short_config = replace(
        config,
        episode=replace(config.episode, duration_s=0.01),
    )
    result = run_episode(short_config, "p")
    output_path = tmp_path / "non-finite.json"
    save_trajectory(
        output_path,
        episode_id="p-smoke-seed-0",
        seed=0,
        result=result,
        config=short_config,
    )
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    payload["transitions"][0]["state"]["position_rad"] = float("nan")
    output_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match=r"transitions\[0\]\.state\.position_rad must be finite",
    ):
        load_trajectory(output_path)


@pytest.mark.parametrize("state_name", ("state", "next_state"))
def test_loading_rejects_position_outside_model_limits(
    tmp_path: Path,
    config: ExperimentConfig,
    state_name: str,
) -> None:
    short_config = replace(
        config,
        episode=replace(config.episode, duration_s=0.01),
    )
    result = run_episode(short_config, "p")
    output_path = tmp_path / f"out-of-position-limit-{state_name}.json"
    save_trajectory(
        output_path,
        episode_id="p-smoke-seed-0",
        seed=0,
        result=result,
        config=short_config,
    )
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    payload["transitions"][0][state_name]["position_rad"] = 100.0
    output_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match=rf"transitions\[0\]\.{state_name}\.position_rad is outside "
        "the model position limits",
    ):
        load_trajectory(output_path)


def test_loading_rejects_out_of_order_steps(
    tmp_path: Path,
    config: ExperimentConfig,
) -> None:
    short_config = replace(
        config,
        episode=replace(config.episode, duration_s=0.02),
    )
    result = run_episode(short_config, "p")
    output_path = tmp_path / "out-of-order.json"
    save_trajectory(
        output_path,
        episode_id="p-smoke-seed-0",
        seed=0,
        result=result,
        config=short_config,
    )
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    payload["transitions"][1]["step_index"] = 0
    output_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="step indices must be contiguous from zero",
    ):
        load_trajectory(output_path)


def test_loading_rejects_time_outside_the_control_schedule(
    tmp_path: Path,
    config: ExperimentConfig,
) -> None:
    short_config = replace(
        config,
        episode=replace(config.episode, duration_s=0.02),
    )
    result = run_episode(short_config, "p")
    output_path = tmp_path / "bad-time.json"
    save_trajectory(
        output_path,
        episode_id="p-smoke-seed-0",
        seed=0,
        result=result,
        config=short_config,
    )
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    payload["transitions"][1]["time_s"] = 0.005
    output_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="times must follow the configured control schedule from zero",
    ):
        load_trajectory(output_path)


def test_loading_rejects_discontinuous_transitions(
    tmp_path: Path,
    config: ExperimentConfig,
) -> None:
    short_config = replace(
        config,
        episode=replace(config.episode, duration_s=0.02),
    )
    result = run_episode(short_config, "pd")
    output_path = tmp_path / "discontinuous.json"
    save_trajectory(
        output_path,
        episode_id="pd-smoke-seed-0",
        seed=0,
        result=result,
        config=short_config,
    )
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    payload["transitions"][1]["state"]["position_rad"] += 0.5
    output_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match=r"next_state at transition 0 does not match state at transition 1",
    ):
        load_trajectory(output_path)


def test_loading_rejects_applied_torque_outside_actuator_limit(
    tmp_path: Path,
    config: ExperimentConfig,
) -> None:
    short_config = replace(
        config,
        episode=replace(config.episode, duration_s=0.01),
    )
    result = run_episode(short_config, "p")
    output_path = tmp_path / "out-of-limit.json"
    save_trajectory(
        output_path,
        episode_id="p-smoke-seed-0",
        seed=0,
        result=result,
        config=short_config,
    )
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    payload["transitions"][0]["applied_torque_nm"] = 3.5
    output_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match=r"transitions\[0\]\.applied_torque_nm exceeds torque limit",
    ):
        load_trajectory(output_path)


def test_loading_rejects_incorrect_saturation_metadata(
    tmp_path: Path,
    config: ExperimentConfig,
) -> None:
    short_config = replace(
        config,
        episode=replace(config.episode, duration_s=0.01),
    )
    result = run_episode(short_config, "p")
    output_path = tmp_path / "bad-saturation.json"
    save_trajectory(
        output_path,
        episode_id="p-smoke-seed-0",
        seed=0,
        result=result,
        config=short_config,
    )
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    payload["transitions"][0]["saturated"] = False
    output_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match=r"transitions\[0\]\.saturated does not match torque clipping",
    ):
        load_trajectory(output_path)
