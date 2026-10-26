from dataclasses import replace
import math
from pathlib import Path

import pytest

import state_transition_dataset.builder as builder_module
from feedback_control.report_evidence import (
    EvaluationEvidence,
    load_evaluation_evidence,
)
from feedback_control.trajectory import StateRecord
from state_transition_dataset.builder import build_rows
from state_transition_dataset.config import load_dataset_config


ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = (
    ROOT / "experiments/02-state-transition-dataset/configs/dataset-v1.toml"
)


@pytest.fixture(scope="module")
def evidence() -> EvaluationEvidence:
    config = load_dataset_config(CONFIG_PATH)
    return load_evaluation_evidence(
        config.source_evaluation_dir,
        config.source_evaluation_config,
    )


def test_builds_complete_scenario_partitions_with_paired_controllers() -> None:
    config = load_dataset_config(CONFIG_PATH)

    build = build_rows(config)

    assert tuple(build.rows_by_split) == ("train", "validation", "test")
    assert len(build.episodes) == 8
    assert build.source_evaluation_sha256 == (
        "f17e2cfb9a56d417918b9a2db497bcdf8b1710aa195c9d83b9366bc5b472b2d9"
    )
    assert build.source_code_version == (
        "7a6d152f5d6f9358681e1b67ad08cbb1f0c3484c"
    )
    assert build.model_path == (
        ROOT / "experiments/01-feedback-control/models/one_link.xml"
    ).resolve()

    expected = config.scenario_to_split
    for episode in build.episodes:
        assert {
            row.split
            for row in build.rows_by_split[expected[episode.scenario_id]]
            if row.scenario_id == episode.scenario_id
        } == {expected[episode.scenario_id]}
    for scenario_id in expected:
        controllers = {
            row.controller_id
            for rows in build.rows_by_split.values()
            for row in rows
            if row.scenario_id == scenario_id
        }
        assert controllers == {"p", "pd"}


def test_preserves_transition_timing_action_context_and_early_boundary(
    evidence: EvaluationEvidence,
) -> None:
    config = load_dataset_config(CONFIG_PATH)
    build = build_rows(config)
    source_episode = evidence.episodes[0]
    source = source_episode.trajectory.transitions[0]
    rows = tuple(
        row
        for row in build.rows_by_split["train"]
        if row.episode_id == source_episode.episode_id
    )
    row = rows[0]

    assert row.step_index == source.step_index
    assert row.time_s == source.time_s
    assert row.control_interval_s == (
        source_episode.trajectory.configuration.control_interval_s
    )
    assert row.state_position_rad == source.state.position_rad
    assert row.state_velocity_rad_s == source.state.velocity_rad_s
    assert row.action_applied_torque_nm == source.applied_torque_nm
    assert row.next_state_position_rad == source.next_state.position_rad
    assert row.next_state_velocity_rad_s == source.next_state.velocity_rad_s
    assert row.target_position_rad == source.target.position_rad
    assert row.target_velocity_rad_s == source.target.velocity_rad_s
    assert row.torque_limit_nm == (
        source_episode.trajectory.configuration.torque_limit_nm
    )
    assert row.environment_id == source_episode.trajectory.environment_id
    assert row.environment_version == source_episode.trajectory.environment_version
    assert row.source_code_version == source_episode.trajectory.code_version
    assert row.config_digest == source_episode.trajectory.config_digest
    assert row.source_trajectory_sha256 == source_episode.trajectory_sha256

    source_final = source_episode.trajectory.transitions[-1]
    assert len(rows) < 400
    assert rows[-1].terminated is True
    assert rows[-1].truncated is False
    assert rows[-1].termination_reason == source_final.termination_reason


@pytest.mark.parametrize("mode", ["unassigned", "multiply-assigned"])
def test_rejects_invalid_scenario_assignment(mode: str) -> None:
    config = load_dataset_config(CONFIG_PATH)
    assignments = list(config.scenario_splits)
    if mode == "unassigned":
        assignments.pop()
        message = "exactly match source evaluation scenarios"
    else:
        assignments.append(assignments[0])
        message = "scenario split assignments must be unique"

    with pytest.raises(ValueError, match=message):
        build_rows(replace(config, scenario_splits=tuple(assignments)))


@pytest.mark.parametrize("invalid_value", [None, float("nan"), float("inf")])
def test_rejects_non_finite_source_state(
    monkeypatch: pytest.MonkeyPatch,
    evidence: EvaluationEvidence,
    invalid_value: float | None,
) -> None:
    source_episode = evidence.episodes[0]
    source_transition = source_episode.trajectory.transitions[0]
    invalid_transition = replace(
        source_transition,
        state=StateRecord(
            position_rad=invalid_value,
            velocity_rad_s=source_transition.state.velocity_rad_s,
        ),
    )
    invalid_trajectory = replace(
        source_episode.trajectory,
        transitions=(invalid_transition,)
        + source_episode.trajectory.transitions[1:],
    )
    invalid_episode = replace(source_episode, trajectory=invalid_trajectory)
    invalid_evidence = replace(
        evidence,
        episodes=(invalid_episode,) + evidence.episodes[1:],
    )
    monkeypatch.setattr(
        builder_module,
        "load_evaluation_evidence",
        lambda *_: invalid_evidence,
    )

    with pytest.raises(ValueError, match="finite source transition values"):
        build_rows(load_dataset_config(CONFIG_PATH))

    if invalid_value is not None:
        assert not math.isfinite(invalid_value)
