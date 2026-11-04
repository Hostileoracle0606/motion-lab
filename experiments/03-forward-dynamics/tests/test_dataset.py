from dataclasses import asdict
from pathlib import Path

import pytest

from forward_dynamics.dataset import load_prediction_cases, validate_case_contexts
from state_transition_dataset.validation import load_and_validate_dataset


ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "experiments/02-state-transition-dataset/data/dataset-v1"
VERSION = "experiment-02-transition-v1"


def test_converts_every_validated_transition_without_repartitioning() -> None:
    cases = load_prediction_cases(DATA, VERSION)
    original = load_and_validate_dataset(DATA, VERSION)

    assert len(cases) == 2540
    assert {split: sum(case.split == split for case in cases) for split in ("train", "validation", "test")} == {
        "train": 940, "validation": 800, "test": 800,
    }
    assert sum(case.terminated for case in cases) == 2
    assert sum(case.truncated for case in cases) == 6
    for case in cases:
        row = next(row for row in original.rows_by_split[case.split] if
                   (row.episode_id, row.step_index) == (case.episode_id, case.step_index))
        assert case.state.position_rad == row.state_position_rad
        assert case.state.velocity_rad_s == row.state_velocity_rad_s
        assert case.applied_torque_nm == row.action_applied_torque_nm
        assert case.expected_next_state.position_rad == row.next_state_position_rad
        assert case.expected_next_state.velocity_rad_s == row.next_state_velocity_rad_s
        assert case.termination_reason == row.termination_reason
        assert "target_position_rad" not in asdict(case)


def test_dataset_identity_is_required() -> None:
    with pytest.raises(ValueError, match="version"):
        load_prediction_cases(DATA, "wrong-version")


def test_case_context_must_match_the_evaluation_protocol() -> None:
    cases = load_prediction_cases(DATA, VERSION)
    with pytest.raises(ValueError, match="context"):
        validate_case_contexts(cases, "different-environment", cases[0].context.environment_version)
