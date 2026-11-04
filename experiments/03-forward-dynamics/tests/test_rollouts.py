from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from forward_dynamics.baselines import ConstantVelocityPredictor, NoChangePredictor
from forward_dynamics.dataset import PredictionCase, load_prediction_cases
from forward_dynamics.evaluation import evaluate_one_step
from forward_dynamics.mujoco_predictor import MujocoPredictor
from forward_dynamics.rollouts import evaluate_rollouts
from forward_dynamics.types import DynamicsContext, DynamicsState


ROOT = Path(__file__).resolve().parents[3]


def episode(episode_id="episode", split="train"):
    states = (
        DynamicsState(0.0, 0.0), DynamicsState(0.1, 1.0),
        DynamicsState(0.3, 2.0), DynamicsState(0.6, 3.0),
    )
    return tuple(
        PredictionCase(
            split, episode_id, f"scenario-{episode_id}", "controller", 7, index,
            states[index], float(index + 1), states[index + 1],
            DynamicsContext("plant", "version", 0.1, 3.0),
            index == 2, False, "failed" if index == 2 else None,
        )
        for index in range(3)
    )


class AccumulatingPredictor:
    predictor_id = "accumulating"

    def __init__(self):
        self.inputs = []

    def predict(self, state, applied_torque_nm, context):
        self.inputs.append((state, applied_torque_nm, context))
        return DynamicsState(state.position_rad + applied_torque_nm, state.velocity_rad_s + 1.0)


def test_predictions_feed_forward_with_recorded_actions_and_endpoint_alignment() -> None:
    predictor = AccumulatingPredictor()
    cases = episode()
    result = evaluate_rollouts(cases, predictor, (2,))
    assert predictor.inputs[:2] == [
        (DynamicsState(0.0, 0.0), 1.0, cases[0].context),
        (DynamicsState(1.0, 1.0), 2.0, cases[1].context),
    ]
    assert predictor.inputs[1][0] != cases[1].state
    first, final = result.predictions
    assert first.predicted_endpoint_state == DynamicsState(3.0, 2.0)
    assert first.expected_endpoint_state == DynamicsState(0.3, 2.0)
    assert first.residual.position_error_rad == pytest.approx(2.7)
    assert first.residual.velocity_error_rad_s == 0.0
    assert (first.split, first.episode_id, first.scenario_id, first.controller_id, first.seed) == (
        "train", "episode", "scenario-episode", "controller", 7,
    )
    assert (first.start_step_index, first.horizon_steps, first.horizon_s, first.predictor_id) == (
        0, 2, 0.2, "accumulating",
    )
    assert final.expected_endpoint_state == DynamicsState(0.6, 3.0)
    assert final.start_step_index == 1


def test_counts_include_unavailable_starts_and_terminal_endpoints() -> None:
    result = evaluate_rollouts(episode(), NoChangePredictor(), (1, 2, 5))
    assert [(count.split, count.episode_id, count.horizon_steps,
             count.eligible_count, count.unavailable_count) for count in result.windows] == [
        ("train", "episode", 1, 3, 0), ("train", "episode", 2, 2, 1),
        ("train", "episode", 5, 0, 3),
    ]
    assert Counter(row.horizon_steps for row in result.predictions) == {1: 3, 2: 2}
    assert {(row.horizon_steps, row.start_step_index) for row in result.predictions} == {
        (1, 0), (1, 1), (1, 2), (2, 0), (2, 1),
    }


def test_horizon_one_matches_one_step_evaluation() -> None:
    cases = episode()
    for predictor in (NoChangePredictor(), ConstantVelocityPredictor(), AccumulatingPredictor()):
        one_step = evaluate_one_step(cases, predictor)
        rollouts = evaluate_rollouts(cases, predictor, (1,)).predictions
        assert len(rollouts) == len(one_step)
        for row, one in zip(rollouts, one_step, strict=True):
            assert row.start_step_index == one.case.step_index
            assert row.predicted_endpoint_state == one.predicted_next_state
            assert row.expected_endpoint_state == one.case.expected_next_state
            assert row.residual == one.residual


def test_windows_stay_within_episode_and_split() -> None:
    result = evaluate_rollouts(
        episode("a", "train") + episode("b", "test"), NoChangePredictor(), (2, 5),
    )
    assert [(row.split, row.episode_id, row.horizon_steps, row.eligible_count,
             row.unavailable_count) for row in result.windows] == [
        ("train", "a", 2, 2, 1), ("train", "a", 5, 0, 3),
        ("test", "b", 2, 2, 1), ("test", "b", 5, 0, 3),
    ]
    assert len(result.predictions) == 4


def test_contiguous_slice_uses_supplied_rows_without_requiring_final_boundary() -> None:
    cases = tuple(replace(case, step_index=case.step_index + 10) for case in episode()[:2])
    result = evaluate_rollouts(cases, NoChangePredictor(), (1, 2, 5))
    assert [(row.eligible_count, row.unavailable_count) for row in result.windows] == [
        (2, 0), (1, 1), (0, 2),
    ]
    assert [(row.start_step_index, row.horizon_steps) for row in result.predictions] == [
        (10, 1), (11, 1), (10, 2),
    ]


@pytest.fixture(scope="module")
def recorded_cases():
    return load_prediction_cases(
        ROOT / "experiments/02-state-transition-dataset/data/dataset-v1",
        "experiment-02-transition-v1",
    )


def test_all_adapters_receive_identical_eligible_windows(recorded_cases) -> None:
    selected = recorded_cases[:3]
    predictors = (
        NoChangePredictor(), ConstantVelocityPredictor(),
        MujocoPredictor(ROOT / "experiments/01-feedback-control/models/one_link.xml"),
    )
    evaluations = [evaluate_rollouts(selected, predictor, (1, 2, 5)) for predictor in predictors]
    assert all(result.windows == evaluations[0].windows for result in evaluations)
    for result in evaluations:
        assert [(row.start_step_index, row.horizon_steps) for row in result.predictions] == [
            (0, 1), (1, 1), (2, 1), (0, 2), (1, 2),
        ]
    reference = evaluations[-1]
    for row in reference.predictions:
        assert row.residual.position_error_rad == pytest.approx(0.0, abs=1e-12, rel=0)
        assert row.residual.velocity_error_rad_s == pytest.approx(0.0, abs=1e-12, rel=0)


def test_failed_70_transition_episodes_remain_visible_at_horizon_100(recorded_cases) -> None:
    lengths = Counter(case.episode_id for case in recorded_cases)
    selected = tuple(case for case in recorded_cases if lengths[case.episode_id] == 70)
    assert len(selected) == 140
    assert sum(case.terminated for case in selected) == 2
    result = evaluate_rollouts(selected, NoChangePredictor(), (100,))
    assert result.predictions == ()
    assert len(result.windows) == 2
    assert all((row.eligible_count, row.unavailable_count) == (0, 70) for row in result.windows)


@pytest.mark.parametrize("horizons", [(), (1, 1), (0,), (-1,), (1.0,), (True,), (2, 1)])
def test_rejects_invalid_horizons(horizons) -> None:
    with pytest.raises(ValueError, match="horizon"):
        evaluate_rollouts(episode(), NoChangePredictor(), horizons)


@pytest.mark.parametrize("field,value", [
    ("context", DynamicsContext("other", "version", 0.1, 3.0)),
    ("context", DynamicsContext("plant", "version", 0.2, 3.0)),
    ("scenario_id", "other"), ("controller_id", "other"), ("seed", 8),
    ("step_index", 3), ("state", DynamicsState(0.2, 1.0)),
    ("state", DynamicsState(0.1, 1.1)), ("terminated", True),
])
def test_rejects_inconsistent_episode_context_continuity_and_boundaries(field, value) -> None:
    cases = list(episode())
    cases[1] = replace(cases[1], **{field: value})
    with pytest.raises(ValueError, match="episode"):
        evaluate_rollouts(tuple(cases), NoChangePredictor(), (1,))


def test_rejects_duplicate_cases_and_episode_split_leakage() -> None:
    cases = episode()
    for invalid in (cases + (cases[0],), cases + (replace(cases[0], split="test"),)):
        with pytest.raises(ValueError):
            evaluate_rollouts(invalid, NoChangePredictor(), (1,))


def test_continuity_uses_dataset_absolute_tolerance_in_both_physical_units() -> None:
    cases = list(episode())
    cases[1] = replace(cases[1], state=DynamicsState(0.1 + 5e-13, 1.0 + 5e-13))
    assert len(evaluate_rollouts(tuple(cases), NoChangePredictor(), (2,)).predictions) == 2
    cases[0] = replace(cases[0], expected_next_state=DynamicsState(1e9, 1e9))
    cases[1] = replace(cases[1], state=DynamicsState(1e9 + 1.0, 1e9))
    with pytest.raises(ValueError, match="continuity"):
        evaluate_rollouts(tuple(cases), NoChangePredictor(), (1,))


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), True])
def test_nonfinite_or_boolean_intermediate_prediction_fails_with_context(invalid) -> None:
    class InvalidIntermediate:
        predictor_id = "broken"

        def predict(self, state, applied_torque_nm, context):
            return DynamicsState(invalid, 0.0) if applied_torque_nm == 2.0 else DynamicsState(1.0, 0.0)

    with pytest.raises(ValueError) as error:
        evaluate_rollouts(episode(), InvalidIntermediate(), (3,))
    assert all(value in str(error.value) for value in ("broken", "episode", "step 1"))


def test_finite_unconstrained_predictions_are_retained() -> None:
    class BeyondJointLimit:
        predictor_id = "unconstrained"

        def predict(self, state, applied_torque_nm, context):
            return DynamicsState(state.position_rad + 100.0, -200.0)

    row, = evaluate_rollouts(episode(), BeyondJointLimit(), (3,)).predictions
    assert row.predicted_endpoint_state == DynamicsState(300.0, -200.0)
    assert row.residual.position_error_rad == 299.4
    assert row.residual.velocity_error_rad_s == -203.0


def test_numpy_prediction_scalars_are_normalized_before_recursive_arithmetic() -> None:
    class FixedWidthFirstPrediction:
        predictor_id = "fixed_width"

        def predict(self, state, applied_torque_nm, context):
            if applied_torque_nm == 1.0:
                return DynamicsState(np.int8(127), np.float32(2.0))
            return DynamicsState(state.position_rad + state.position_rad, state.velocity_rad_s)

    result = evaluate_rollouts(episode(), FixedWidthFirstPrediction(), (2,))
    assert result.predictions[0].predicted_endpoint_state == DynamicsState(254.0, 2.0)
    assert type(result.predictions[0].predicted_endpoint_state.position_rad) is float


def test_invalid_recorded_inputs_fail_even_when_no_window_is_eligible() -> None:
    cases = list(episode())
    cases[1] = replace(cases[1], applied_torque_nm=np.int8(-128))
    with pytest.raises(ValueError, match="torque"):
        evaluate_rollouts(tuple(cases), NoChangePredictor(), (100,))


def test_rejects_empty_cases_and_blank_predictor_identity() -> None:
    with pytest.raises(ValueError):
        evaluate_rollouts((), NoChangePredictor(), (1,))
    predictor = AccumulatingPredictor()
    predictor.predictor_id = " "
    with pytest.raises(ValueError, match="predictor_id"):
        evaluate_rollouts(episode(), predictor, (1,))
