from dataclasses import replace
from pathlib import Path

import pytest

from forward_dynamics.baselines import NoChangePredictor
from forward_dynamics.dataset import load_prediction_cases
from forward_dynamics.evaluation import evaluate_one_step, summarize_predictions
from forward_dynamics.types import DynamicsState


ROOT = Path(__file__).resolve().parents[3]


def cases():
    return load_prediction_cases(
        ROOT / "experiments/02-state-transition-dataset/data/dataset-v1",
        "experiment-02-transition-v1",
    )


def test_one_step_resets_to_observation_and_does_not_expose_expected_state() -> None:
    selected = cases()[:2]

    class WrongPredictor:
        predictor_id = "wrong"

        def __init__(self):
            self.inputs = []

        def predict(self, state, applied_torque_nm, context):
            self.inputs.append((state, applied_torque_nm, context))
            return DynamicsState(99.0, -99.0)

    predictor = WrongPredictor()
    rows = evaluate_one_step(selected, predictor)
    assert predictor.inputs == [(case.state, case.applied_torque_nm, case.context) for case in selected]
    assert predictor.inputs[1][0] != rows[0].predicted_next_state
    assert tuple(row.case for row in rows) == selected


def test_aggregates_preserve_groups_and_component_units() -> None:
    base = cases()[0]
    selected = (
        replace(base, state=DynamicsState(0.0, 0.0), expected_next_state=DynamicsState(-1.0, 0.0)),
        replace(base, step_index=1, state=DynamicsState(0.0, 0.0), expected_next_state=DynamicsState(3.0, 4.0)),
    )
    summaries = summarize_predictions(evaluate_one_step(selected, NoChangePredictor()))
    assert {group["group"] for group in summaries} == {"episode", "scenario", "controller", "split"}
    for group in summaries:
        assert group["summary"]["sample_count"] == 2
        assert group["summary"]["position_mae_rad"] == 2.0
        assert group["summary"]["position_rmse_rad"] == pytest.approx(5.0**0.5)
        assert group["summary"]["velocity_mae_rad_s"] == 2.0
        assert group["summary"]["velocity_rmse_rad_s"] == pytest.approx(8.0**0.5)


def test_invalid_prediction_error_identifies_predictor_episode_and_step() -> None:
    class NonFinitePredictor:
        predictor_id = "broken"

        def predict(self, state, applied_torque_nm, context):
            return DynamicsState(float("nan"), 0.0)

    case = cases()[0]
    with pytest.raises(ValueError) as error:
        evaluate_one_step((case,), NonFinitePredictor())
    assert all(value in str(error.value) for value in ("broken", case.episode_id, "step 0"))
