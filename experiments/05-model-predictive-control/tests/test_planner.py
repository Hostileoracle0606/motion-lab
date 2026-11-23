import numpy as np
import pytest

from forward_dynamics.types import DynamicsState
from model_predictive_control.contracts import ControlDecision, NOMINAL_CONTEXT, POSITION_LIMIT_RAD
from model_predictive_control.planner import CostWeights, plan, score_candidates


class Integrator:
    def predict(self, state, applied_torque_nm, context):
        return DynamicsState(state.position_rad + applied_torque_nm, applied_torque_nm)


class RejectPositive:
    def predict(self, state, applied_torque_nm, context):
        if applied_torque_nm > 0:
            raise ValueError('invalid preview')
        return DynamicsState(state.position_rad, 0.0)


def test_hand_scored_recurrence_and_tie_preserves_first():
    result = score_candidates(Integrator(), NOMINAL_CONTEXT, DynamicsState(0, 0), 0,
                              CostWeights(1, 2, 1, 3), np.array([[0.5, -0.5], [-0.5, 0.5], [1, 0]]))
    assert result.actions_nm == (0.5, -0.5)
    assert result.cost == pytest.approx(1.75)
    assert (result.candidates, result.feasible_candidates, result.predictor_calls) == (3, 3, 6)


def test_invalid_preview_rejects_only_candidate_and_counts_attempts():
    result = score_candidates(RejectPositive(), NOMINAL_CONTEXT, DynamicsState(0, 0), 0,
                              CostWeights(1, 0, 0, 0), np.array([[0, 1], [0, 0]]))
    assert result.actions_nm == (0, 0)
    assert (result.candidates, result.feasible_candidates, result.predictor_calls) == (2, 1, 4)


def test_terminal_velocity_weight_prefers_the_plan_that_comes_to_rest():
    # Both plans move by one radian and have the same sum of squared velocities.
    # Only the second plan predicts zero terminal velocity.
    candidates = np.array([[0, 1], [1, 0]])
    args = (Integrator(), NOMINAL_CONTEXT, DynamicsState(0, 0), 1)
    baseline = score_candidates(*args, CostWeights(0, 1, 0, 1), candidates)
    penalized = score_candidates(*args, CostWeights(0, 1, 0, 1, terminal_velocity=4), candidates)
    assert baseline.actions_nm == (0, 1)
    assert penalized.actions_nm == (1, 0)
    assert penalized.cost == 1.0
    assert penalized.predictor_calls == baseline.predictor_calls == 4


def test_out_of_range_preview_and_infeasible_accounting():
    result = score_candidates(Integrator(), NOMINAL_CONTEXT, DynamicsState(1.5, 0), 0,
                              CostWeights(1, 0, 0, 0), np.array([[1, 0], [2, 0]]))
    assert result.actions_nm is None and result.cost is None
    assert (result.candidates, result.feasible_candidates, result.predictor_calls) == (2, 0, 2)


@pytest.mark.parametrize('values', [
    (-1, 0, 0, 0), (float('nan'), 0, 0, 0), (0, float('inf'), 0, 0), (True, 0, 0, 0),
    (0, 0, 0, 0, -1), (0, 0, 0, 0, float('nan')), (0, 0, 0, 0, True),
])
def test_invalid_weights(values):
    with pytest.raises(ValueError):
        CostWeights(*values)


@pytest.mark.parametrize('matrix', [[], [[4]], [[float('nan')]], [[1], [1, 2]], [1, 2], [[]]])
def test_invalid_candidates(matrix):
    with pytest.raises(ValueError):
        score_candidates(Integrator(), NOMINAL_CONTEXT, DynamicsState(0, 0), 0, CostWeights(1, 1, 1, 1), matrix)


@pytest.mark.parametrize('state,target', [(DynamicsState(POSITION_LIMIT_RAD + 0.1, 0), 0), (DynamicsState(0, 0), POSITION_LIMIT_RAD + 0.1), (DynamicsState(0, float('nan')), 0)])
def test_invalid_initial_state_or_target(state, target):
    with pytest.raises(ValueError):
        score_candidates(Integrator(), NOMINAL_CONTEXT, state, target, CostWeights(1, 1, 1, 1), [[0]])


@pytest.mark.parametrize('horizon,count', [(0, 1), (1, 0), (True, 1), (1, 1.5)])
def test_plan_rejects_invalid_dimensions(horizon, count):
    with pytest.raises(ValueError):
        plan(Integrator(), NOMINAL_CONTEXT, DynamicsState(0, 0), 0, CostWeights(1, 1, 1, 1), horizon=horizon, candidate_count=count, rng=np.random.default_rng(1))


def test_plan_draws_exact_seeded_candidates():
    rng = np.random.default_rng(4)
    expected_rng = np.random.default_rng(4)
    expected = expected_rng.uniform(-3, 3, size=(3, 2))
    expected_result = score_candidates(Integrator(), NOMINAL_CONTEXT, DynamicsState(0, 0), 0, CostWeights(1, 1, 1, 1), expected)
    result = plan(Integrator(), NOMINAL_CONTEXT, DynamicsState(0, 0), 0, CostWeights(1, 1, 1, 1), horizon=2, candidate_count=3, rng=rng)
    assert result == expected_result
    assert rng.random() == expected_rng.random()


def test_feedback_candidate_uses_the_predicted_state_within_candidate_budget():
    class QuarterStep:
        def predict(self, state, applied_torque_nm, context):
            return DynamicsState(state.position_rad + applied_torque_nm / 4, 0.0)

    def feedback(state, target):
        return ControlDecision(15.0 if state.position_rad < target else 0.0)

    result = plan(
        QuarterStep(), NOMINAL_CONTEXT, DynamicsState(0, 0), 0.75,
        CostWeights(1, 0, 0, 1), horizon=2, candidate_count=2,
        rng=np.random.default_rng(0), feedback=feedback,
    )
    assert result.actions_nm == (3.0, 0.0)
    assert result.cost == 0.0
    assert (result.candidates, result.feasible_candidates, result.predictor_calls) == (2, 2, 4)


def test_unexpected_predictor_exception_propagates():
    class Broken:
        def predict(self, state, applied_torque_nm, context):
            raise OverflowError('predictor bug')
    with pytest.raises(OverflowError, match='predictor bug'):
        score_candidates(Broken(), NOMINAL_CONTEXT, DynamicsState(0, 0), 0, CostWeights(1, 0, 0, 0), [[0]])


def test_nonfinite_prediction_rejects_only_that_candidate():
    class SometimesInvalid:
        def predict(self, state, applied_torque_nm, context):
            return DynamicsState(float('nan') if applied_torque_nm else 0, 0)
    result = score_candidates(SometimesInvalid(), NOMINAL_CONTEXT, DynamicsState(0, 0), 0, CostWeights(1, 0, 0, 0), [[1], [0]])
    assert result.actions_nm == (0,)
    assert (result.feasible_candidates, result.predictor_calls) == (1, 2)
