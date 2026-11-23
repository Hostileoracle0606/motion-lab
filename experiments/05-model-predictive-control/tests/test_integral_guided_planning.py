from copy import copy

import numpy as np
import pytest

from forward_dynamics.types import DynamicsState
from model_predictive_control.closed_loop import IntegralFeedbackController, PDController
from model_predictive_control.contracts import NOMINAL_CONTEXT
from model_predictive_control.guided_planning import plan_integral_guided
from model_predictive_control.planner import CostWeights


class DisplacementPredictor:
    def predict_batch(self, states, actions, context):
        return np.column_stack((states[:, 0] + actions, states[:, 1]))


def guided(predictor, feedback, references, weights=CostWeights(1, 0, 0, 0)):
    return plan_integral_guided(
        predictor, NOMINAL_CONTEXT, DynamicsState(0, 0), weights, references,
        feedback=feedback, rng=np.random.default_rng(0), candidate_count=2,
        correction_limit_nm=0.25, correction_block_steps=5, minimum_relative_improvement=0.02,
    )


def test_preview_preserves_live_integral_memory_and_prefers_baseline_on_a_tie():
    feedback = IntegralFeedbackController(PDController(kp=1.0, kd=0.0), 1.0, 1.0)
    feedback(DynamicsState(0, 0), 0.5)
    independent = copy(feedback)
    result = guided(DisplacementPredictor(), feedback, np.array([[0.5, 0], [0.5, 0], [0.5, 0]]),
                    CostWeights(0, 0, 0, 0))
    expected = independent(DynamicsState(0, 0), 0.5).requested_torque_nm
    assert result.actions_nm[0] == pytest.approx(expected)
    assert result.predictor_calls == 4
    assert feedback(DynamicsState(0, 0), 0.5).requested_torque_nm == expected


def test_tracking_scores_the_reference_at_the_predicted_next_step():
    feedback = IntegralFeedbackController(PDController(kp=1.0, kd=0.0), 0.1, 1.0)
    result = guided(DisplacementPredictor(), feedback, np.array([[0.0, 0.0], [-0.1, 0.0]]))
    assert result.actions_nm == pytest.approx((-0.1151066431,))


def test_small_predicted_improvement_keeps_the_integral_action():
    feedback = IntegralFeedbackController(PDController(kp=1.0, kd=0.0), 0.1, 1.0)
    result = guided(DisplacementPredictor(), feedback, np.array([[0.0, 0.0], [-0.0577, 0.0]]))
    assert result.actions_nm == (0.0,)


def test_no_comparable_integral_prediction_requests_fallback():
    class RejectZero(DisplacementPredictor):
        def predict_batch(self, states, actions, context):
            predictions = super().predict_batch(states, actions, context)
            predictions[actions == 0, 0] = np.nan
            return predictions

    feedback = IntegralFeedbackController(PDController(kp=1.0, kd=0.0), 0.1, 1.0)
    result = guided(RejectZero(), feedback, np.array([[0.0, 0.0], [-0.1, 0.0]]))
    assert result.actions_nm is None
    assert result.feasible_candidates == 1


def test_real_feedback_tracks_velocity_reference():
    feedback = IntegralFeedbackController(PDController(kp=1.0, kd=0.5), 0.1, 1.0)
    decision = feedback(DynamicsState(0.2, 0.4), 0.2, target_velocity_rad_s=1.0)
    assert decision.requested_torque_nm == pytest.approx(0.3)
