"""Batched torque corrections around a preview of integral feedback."""

from numbers import Integral

import numpy as np

from forward_dynamics.types import DynamicsContext, DynamicsState, Predictor, validate_finite_scalar
from model_predictive_control.closed_loop import IntegralFeedbackController
from model_predictive_control.contracts import CONTROL_INTERVAL_S, POSITION_LIMIT_RAD, TORQUE_LIMIT_NM, PlanResult
from model_predictive_control.planner import CostWeights, _score_batched


def plan_integral_guided(
    predictor: Predictor, context: DynamicsContext, state: DynamicsState, weights: CostWeights,
    references: np.ndarray, *, feedback: IntegralFeedbackController, rng: np.random.Generator,
    candidate_count: int, correction_limit_nm: float, correction_block_steps: int,
    minimum_relative_improvement: float,
) -> PlanResult:
    """Compare bounded adjustments with a zero-correction plan from the live feedback snapshot."""
    reference = np.asarray(references, dtype=float)
    if (reference.ndim != 2 or reference.shape[1] != 2 or len(reference) < 2
            or not np.isfinite(reference).all() or (np.abs(reference[:, 0]) > POSITION_LIMIT_RAD).any()):
        raise ValueError("reference must contain current and future position/velocity pairs within the joint limits")
    for name, value in (("candidate_count", candidate_count), ("correction_block_steps", correction_block_steps)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    validate_finite_scalar("correction limit", correction_limit_nm)
    validate_finite_scalar("minimum relative improvement", minimum_relative_improvement)
    if not 0 <= correction_limit_nm <= context.torque_limit_nm or not 0 <= minimum_relative_improvement <= 1:
        raise ValueError("correction limit or improvement threshold is out of range")
    if context.control_interval_s != CONTROL_INTERVAL_S or context.torque_limit_nm != TORQUE_LIMIT_NM:
        raise ValueError("prediction interval and torque limit must match integral feedback")
    predict_batch = getattr(predictor, "predict_batch", None)
    if not callable(predict_batch):
        raise ValueError("integral guidance requires a batch-capable predictor")
    horizon = len(reference) - 1
    blocks = (horizon + correction_block_steps - 1) // correction_block_steps
    offsets = rng.uniform(-correction_limit_nm, correction_limit_nm, size=(candidate_count, blocks))
    offsets[0] = 0
    candidates = np.repeat(offsets, correction_block_steps, axis=1)[:, :horizon]
    integral_torques = np.full(candidate_count, feedback.integral_torque_nm)
    kp, kd = feedback.pd.kp, feedback.pd.kd
    increment = feedback.integral_gain * CONTROL_INTERVAL_S
    bound = feedback.integral_limit_nm
    limit = float(context.torque_limit_nm)

    def adjust(states, corrections, active, tick):
        error = reference[tick, 0] - states[:, 0]
        raw_pd = kp * error + kd * (reference[tick, 1] - states[:, 1])
        previous = integral_torques[active]
        proposal = np.clip(previous + increment * error, -bound, bound)
        combined = raw_pd + proposal
        windup = ((combined > limit) & (proposal > previous)) | ((combined < -limit) & (proposal < previous))
        integral_torques[active] = np.where(windup, previous, proposal)
        return np.clip(raw_pd + integral_torques[active] + corrections, -limit, limit)

    return _score_batched(
        predictor, predict_batch, context, state, float(reference[0, 0]), weights, candidates, None,
        references=reference, adjust_actions=adjust, minimum_relative_improvement=minimum_relative_improvement,
    )
