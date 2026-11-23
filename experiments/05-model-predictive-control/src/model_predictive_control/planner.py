"""Random-shooting planning over bounded, open-loop torque sequences."""

from collections.abc import Callable
from dataclasses import dataclass
import math
from numbers import Integral

import numpy as np

from forward_dynamics.types import (
    DynamicsContext, DynamicsState, Predictor, validate_finite_scalar,
    validate_prediction_inputs, validate_state,
)
from model_predictive_control.contracts import Controller, POSITION_LIMIT_RAD, PlanResult


@dataclass(frozen=True)
class CostWeights:
    position: float
    velocity: float
    effort: float
    terminal_position: float
    terminal_velocity: float = 0.0

    def __post_init__(self) -> None:
        for name in ("position", "velocity", "effort", "terminal_position", "terminal_velocity"):
            value = getattr(self, name)
            validate_finite_scalar(name, value)
            if value < 0:
                raise ValueError(f"{name} must be nonnegative")


def score_candidates(
    predictor: Predictor, context: DynamicsContext, state: DynamicsState,
    target_position_rad: float, weights: CostWeights, candidates_nm: np.ndarray,
) -> PlanResult:
    """Evaluate every candidate, rejecting invalid predicted trajectories."""
    result, _ = _score_batch(predictor, context, state, target_position_rad, weights, candidates_nm)
    return result


def _validated_candidates(
    context: DynamicsContext, state: DynamicsState, target_position_rad: float,
    weights: CostWeights, candidates_nm: np.ndarray,
) -> np.ndarray:
    validate_prediction_inputs(state, 0.0, context)
    validate_finite_scalar("target_position_rad", target_position_rad)
    if abs(state.position_rad) > POSITION_LIMIT_RAD or abs(target_position_rad) > POSITION_LIMIT_RAD:
        raise ValueError("initial position and target must lie within the joint limits")
    if not isinstance(weights, CostWeights):
        raise ValueError("weights must be CostWeights")
    try:
        candidates = np.asarray(candidates_nm, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("candidates must be a finite matrix") from exc
    if (candidates.ndim != 2 or not all(candidates.shape)
            or not np.isfinite(candidates).all()
            or np.any(np.abs(candidates) > context.torque_limit_nm)):
        raise ValueError("candidates must be a nonempty bounded finite matrix")
    return candidates


def _feedback_torque(
    feedback: Controller, state: DynamicsState, target_position_rad: float, limit: float,
) -> float:
    request = feedback(state, target_position_rad).requested_torque_nm
    validate_finite_scalar("feedback torque", request)
    return min(limit, max(-limit, float(request)))


def _score_batch(
    predictor: Predictor, context: DynamicsContext, state: DynamicsState,
    target_position_rad: float, weights: CostWeights, candidates_nm: np.ndarray,
    *, feedback: Controller | None = None,
) -> tuple[PlanResult, list[tuple[int, float]]]:
    """Return the public result and feasible candidate indices and costs."""
    candidates = _validated_candidates(context, state, target_position_rad, weights, candidates_nm)

    best_actions = None
    best_cost = None
    feasible = calls = 0
    ranked_candidates = []
    for index, candidate in enumerate(candidates):
        current = state
        cost = 0.0
        uses_feedback = feedback is not None and index == len(candidates) - 1
        feedback_actions = []
        for torque in candidate:
            if uses_feedback:
                torque = _feedback_torque(feedback, current, target_position_rad, float(context.torque_limit_nm))
                feedback_actions.append(torque)
            calls += 1
            try:
                current = predictor.predict(current, float(torque), context)
            except ValueError:
                break
            try:
                validate_state(current)
                if abs(current.position_rad) > POSITION_LIMIT_RAD:
                    raise ValueError("predicted position outside joint limits")
                error = float(current.position_rad) - float(target_position_rad)
                cost += (float(weights.position) * error ** 2
                         + float(weights.velocity) * float(current.velocity_rad_s) ** 2
                         + float(weights.effort) * float(torque) ** 2)
                if not math.isfinite(cost):
                    raise ValueError("nonfinite cost")
            except (ValueError, OverflowError):
                break
        else:
            try:
                cost += float(weights.terminal_position) * error ** 2
                if weights.terminal_velocity:
                    cost += float(weights.terminal_velocity) * float(current.velocity_rad_s) ** 2
                if not math.isfinite(cost):
                    continue
            except OverflowError:
                continue
            feasible += 1
            ranked_candidates.append((index, cost))
            if best_cost is None or cost < best_cost:
                best_cost = cost
                best_actions = (tuple(feedback_actions) if uses_feedback
                                else tuple(float(torque) for torque in candidate))
    return PlanResult(best_actions, best_cost, len(candidates), feasible, calls), ranked_candidates


def _retry_scalar(
    predictor: Predictor, context: DynamicsContext, states: np.ndarray, actions: np.ndarray,
) -> np.ndarray:
    predictions = np.full_like(states, np.nan)
    for index, (row, torque) in enumerate(zip(states, actions, strict=True)):
        try:
            answer = predictor.predict(DynamicsState(*map(float, row)), float(torque), context)
            validate_state(answer)
        except ValueError:
            continue
        predictions[index] = (answer.position_rad, answer.velocity_rad_s)
    return predictions


def _score_batched(
    predictor: Predictor, predict_batch: Callable[[np.ndarray, np.ndarray, DynamicsContext], np.ndarray],
    context: DynamicsContext, state: DynamicsState, target_position_rad: float,
    weights: CostWeights, candidates_nm: np.ndarray, feedback: Controller | None,
) -> PlanResult:
    candidates = _validated_candidates(context, state, target_position_rad, weights, candidates_nm)
    count, horizon = candidates.shape
    active = np.arange(count)
    states = np.tile((float(state.position_rad), float(state.velocity_rad_s)), (count, 1))
    costs = np.zeros(count)
    calls = 0
    for tick in range(horizon):
        if feedback is not None and active[-1] == count - 1:
            feedback_state = DynamicsState(*map(float, states[-1]))
            candidates[-1, tick] = _feedback_torque(
                feedback, feedback_state, target_position_rad, float(context.torque_limit_nm),
            )
        actions = candidates[active, tick]
        calls += active.size
        try:
            predictions = predict_batch(states, actions, context)
        except ValueError:
            calls += active.size
            predictions = _retry_scalar(predictor, context, states, actions)
        if (not isinstance(predictions, np.ndarray) or predictions.shape != states.shape
                or predictions.dtype.kind not in "iuf"):
            raise ValueError("batch predictions must be a real numeric [N,2] array")
        predictions = predictions.astype(float, copy=False)
        position, velocity = predictions.T
        with np.errstate(over="ignore", invalid="ignore"):
            position_error = position - float(target_position_rad)
            costs += (float(weights.position) * position_error ** 2
                      + float(weights.velocity) * velocity ** 2
                      + float(weights.effort) * actions ** 2)
            if tick == horizon - 1:
                costs += float(weights.terminal_position) * position_error ** 2
                if weights.terminal_velocity:
                    costs += float(weights.terminal_velocity) * velocity ** 2
        feasible = (np.isfinite(predictions).all(axis=1) & (np.abs(position) <= POSITION_LIMIT_RAD)
                    & np.isfinite(costs))
        active = active[feasible]
        if not active.size:
            return PlanResult(None, None, count, 0, int(calls))
        states = predictions[feasible]
        costs = costs[feasible]
    best = int(np.argmin(costs))
    return PlanResult(tuple(map(float, candidates[active[best]])), float(costs[best]),
                      count, active.size, int(calls))


def plan(
    predictor: Predictor, context: DynamicsContext, state: DynamicsState,
    target_position_rad: float, weights: CostWeights, *, horizon: int,
    candidate_count: int, rng: np.random.Generator, feedback: Controller | None = None,
    batched: bool = False,
) -> PlanResult:
    """Sample candidate_count plans, optionally using feedback and batched prediction.

    Predictor work counts submitted state-action rows, including scalar
    retries of failed batches.
    """
    for name, value in (("horizon", horizon), ("candidate_count", candidate_count)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    validate_prediction_inputs(state, 0.0, context)
    candidates = rng.uniform(-context.torque_limit_nm, context.torque_limit_nm,
                             size=(candidate_count, horizon))
    if batched:
        predict_batch = getattr(predictor, "predict_batch", None)
        if not callable(predict_batch):
            raise ValueError("batched planning requires a batch-capable predictor")
        return _score_batched(
            predictor, predict_batch, context, state, target_position_rad, weights, candidates, feedback,
        )
    result, _ = _score_batch(
        predictor, context, state, target_position_rad, weights, candidates, feedback=feedback,
    )
    return result
