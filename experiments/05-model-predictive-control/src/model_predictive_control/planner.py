"""Random-shooting planning over bounded, open-loop torque sequences."""

from dataclasses import dataclass
import math
from numbers import Integral

import numpy as np

from forward_dynamics.types import (
    DynamicsContext, DynamicsState, Predictor, validate_finite_scalar,
    validate_prediction_inputs, validate_state,
)
from model_predictive_control.contracts import POSITION_LIMIT_RAD, PlanResult


@dataclass(frozen=True)
class CostWeights:
    position: float
    velocity: float
    effort: float
    terminal_position: float

    def __post_init__(self) -> None:
        for name in ("position", "velocity", "effort", "terminal_position"):
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


def _score_batch(
    predictor: Predictor, context: DynamicsContext, state: DynamicsState,
    target_position_rad: float, weights: CostWeights, candidates_nm: np.ndarray,
) -> tuple[PlanResult, list[tuple[int, float]]]:
    """Return the public result and feasible candidate indices and costs."""
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

    best_actions = None
    best_cost = None
    feasible = calls = 0
    ranked_candidates = []
    for index, candidate in enumerate(candidates):
        current = state
        cost = 0.0
        for torque in candidate:
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
                if not math.isfinite(cost):
                    continue
            except OverflowError:
                continue
            feasible += 1
            ranked_candidates.append((index, cost))
            if best_cost is None or cost < best_cost:
                best_cost = cost
                best_actions = tuple(float(torque) for torque in candidate)
    return PlanResult(best_actions, best_cost, len(candidates), feasible, calls), ranked_candidates


def plan(
    predictor: Predictor, context: DynamicsContext, state: DynamicsState,
    target_position_rad: float, weights: CostWeights, *, horizon: int,
    candidate_count: int, rng: np.random.Generator,
) -> PlanResult:
    """Sample exactly candidate_count torque sequences using the supplied RNG."""
    for name, value in (("horizon", horizon), ("candidate_count", candidate_count)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    validate_prediction_inputs(state, 0.0, context)
    candidates = rng.uniform(-context.torque_limit_nm, context.torque_limit_nm,
                             size=(candidate_count, horizon))
    return score_candidates(predictor, context, state, target_position_rad, weights, candidates)
