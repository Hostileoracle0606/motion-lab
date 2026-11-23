"""Bounded, seeded cross-entropy planning with a fixed rollout budget."""

from numbers import Integral

import numpy as np

from forward_dynamics.types import (
    DynamicsContext, DynamicsState, Predictor, validate_finite_scalar,
    validate_prediction_inputs,
)
from model_predictive_control.contracts import PlanResult
from model_predictive_control.planner import CostWeights, _score_batch


def plan_cem(
    predictor: Predictor, context: DynamicsContext, state: DynamicsState,
    target_position_rad: float, weights: CostWeights, *, horizon: int,
    candidate_count: int, iterations: int, elite_count: int,
    min_std_nm: float, rng: np.random.Generator,
) -> PlanResult:
    """Score a fixed number of candidates over successive CEM rounds."""
    for name, value in (("horizon", horizon), ("candidate_count", candidate_count),
                        ("iterations", iterations), ("elite_count", elite_count)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    batch_size, remainder = divmod(candidate_count, iterations)
    if elite_count > batch_size:
        raise ValueError("elite_count must not exceed the smallest batch")
    validate_finite_scalar("min_std_nm", min_std_nm)
    if min_std_nm <= 0:
        raise ValueError("min_std_nm must be positive")
    validate_prediction_inputs(state, 0.0, context)

    mean = std = None
    best_actions = best_cost = None
    feasible = calls = 0
    for round_index in range(iterations):
        size = (batch_size + (round_index < remainder), horizon)
        if mean is None:
            candidates = rng.uniform(-context.torque_limit_nm, context.torque_limit_nm, size=size)
        else:
            candidates = np.clip(rng.normal(mean, std, size=size),
                                 -context.torque_limit_nm, context.torque_limit_nm)
        result, scored = _score_batch(predictor, context, state, target_position_rad, weights, candidates)
        feasible += result.feasible_candidates
        calls += result.predictor_calls
        if result.cost is not None and (best_cost is None or result.cost < best_cost):
            best_actions, best_cost = result.actions_nm, result.cost
        if scored:
            scored.sort(key=lambda item: item[1])  # Stable: earlier candidates win ties.
            elites = candidates[[index for index, _ in scored[:elite_count]]]
            mean = elites.mean(axis=0)
            std = np.maximum(elites.std(axis=0), min_std_nm)
    return PlanResult(best_actions, best_cost, candidate_count, feasible, calls)
